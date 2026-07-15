'''
Headless TrapTagger integration for depth-estimation-traptagger.

Uses manifest.json staged by TrapTagger (see TRAPTAGGER.md). Calibration and trap
images use supplied bounding boxes from the manifest — MegaDetector is not used.
'''

import os
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from config import Config
from custom_types import (
    DepthEstimationModel,
    DetectionSamplingMethod,
    SampleFrom,
)
from sam import SAM
from utils import (
    blur_and_downsample,
    calibrate,
    calibrate_v0,
    crop,
    exception_to_str,
    get_calibration_frame_dist,
    get_extension_agnostic_path,
    imread,
    multi_file_extension_glob,
    piecewise_linear_calibration,
    resize,
)

from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
  '''Minimal image loader matching run.ImageDataset (avoids importing run.py).'''

  def __init__(self, image_paths, crop_top, crop_bottom, crop_left, crop_right, resize_shape=None):
    self.image_paths = image_paths
    self.crop_top = crop_top
    self.crop_bottom = crop_bottom
    self.crop_left = crop_left
    self.crop_right = crop_right
    self.resize_shape = resize_shape

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    img = imread(image_path)
    img = crop(img, self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
    if self.resize_shape:
      img = resize(img, self.resize_shape)
    return img, image_path

def collate_fn(batch):
  imgs = [item[0] for item in batch]
  image_paths = [item[1] for item in batch]
  return imgs, image_paths


@dataclass
class CalibState:
  calibration_map: Optional[Callable]
  farthest_calibration_frame_disp: Optional[np.ndarray]
  farthest_calibration_frame_disp_raw: Optional[np.ndarray]
  exp: float
  do_calibrate: Callable
  ok: bool
  piecewise_x: Optional[np.ndarray] = None
  piecewise_y: Optional[np.ndarray] = None


def _rebuild_calibration_map(exp, piecewise_x, piecewise_y):
  calibration_base = piecewise_linear_calibration(piecewise_x, piecewise_y, eps=1e-6)

  def calibration_map_fn(data):
    return calibration_base(np.asarray(data) ** exp) ** exp

  return calibration_map_fn


def save_calib_state(calib: CalibState, path: str) -> None:
  '''Persist a fitted CalibState for reuse across trap batches.'''
  if not calib.ok or calib.piecewise_x is None or calib.piecewise_y is None:
    return

  farthest_data = farthest_mask = None
  farthest_raw_data = farthest_raw_mask = None
  if calib.farthest_calibration_frame_disp is not None:
    farthest_data = calib.farthest_calibration_frame_disp.data
    farthest_mask = calib.farthest_calibration_frame_disp.mask
  if calib.farthest_calibration_frame_disp_raw is not None:
    farthest_raw_data = calib.farthest_calibration_frame_disp_raw.data
    farthest_raw_mask = calib.farthest_calibration_frame_disp_raw.mask

  os.makedirs(os.path.dirname(path), exist_ok=True)
  np.savez_compressed(
    path,
    ok=calib.ok,
    exp=calib.exp,
    piecewise_x=calib.piecewise_x,
    piecewise_y=calib.piecewise_y,
    farthest_data=farthest_data,
    farthest_mask=farthest_mask,
    farthest_raw_data=farthest_raw_data,
    farthest_raw_mask=farthest_raw_mask,
  )


def load_calib_state(path: str, do_calibrate: Callable) -> CalibState:
  '''Restore a CalibState saved by save_calib_state().'''
  with np.load(path, allow_pickle=False) as data:
    ok = bool(data['ok'])
    exp = float(data['exp'])
    piecewise_x = data['piecewise_x']
    piecewise_y = data['piecewise_y']
    farthest_data = data['farthest_data']
    farthest_mask = data['farthest_mask']
    farthest_raw_data = data['farthest_raw_data']
    farthest_raw_mask = data['farthest_raw_mask']

  calibration_map = _rebuild_calibration_map(exp, piecewise_x, piecewise_y)
  farthest_calibration_frame_disp = np.ma.masked_array(farthest_data, farthest_mask)
  farthest_calibration_frame_disp_raw = np.ma.masked_array(
    farthest_raw_data,
    farthest_raw_mask,
  )
  return CalibState(
    calibration_map,
    farthest_calibration_frame_disp,
    farthest_calibration_frame_disp_raw,
    exp,
    do_calibrate,
    ok,
    piecewise_x=piecewise_x,
    piecewise_y=piecewise_y,
  )


def traptagger_default_config() -> Config:
  '''Server-side defaults for TrapTagger depth jobs.'''
  return Config(
    min_depth=1.0,
    max_depth=30.0,
    batch_size=4,
    num_workers=0,
    make_figures=False,
    detect_humans=False,
    depth_estimation_model=DepthEstimationModel.DPT,
    detection_sampling_method=DetectionSamplingMethod.SAM,
    sample_from=SampleFrom.DETECTION,
  )


def _load_manifest(transect_dir: str) -> dict:
  path = os.path.join(transect_dir, 'manifest.json')
  with open(path) as f:
    return json.load(f)


def _json_box(box) -> Optional[List[float]]:
  '''Serialize a box [xmin, ymin, xmax, ymax] for bbox audit JSON.'''
  if box is None:
    return None
  return [round(float(box[0]), 4), round(float(box[1]), 4),
          round(float(box[2]), 4), round(float(box[3]), 4)]


def _image_size_hw(shape) -> List[int]:
  '''Return [height, width] from an image ndarray shape.'''
  return [int(shape[0]), int(shape[1])]


def _new_bbox_audit(manifest: dict, config: Config, transect_id: str) -> dict:
  '''Initialize bbox audit payload for a transect job.'''
  return {
    'cameragroup_id': manifest.get('cameragroup_id'),
    'cam_name': manifest.get('cam_name'),
    'transect_id': transect_id,
    'job_config': {
      'depth_estimation_model': str(config.depth_estimation_model),
      'detection_sampling_method': str(config.detection_sampling_method),
      'sample_from': str(config.sample_from),
      'crop_top': config.crop_top,
      'crop_bottom': config.crop_bottom,
      'crop_left': config.crop_left,
      'crop_right': config.crop_right,
      'resize_shape': None,
    },
    'calibration': [],
    'trap': [],
  }


def _box_from_pixels(bbox_pixels) -> np.ndarray:
  x1, y1, x2, y2 = bbox_pixels
  return np.array([float(x1), float(y1), float(x2), float(y2)], dtype=np.float32)


def _clip_box_to_image(box, img_shape) -> np.ndarray:
  box = box.copy()
  box[0] = max(0.0, min(float(img_shape[1] - 1), float(box[0])))
  box[1] = max(0.0, min(float(img_shape[0] - 1), float(box[1])))
  box[2] = max(0.0, min(float(img_shape[1] - 1), float(box[2])))
  box[3] = max(0.0, min(float(img_shape[0] - 1), float(box[3])))
  return box


def _cropped_image_shape(img_shape, config: Config):
  '''Image (height, width) after ImageDataset crop, before optional resize.'''
  height, width = img_shape[0], img_shape[1]
  crop_bottom = height if config.crop_bottom <= 0 else height - config.crop_bottom
  crop_right = width if config.crop_right <= 0 else width - config.crop_right
  return (
    max(0, crop_bottom - config.crop_top),
    max(0, crop_right - config.crop_left),
  )


def _transform_box_for_preprocessing(box, img_shape, config: Config, resize_shape=None) -> np.ndarray:
  '''
  Map bbox_pixels in original image coordinates to the crop+resize space used by
  ImageDataset (same transform as run.py MegaDetector boxes on resized imgs).
  '''
  box = box.copy()
  box[0] -= config.crop_left
  box[2] -= config.crop_left
  box[1] -= config.crop_top
  box[3] -= config.crop_top

  cropped_shape = _cropped_image_shape(img_shape, config)
  box = _clip_box_to_image(box, cropped_shape)

  if resize_shape is not None and cropped_shape[0] > 0 and cropped_shape[1] > 0:
    scale_x = float(resize_shape[1]) / float(cropped_shape[1])
    scale_y = float(resize_shape[0]) / float(cropped_shape[0])
    box[0] *= scale_x
    box[2] *= scale_x
    box[1] *= scale_y
    box[3] *= scale_y
    box = _clip_box_to_image(box, resize_shape)

  return box


def _calibration_entries_by_filename(manifest: dict) -> dict:
  lookup = {}
  for item in manifest.get('calibration', []):
    rel = item.get('relative_path')
    if rel:
      lookup[os.path.basename(rel)] = item
  return lookup


def _init_models(config: Config):
  do_calibrate = calibrate
  if config.depth_estimation_model == DepthEstimationModel.DPT:
    from dpt import DPT
    depth_estimation_model = DPT()
  elif config.depth_estimation_model == DepthEstimationModel.DPT_PYTORCH:
    from dpt_pytorch import DPTPyTorch
    depth_estimation_model = DPTPyTorch()
    do_calibrate = calibrate_v0
  elif config.depth_estimation_model == DepthEstimationModel.DEPTH_AHYTHING_METRIC:
    from depth_anything import DepthAnything
    depth_estimation_model = DepthAnything()
  elif config.depth_estimation_model == DepthEstimationModel.METRIC_3D_V2_VIT_S:
    from metric3d import Metric3D
    depth_estimation_model = Metric3D()
  elif config.depth_estimation_model == DepthEstimationModel.MONODEPTH2:
    from monodepth2 import MonoDepth2
    depth_estimation_model = MonoDepth2()
  elif config.depth_estimation_model == DepthEstimationModel.DEPTH_PRO:
    from depthpro import DepthPro
    depth_estimation_model = DepthPro()
  else:
    raise ValueError('Invalid depth estimation model {}'.format(config.depth_estimation_model))

  sam = SAM()
  return depth_estimation_model, sam, do_calibrate


def _preprocess_calibration_masks_from_manifest(
  transect_dir, manifest, config, sam, bbox_audit=None,
):
  '''
  Build calibration_frames_masks/ using TrapTagger-supplied bbox_pixels and SAM.
  '''
  calibration_dir = os.path.join(transect_dir, 'calibration_frames')
  if not os.path.isdir(calibration_dir):
    return

  calibration_frame_filenames = sorted(list(set(multi_file_extension_glob(
    os.path.join(calibration_dir, '*'),
    config.intensity_image_extensions,
  ))))
  if not calibration_frame_filenames:
    return

  masks_dir = os.path.join(transect_dir, 'calibration_frames_masks')
  os.makedirs(masks_dir, exist_ok=True)
  cal_entries = _calibration_entries_by_filename(manifest)

  for img_path in calibration_frame_filenames:
    filename = os.path.basename(img_path)
    entry = cal_entries.get(filename)
    if not entry or 'bbox_pixels' not in entry:
      logging.warning(
        'No manifest bbox for calibration image %s; skipping mask generation',
        filename,
      )
      continue

    img = imread(img_path)
    if img is None:
      logging.warning('Unable to read calibration image %s', filename)
      continue

    box = _clip_box_to_image(_box_from_pixels(entry['bbox_pixels']), img.shape)
    if box[2] <= box[0] or box[3] <= box[1]:
      logging.warning('Invalid manifest bbox for calibration image %s', filename)
      continue

    mask_path = os.path.join(masks_dir, filename)
    if os.path.isfile(mask_path):
      logging.info('Using existing calibration mask for %s', filename)
      if bbox_audit is not None:
        bbox_audit['calibration'].append({
          'known_distance': entry.get('known_distance'),
          'image_name': filename,
          'relative_path': entry.get('relative_path'),
          'image_size': _image_size_hw(img.shape),
          'bbox_pixels': _json_box(box),
          'reused_mask': True,
        })
      continue

    if bbox_audit is not None:
      bbox_audit['calibration'].append({
        'known_distance': entry.get('known_distance'),
        'image_name': filename,
        'relative_path': entry.get('relative_path'),
        'image_size': _image_size_hw(img.shape),
        'bbox_pixels': _json_box(box),
      })

    masks = sam(img, np.array([box]))
    combined_mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    for mask in masks:
      combined_mask[mask] = 255

    cv2.imwrite(mask_path, combined_mask)
    logging.info('Saved calibration mask from TrapTagger bbox for %s', filename)


def _calibrate_transect(
  transect_dir: str,
  transect_id: str,
  manifest: dict,
  config: Config,
  depth_estimation_model,
  sam,
  do_calibrate,
  bbox_audit=None,
) -> CalibState:
  eps = 1e-6
  exp = -1 if config.calibrate_metric else 1

  if config.depth_estimation_model == DepthEstimationModel.DEPTH_AHYTHING_METRIC:
    return CalibState(None, None, None, exp, do_calibrate, True)

  _preprocess_calibration_masks_from_manifest(
    transect_dir, manifest, config, sam, bbox_audit,
  )

  calibration_frames = {}
  farthest_calibration_frame_disp_raw = None
  calibration_map = None
  farthest_calibration_frame_disp = None

  calibration_frame_filenames = sorted(list(set(
    multi_file_extension_glob(
      os.path.join(transect_dir, 'calibration_frames', '*'),
      config.intensity_image_extensions,
    )
    + multi_file_extension_glob(
      os.path.join(transect_dir, 'calibration_frames_cropped', '*'),
      config.intensity_image_extensions,
    )
  )))

  if calibration_frame_filenames:
    calibration_dataset = ImageDataset(
      calibration_frame_filenames,
      config.crop_top,
      config.crop_bottom,
      config.crop_left,
      config.crop_right,
    )
    calibration_dataloader = DataLoader(
      calibration_dataset,
      batch_size=config.batch_size,
      shuffle=False,
      collate_fn=collate_fn,
      num_workers=config.num_workers,
    )

    for imgs, image_paths in calibration_dataloader:
      disps = depth_estimation_model(imgs)
      for i, disp in enumerate(disps):
        calibration_frame_filename = image_paths[i]
        calibration_frame_id = os.path.splitext(
          os.path.basename(calibration_frame_filename)
        )[0]
        dist = get_calibration_frame_dist(transect_dir, calibration_frame_id)
        mask_path = get_extension_agnostic_path(
          os.path.join(
            transect_dir,
            'calibration_frames_masks',
            calibration_frame_id,
          ),
          config.intensity_image_extensions,
        )
        mask_img = imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
        if mask_img is None:
          logging.warning(
            'Missing calibration mask for %s in transect %s',
            calibration_frame_id,
            transect_id,
          )
          continue
        mask = crop(
          mask_img > 127,
          config.crop_top,
          config.crop_bottom,
          config.crop_left,
          config.crop_right,
        )
        disp = np.ma.masked_where(mask, disp)
        calibration_frames[dist] = disp

  calibration_frames = OrderedDict(sorted(calibration_frames.items(), key=lambda kv: kv[0]))
  farthest_calibration_frame_disp_raw = (
    list(calibration_frames.values())[-1] if len(calibration_frames) > 0 else None
  )

  if farthest_calibration_frame_disp_raw is None:
    return CalibState(None, None, None, exp, do_calibrate, False)

  try:
    x, y = [], []
    for dist, disp in calibration_frames.items():
      disp = resize(disp, farthest_calibration_frame_disp_raw.shape)
      if config.calibrate_metric:
        disp = np.clip(disp, eps, np.inf)
      disp_calibrated = do_calibrate(
        disp ** exp,
        farthest_calibration_frame_disp_raw ** exp,
        config.calibration_regression_method,
      )(disp.data ** exp) ** exp
      disp_calibrated = np.ma.masked_where(disp.mask, disp_calibrated)
      x.append(float(np.median(disp_calibrated.data[disp_calibrated.mask])))
      y.append(float(dist ** -1))

    calibration_base = piecewise_linear_calibration(
      np.array(x) ** exp,
      np.array(y) ** exp,
      eps=eps,
    )
    piecewise_x = np.array(x) ** exp
    piecewise_y = np.array(y) ** exp

    def calibration_map_fn(data, exp=exp, calibration_base=calibration_base):
      return calibration_base(np.asarray(data) ** exp) ** exp

    calibration_map = calibration_map_fn
    farthest_calibration_frame_disp = np.ma.masked_where(
      farthest_calibration_frame_disp_raw.mask,
      calibration_map_fn(farthest_calibration_frame_disp_raw.data),
    )
    return CalibState(
      calibration_map,
      farthest_calibration_frame_disp,
      farthest_calibration_frame_disp_raw,
      exp,
      do_calibrate,
      True,
      piecewise_x=piecewise_x,
      piecewise_y=piecewise_y,
    )
  except Exception as e:
    logging.warning(
      'Failed calibrating transect %s: %s',
      transect_id,
      exception_to_str(e),
    )
    return CalibState(None, None, None, exp, do_calibrate, False)


def _animal_mask_from_box(img_shape, box) -> np.ndarray:
  animal_mask = np.zeros(img_shape[0:2], dtype=bool)
  ymin = max(0, min(img_shape[0] - 2, round(box[1])))
  ymax = max(0, min(img_shape[0] - 1, round(box[3])))
  xmin = max(0, min(img_shape[1] - 2, round(box[0])))
  xmax = max(0, min(img_shape[1] - 1, round(box[2])))
  animal_mask[ymin:ymax, xmin:xmax] = True
  return animal_mask


def _sample_depth_at_box(depth, box, mask, config: Config) -> float:
  if box[2] <= box[0] or box[3] <= box[1]:
    raise ValueError('invalid bbox')

  if config.detection_sampling_method == DetectionSamplingMethod.BBOX_BOTTOM:
    sample_location = (
      max(0, min(depth.shape[0] - 1, round(box[3]))),
      max(0, min(depth.shape[1] - 1, round(box[0] + (box[2] - box[0]) / 2))),
    )
    return float(depth[sample_location])

  if config.detection_sampling_method == DetectionSamplingMethod.BBOX_PERCENTILE:
    ymin = max(0, min(depth.shape[0] - 2, round(box[1])))
    ymax = max(0, min(depth.shape[0] - 1, round(box[3])))
    xmin = max(0, min(depth.shape[1] - 2, round(box[0])))
    xmax = max(0, min(depth.shape[1] - 1, round(box[2])))
    depth_cropped = depth[ymin:ymax, xmin:xmax]
    sampled = np.percentile(
      depth_cropped,
      config.bbox_sampling_percentile,
      method='nearest',
    )
    return float(sampled)

  if config.detection_sampling_method == DetectionSamplingMethod.SAM:
    if mask is None:
      raise ValueError('SAM sampling requires a mask')
    ymin = max(0, min(depth.shape[0] - 2, round(box[1])))
    ymax = max(0, min(depth.shape[0] - 1, round(box[3])))
    xmin = max(0, min(depth.shape[1] - 2, round(box[0])))
    xmax = max(0, min(depth.shape[1] - 1, round(box[2])))
    depth_cropped = depth[ymin:ymax, xmin:xmax]
    mask_cropped = mask[ymin:ymax, xmin:xmax]
    mask_padded = np.pad(mask_cropped, ((1, 1), (1, 1)))
    dist = cv2.distanceTransform(
      (mask_padded * 255).astype(np.uint8),
      cv2.DIST_L2,
      cv2.DIST_MASK_3,
    )
    sample_location = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    sample_location = (
      max(0, min(mask_cropped.shape[0], sample_location[0] - 1)),
      max(0, min(mask_cropped.shape[1], sample_location[1] - 1)),
    )
    return float(depth_cropped[sample_location[0], sample_location[1]])

  raise RuntimeError(
    'Invalid detection_sampling_method {}'.format(config.detection_sampling_method)
  )


def _compute_disps_for_images(
  imgs: List[np.ndarray],
  config: Config,
  depth_estimation_model,
  calib: CalibState,
  animal_masks: List[np.ndarray],
) -> List[Optional[np.ndarray]]:
  eps = 1e-6
  exp = calib.exp
  do_calibrate = calib.do_calibrate

  if config.depth_estimation_model in [
    DepthEstimationModel.DEPTH_AHYTHING_METRIC,
    DepthEstimationModel.DEPTH_PRO,
  ]:
    depths = depth_estimation_model(imgs)
    return [np.clip(depth, config.min_depth, config.max_depth) ** -1 for depth in depths]

  if calib.farthest_calibration_frame_disp is None or calib.calibration_map is None:
    return [None for _ in imgs]

  disps = [None for _ in imgs]
  computed_disps = depth_estimation_model(imgs)
  for i, disp in enumerate(computed_disps):
    if config.calibrate_metric:
      disp = np.clip(disp, eps, np.inf)

    mask = (calib.farthest_calibration_frame_disp ** -1) >= (config.max_depth - eps)
    if config.calibration_mask_animals:
      mask = mask | animal_masks[i]
    disp_masked = np.ma.masked_where(mask, disp ** exp)
    farthest_disp_raw_masked = np.ma.masked_where(
      mask,
      calib.farthest_calibration_frame_disp_raw ** exp,
    )

    if config.calibrate_blur:
      disp_aligned = do_calibrate(
        blur_and_downsample(disp_masked),
        blur_and_downsample(farthest_disp_raw_masked),
        config.calibration_regression_method,
      )(disp ** exp) ** exp
    else:
      disp_aligned = do_calibrate(
        disp_masked,
        farthest_disp_raw_masked,
        config.calibration_regression_method,
      )(disp ** exp) ** exp
    disps[i] = calib.calibration_map(disp_aligned)

  return disps


def _estimate_trap_detections(
  transect_dir: str,
  manifest: dict,
  config: Config,
  depth_estimation_model,
  sam,
  calib: CalibState,
  bbox_audit=None,
) -> Dict[str, dict]:
  results = {}
  trap_entries = manifest.get('trap', [])

  for entry in trap_entries:
    det_id = str(entry['detection_id'])
    if not entry.get('download_ok', True):
      results[det_id] = {'distance': None, 'error': 's3_download_failed'}
      if bbox_audit is not None:
        bbox_audit['trap'].append({
          'detection_id': entry['detection_id'],
          'relative_path': entry.get('relative_path'),
          'error': 's3_download_failed',
        })
    elif 'bbox_pixels' not in entry:
      results[det_id] = {'distance': None, 'error': 'missing_bbox'}
      if bbox_audit is not None:
        bbox_audit['trap'].append({
          'detection_id': entry['detection_id'],
          'relative_path': entry.get('relative_path'),
          'error': 'missing_bbox',
        })

  valid_entries = [
    e for e in trap_entries
    if e.get('download_ok', True) and 'bbox_pixels' in e
  ]

  if not valid_entries:
    return results

  if (
    config.depth_estimation_model not in [
      DepthEstimationModel.DEPTH_AHYTHING_METRIC,
      DepthEstimationModel.DEPTH_PRO,
    ]
    and not calib.ok
  ):
    for entry in valid_entries:
      results[str(entry['detection_id'])] = {
        'distance': None,
        'error': 'calibration_failed',
      }
      if bbox_audit is not None:
        bbox_audit['trap'].append({
          'detection_id': entry['detection_id'],
          'relative_path': entry.get('relative_path'),
          'bbox_pixels_staged': _json_box(_box_from_pixels(entry['bbox_pixels'])),
          'error': 'calibration_failed',
        })
    return results

  resize_shape = (
    calib.farthest_calibration_frame_disp.shape
    if calib.farthest_calibration_frame_disp is not None
    else None
  )

  for i in range(0, len(valid_entries), config.batch_size):
    batch_entries = valid_entries[i:i + config.batch_size]
    image_paths = [os.path.join(transect_dir, e['relative_path']) for e in batch_entries]
    batch_boxes_orig = [_box_from_pixels(e['bbox_pixels']) for e in batch_entries]

    detection_dataset = ImageDataset(
      image_paths,
      config.crop_top,
      config.crop_bottom,
      config.crop_left,
      config.crop_right,
      resize_shape=resize_shape,
    )
    detection_dataloader = DataLoader(
      detection_dataset,
      batch_size=len(batch_entries),
      shuffle=False,
      collate_fn=collate_fn,
      num_workers=config.num_workers,
    )

    for imgs, batch_paths in detection_dataloader:
      batch_boxes = []
      for j, image_path in enumerate(batch_paths):
        orig_img = imread(image_path)
        if orig_img is None:
          batch_boxes.append(None)
          continue
        box = _transform_box_for_preprocessing(
          _clip_box_to_image(batch_boxes_orig[j], orig_img.shape),
          orig_img.shape,
          config,
          resize_shape,
        )
        batch_boxes.append(_clip_box_to_image(box, imgs[j].shape))

      batch_animal_masks = [
        _animal_mask_from_box(imgs[j].shape, batch_boxes[j])
        if batch_boxes[j] is not None
        else np.zeros(imgs[j].shape[0:2], dtype=bool)
        for j in range(len(imgs))
      ]

      if config.detection_sampling_method == DetectionSamplingMethod.SAM:
        batch_masks = [[None] for _ in batch_boxes]
        sam_imgs = []
        sam_boxes = []
        sam_indices = []
        for j in range(len(imgs)):
          if batch_boxes[j] is None:
            continue
          if batch_boxes[j][2] <= batch_boxes[j][0] or batch_boxes[j][3] <= batch_boxes[j][1]:
            continue
          sam_imgs.append(imgs[j])
          sam_boxes.append([batch_boxes[j]])
          sam_indices.append(j)
        if sam_imgs:
          sam_results = sam(sam_imgs, sam_boxes)
          for idx, masks in zip(sam_indices, sam_results):
            batch_masks[idx] = masks
      else:
        batch_masks = [[None] for _ in batch_boxes]

      disps = _compute_disps_for_images(
        imgs,
        config,
        depth_estimation_model,
        calib,
        batch_animal_masks,
      )

      for j, entry in enumerate(batch_entries):
        det_id = str(entry['detection_id'])
        image_path = batch_paths[j]
        orig_img = imread(image_path)
        staged_size = _image_size_hw(orig_img.shape) if orig_img is not None else None
        model_input_size = _image_size_hw(imgs[j].shape)

        trap_audit = {
          'detection_id': entry['detection_id'],
          'image_name': os.path.basename(image_path),
          'relative_path': entry.get('relative_path'),
          'staged_image_size': staged_size,
          'model_input_size': model_input_size,
          'bbox_pixels_staged': _json_box(batch_boxes_orig[j]),
          'bbox_pixels_model_input': None,
          'distance': None,
          'error': None,
        }

        if batch_boxes[j] is None:
          trap_audit['error'] = 'image_read_failed'
          results[det_id] = {'distance': None, 'error': 'image_read_failed'}
          if bbox_audit is not None:
            bbox_audit['trap'].append(trap_audit)
          continue
        trap_audit['bbox_pixels_model_input'] = _json_box(batch_boxes[j])

        if batch_boxes[j][2] <= batch_boxes[j][0] or batch_boxes[j][3] <= batch_boxes[j][1]:
          trap_audit['error'] = 'invalid_bbox'
          results[det_id] = {'distance': None, 'error': 'invalid_bbox'}
          if bbox_audit is not None:
            bbox_audit['trap'].append(trap_audit)
          continue

        disp = disps[j]
        if disp is None:
          trap_audit['error'] = 'depth_estimation_failed'
          results[det_id] = {'distance': None, 'error': 'depth_estimation_failed'}
          if bbox_audit is not None:
            bbox_audit['trap'].append(trap_audit)
          continue

        try:
          if config.sample_from == SampleFrom.REFERENCE:
            disp = calib.farthest_calibration_frame_disp

          depth = np.clip(disp, config.max_depth ** -1, config.min_depth ** -1) ** -1
          box = batch_boxes[j]
          mask = batch_masks[j][0]
          sampled_depth = _sample_depth_at_box(depth, box, mask, config)
          trap_audit['distance'] = round(float(sampled_depth), 4)
          results[det_id] = {'distance': trap_audit['distance'], 'error': None}
        except Exception as e:
          logging.exception('Trap detection %s failed', det_id)
          trap_audit['error'] = str(e)
          results[det_id] = {'distance': None, 'error': str(e)}

        if bbox_audit is not None:
          bbox_audit['trap'].append(trap_audit)

  return results


def run_transect_job(
  data_dir: str,
  transect_id: str,
  config: Optional[Config] = None,
  collect_bbox_audit: bool = False,
  cached_calib_path: Optional[str] = None,
  calib_cache_path: Optional[str] = None,
) -> dict:
  '''
  Headless TrapTagger integration for a single transect (one cameragroup).

  data_dir: job root containing transects/ and results/
  transect_id: folder name under transects/ (sanitized cam name)
  collect_bbox_audit: when True, include a 'bbox_audit' key in the return value
  cached_calib_path: optional path to a saved CalibState npz (skips calibration)
  calib_cache_path: optional path to write CalibState after fitting (batch 1)

  Returns:
    {str(detection_id): {"distance": float|None, "error": str|None}}
    When collect_bbox_audit is True, also includes key 'bbox_audit' (dict).
  '''
  config = config or traptagger_default_config()
  config.data_dir = data_dir
  os.makedirs(os.path.join(data_dir, 'results'), exist_ok=True)

  transect_dir = os.path.join(data_dir, 'transects', transect_id)
  if not os.path.isdir(transect_dir):
    raise FileNotFoundError('Transect directory not found: {}'.format(transect_dir))

  manifest = _load_manifest(transect_dir)
  bbox_audit = _new_bbox_audit(manifest, config, transect_id) if collect_bbox_audit else None

  depth_estimation_model, sam, do_calibrate = _init_models(config)
  if cached_calib_path and os.path.isfile(cached_calib_path):
    calib = load_calib_state(cached_calib_path, do_calibrate)
    logging.info(
      'Reusing cached calibration state for transect %s from %s',
      transect_id,
      cached_calib_path,
    )
  else:
    calib = _calibrate_transect(
      transect_dir,
      transect_id,
      manifest,
      config,
      depth_estimation_model,
      sam,
      do_calibrate,
      bbox_audit=bbox_audit,
    )
    if calib_cache_path and calib.ok:
      save_calib_state(calib, calib_cache_path)
      logging.info(
        'Saved calibration state for transect %s to %s',
        transect_id,
        calib_cache_path,
      )

  if bbox_audit is not None:
    resize_shape = (
      list(calib.farthest_calibration_frame_disp.shape)
      if calib.farthest_calibration_frame_disp is not None
      else None
    )
    bbox_audit['job_config']['resize_shape'] = resize_shape

  results = _estimate_trap_detections(
    transect_dir,
    manifest,
    config,
    depth_estimation_model,
    sam,
    calib,
    bbox_audit=bbox_audit,
  )

  if bbox_audit is not None:
    return {'results': results, 'bbox_audit': bbox_audit}
  return results
