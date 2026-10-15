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
    METRIC_DEPTH_ESTIMATION_MODELS,
    DepthEstimationModel,
    DetectionSamplingMethod,
    MetricCalibrationMethod,
    SampleFrom,
)
from sam import SAM
from utils import (
    blur_and_downsample,
    calibrate,
    calibrate_metric_model,
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
from registration import ImageRegistration, info_bar_mask, warp_to_reference

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
  metric_pred_depths: Optional[np.ndarray] = None
  metric_true_depths: Optional[np.ndarray] = None
  metric_calibration_method: Optional[MetricCalibrationMethod] = None
  metric_correct: Optional[Callable] = None
  # reference calibration image (after crop) and the pixels of it that must not
  # take part in depth alignment (calibration subject, info strip)
  reference_image: Optional[np.ndarray] = None
  reference_ignore: Optional[np.ndarray] = None
  # reference pixels at or beyond max_depth (sky), excluded from alignment
  reference_far_mask: Optional[np.ndarray] = None
  registration: Optional[ImageRegistration] = None


def _rebuild_calibration_map(exp, piecewise_x, piecewise_y):
  calibration_base = piecewise_linear_calibration(piecewise_x, piecewise_y, eps=1e-6)

  def calibration_map_fn(data):
    return calibration_base(np.asarray(data) ** exp) ** exp

  return calibration_map_fn


def _has_relative_calibration(calib: CalibState) -> bool:
  return calib.piecewise_x is not None and calib.piecewise_y is not None and np.size(calib.piecewise_x) > 0


def _has_metric_calibration(calib: CalibState) -> bool:
  return (
    calib.metric_pred_depths is not None
    and calib.metric_true_depths is not None
    and np.size(calib.metric_pred_depths) > 0
    and calib.metric_calibration_method is not None
    and calib.metric_calibration_method != MetricCalibrationMethod.NONE
  )


def _npz_array(value):
  '''Store a missing frame as an empty numeric array.

  np.savez turns None into an object array, and load_calib_state opens the
  cache with allow_pickle=False. Metric calibration leaves the corrected
  disparity frame unset, so those fields must not be None.
  '''
  if value is None:
    return np.zeros(0, dtype=np.float32)
  return value


def save_calib_state(calib: CalibState, path: str) -> None:
  '''Persist a fitted CalibState for reuse across trap batches.'''
  if not calib.ok or not (_has_relative_calibration(calib) or _has_metric_calibration(calib)):
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
    piecewise_x=np.asarray([] if calib.piecewise_x is None else calib.piecewise_x),
    piecewise_y=np.asarray([] if calib.piecewise_y is None else calib.piecewise_y),
    farthest_data=_npz_array(farthest_data),
    farthest_mask=_npz_array(farthest_mask),
    farthest_raw_data=_npz_array(farthest_raw_data),
    farthest_raw_mask=_npz_array(farthest_raw_mask),
    metric_pred_depths=np.asarray([] if calib.metric_pred_depths is None else calib.metric_pred_depths),
    metric_true_depths=np.asarray([] if calib.metric_true_depths is None else calib.metric_true_depths),
    metric_calibration_method=np.array(
      '' if calib.metric_calibration_method is None else calib.metric_calibration_method.name
    ),
    reference_image=_npz_array(calib.reference_image),
    reference_ignore=_npz_array(calib.reference_ignore),
  )


def _rebuild_metric_correct(pred_depths, true_depths, method, exp):
  if method is None or method == MetricCalibrationMethod.NONE or np.size(pred_depths) == 0:
    return None
  return calibrate_metric_model(pred_depths, true_depths, method, exp=exp)


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
    files = set(data.files)
    if 'metric_pred_depths' in files and np.size(data['metric_pred_depths']) > 0:
      metric_pred_depths = np.asarray(data['metric_pred_depths'])
      metric_true_depths = np.asarray(data['metric_true_depths'])
      method_name = str(np.array(data['metric_calibration_method']).item())
      metric_method = MetricCalibrationMethod[method_name] if method_name else None
    else:
      metric_pred_depths = None
      metric_true_depths = None
      metric_method = None
    reference_image = None
    reference_ignore = None
    if 'reference_image' in files and np.size(data['reference_image']) > 0:
      reference_image = np.asarray(data['reference_image'], dtype=np.uint8)
      reference_ignore = np.asarray(data['reference_ignore']).astype(bool)

  has_relative = piecewise_x is not None and np.size(piecewise_x) > 0
  if has_relative:
    calibration_map = _rebuild_calibration_map(exp, piecewise_x, piecewise_y)
    farthest_calibration_frame_disp = np.ma.masked_array(farthest_data, farthest_mask)
    farthest_calibration_frame_disp_raw = np.ma.masked_array(
      farthest_raw_data,
      farthest_raw_mask,
    )
  else:
    calibration_map = None
    # Leave the corrected disparity frame unset so trap photos are not resized
    # to the calibration frame. The raw reference is what metric alignment uses.
    farthest_calibration_frame_disp = None
    farthest_calibration_frame_disp_raw = _restore_masked_frame(
      farthest_raw_data,
      farthest_raw_mask,
    )
    piecewise_x = None
    piecewise_y = None

  return CalibState(
    calibration_map,
    farthest_calibration_frame_disp,
    farthest_calibration_frame_disp_raw,
    exp,
    do_calibrate,
    ok,
    piecewise_x=piecewise_x,
    piecewise_y=piecewise_y,
    metric_pred_depths=metric_pred_depths,
    metric_true_depths=metric_true_depths,
    metric_calibration_method=metric_method,
    metric_correct=_rebuild_metric_correct(
      metric_pred_depths, metric_true_depths, metric_method, exp,
    ),
    reference_image=reference_image,
    reference_ignore=reference_ignore,
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
    depth_estimation_model=DepthEstimationModel.UNIDEPTH_V2_LARGE,
    metric_calibration_method=MetricCalibrationMethod.PIECEWISE_LINEAR,
    detection_sampling_method=DetectionSamplingMethod.SAM,
    sample_from=SampleFrom.DETECTION,
    register_to_reference=True,
    mask_info_bar=True,
    calibration_mask_animals=True,
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
  elif config.depth_estimation_model == DepthEstimationModel.UNIDEPTH_V2:
    from unidepthv2 import UniDepthv2
    depth_estimation_model = UniDepthv2()
  elif config.depth_estimation_model == DepthEstimationModel.UNIDEPTH_V2_LARGE:
    from unidepthv2_large import UniDepthv2Large
    depth_estimation_model = UniDepthv2Large()
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


def _predict_metric_depths(depth_estimation_model, imgs):
  '''Run a metric depth model on a batch. Fall back to one image at a time.'''
  try:
    out = depth_estimation_model(imgs)
    if isinstance(out, list) and len(out) == len(imgs):
      return out
  except Exception:
    logging.exception('Batched metric depth inference failed; retrying per image')
  return [depth_estimation_model(img) for img in imgs]


def _subject_mask_for_depth(mask_img, config: Config, depth_shape):
  mask = crop(
    mask_img > 127,
    config.crop_top,
    config.crop_bottom,
    config.crop_left,
    config.crop_right,
  )
  if mask.shape[0:2] != depth_shape[0:2]:
    mask = resize(mask.astype(np.uint8) * 255, depth_shape) > 127
  return mask


def _known_calibration_distance(entry, transect_dir, calibration_frame_id):
  if entry and entry.get('known_distance') is not None:
    return float(entry['known_distance'])
  return float(get_calibration_frame_dist(transect_dir, calibration_frame_id))


def _metric_fit_space(metres, exp, eps):
  '''
  DPT aligns disparity. With exp == 1 that is inverse depth; with exp == -1 it
  is metres. Metric maps are stored in metres, so convert into that same space.
  '''
  metres = np.clip(np.asarray(metres, dtype=np.float32), eps, np.inf)
  if exp == 1:
    return metres ** -1
  return metres


def _metres_from_fit_space(fitted, exp, eps):
  '''Convert an aligned map back to metres.'''
  fitted = np.clip(np.asarray(fitted, dtype=np.float32), eps, np.inf)
  if exp == 1:
    return fitted ** -1
  return fitted


def _align_metric_depth(depth, exclude_mask, reference, do_calibrate, regression_method, exp, eps=1e-6):
  '''
  Line this metric depth map up with a reference calibration frame.

  The fit uses background pixels only. exclude_mask marks the subject and is
  left out of the fit, then the fitted scale and shift are applied to every
  pixel. Returns metric depth in metres.
  '''
  ref_data = np.clip(np.asarray(np.ma.getdata(reference), dtype=np.float32), eps, np.inf)
  ref_mask = np.ma.getmaskarray(reference)
  if ref_mask.shape != ref_data.shape:
    ref_mask = np.zeros(ref_data.shape[:2], dtype=bool)

  depth_m = np.clip(np.asarray(depth, dtype=np.float32), eps, np.inf)
  depth_small = np.asarray(resize(depth_m, ref_data.shape), dtype=np.float32)
  if exclude_mask is None:
    exclude_small = np.zeros(ref_data.shape[:2], dtype=bool)
  else:
    exclude_u8 = (np.asarray(exclude_mask) > 0).astype(np.uint8) * 255
    exclude_small = np.asarray(resize(exclude_u8, ref_data.shape)) > 127

  src_fit = _metric_fit_space(depth_small, exp, eps)
  ref_fit = _metric_fit_space(ref_data, exp, eps)
  src_masked = np.ma.masked_where(exclude_small | ref_mask, src_fit)
  ref_masked = np.ma.masked_where(ref_mask, ref_fit)
  align = do_calibrate(src_masked, ref_masked, regression_method)
  aligned_fit = align(_metric_fit_space(depth_m, exp, eps))
  return _metres_from_fit_space(aligned_fit, exp, eps)


def _restore_masked_frame(data, mask):
  '''Rebuild a masked calibration frame from arrays stored in an npz cache.'''
  if data is None:
    return None
  arr = np.asarray(data)
  if arr.dtype == object or arr.ndim < 2 or arr.size == 0:
    return None
  if mask is None:
    mask_arr = np.zeros(arr.shape, dtype=bool)
  else:
    mask_arr = np.asarray(mask).astype(bool)
    if mask_arr.shape != arr.shape:
      mask_arr = np.zeros(arr.shape, dtype=bool)
  return np.ma.masked_array(arr.astype(np.float32), mask_arr)


def _ignore_mask(img, subject_mask, config: Config) -> np.ndarray:
  '''Pixels of a frame excluded from depth alignment: the subject and, optionally, the info strip.'''
  if subject_mask is None:
    ignore = np.zeros(img.shape[0:2], dtype=bool)
  else:
    ignore = np.asarray(subject_mask, dtype=bool)
    if ignore.shape != img.shape[0:2]:
      ignore = resize(ignore.astype(np.uint8) * 255, img.shape[0:2]) > 127
  if config.mask_info_bar:
    ignore = ignore | info_bar_mask(img)
  return ignore


def _register(calib: CalibState, img, config: Config):
  '''Homography from img to the reference calibration frame, or None.'''
  if not config.register_to_reference or calib.reference_image is None:
    return None
  if calib.registration is None:
    calib.registration = ImageRegistration(calib.reference_image)
  H = calib.registration(img)
  if H is None:
    logging.info('Could not register a frame to the reference calibration frame; assuming the camera did not move')
  return H


def _reference_far_mask(calib: CalibState, config: Config, eps=1e-6) -> Optional[np.ndarray]:
  '''Reference pixels at or beyond max_depth after calibration (sky, far background).'''
  if calib.reference_far_mask is not None:
    return calib.reference_far_mask
  if calib.farthest_calibration_frame_disp is not None:
    disp = np.clip(np.asarray(np.ma.getdata(calib.farthest_calibration_frame_disp), dtype=np.float64), eps, np.inf)
    far = disp ** -1 >= config.max_depth - eps
  elif calib.farthest_calibration_frame_disp_raw is not None:
    depth = np.asarray(np.ma.getdata(calib.farthest_calibration_frame_disp_raw), dtype=np.float64)
    if calib.metric_correct is not None:
      depth = calib.metric_correct(depth)
    far = depth >= config.max_depth - eps
  else:
    return None
  calib.reference_far_mask = far
  return far


def _fit_to_reference(src_fit, img, subject_mask, ref_fit, calib: CalibState, config: Config, do_calibrate):
  '''
  Fit the per-image scale and shift mapping src_fit onto the reference frame.

  src_fit is a depth or disparity map of img in its own pixel coordinates.
  The frame is registered to the reference calibration frame and warped into
  its coordinates so that the regression compares the same scene points even
  if the camera moved after calibration. Without a registration the map is
  resized onto the reference, as before. Returns the fitted callable, which is
  applied to maps in img's own coordinates.
  '''
  ref_data = np.asarray(np.ma.getdata(ref_fit), dtype=np.float32)
  ref_shape = ref_data.shape[0:2]
  src_fit = np.asarray(src_fit, dtype=np.float32)
  ignore = _ignore_mask(img, subject_mask, config).astype(np.uint8)

  H = _register(calib, img, config)
  if H is not None:
    src_ref = warp_to_reference(src_fit, H, ref_shape)
    ignore_ref = warp_to_reference(ignore, H, ref_shape, nearest=True, border_value=1) > 0
  else:
    src_ref = np.asarray(resize(src_fit, ref_shape), dtype=np.float32)
    ignore_ref = resize(ignore * 255, ref_shape) > 127

  valid = ~ignore_ref & ~np.ma.getmaskarray(ref_fit) & np.isfinite(src_ref) & np.isfinite(ref_data)
  if calib.reference_ignore is not None and calib.reference_ignore.shape == ref_shape:
    valid &= ~calib.reference_ignore
  far = _reference_far_mask(calib, config)
  if far is not None and far.shape == ref_shape:
    valid &= ~far

  return do_calibrate(
    np.ma.masked_where(~valid, src_ref),
    np.ma.masked_where(~valid, ref_data),
    config.calibration_regression_method,
  )


def _calibrate_metric_transect(
  transect_dir, transect_id, manifest, config, depth_estimation_model, sam,
  do_calibrate, bbox_audit=None,
) -> CalibState:
  '''Fit one transect-level correction from known-distance calibration frames.'''
  eps = 1e-6
  exp = -1 if config.calibrate_metric else 1
  _preprocess_calibration_masks_from_manifest(
    transect_dir, manifest, config, sam, bbox_audit,
  )

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
  cal_entries = _calibration_entries_by_filename(manifest)
  frames = []

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
      depths = _predict_metric_depths(depth_estimation_model, imgs)
      for i, depth in enumerate(depths):
        calibration_frame_id = os.path.splitext(os.path.basename(image_paths[i]))[0]
        entry = cal_entries.get(os.path.basename(image_paths[i]))
        try:
          dist = _known_calibration_distance(entry, transect_dir, calibration_frame_id)
        except Exception as e:
          logging.warning(
            'Skipping calibration frame %s in transect %s: %s',
            calibration_frame_id, transect_id, exception_to_str(e),
          )
          continue
        mask_path = get_extension_agnostic_path(
          os.path.join(transect_dir, 'calibration_frames_masks', calibration_frame_id),
          config.intensity_image_extensions,
        )
        mask_img = imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
        if mask_img is None:
          logging.warning(
            'Missing calibration mask for %s in transect %s',
            calibration_frame_id, transect_id,
          )
          continue
        mask = _subject_mask_for_depth(mask_img, config, np.asarray(depth).shape)
        if not np.any(mask):
          logging.warning(
            'No predicted depth inside the subject mask for %s in transect %s',
            calibration_frame_id, transect_id,
          )
          continue
        frames.append((dist, np.asarray(depth, dtype=np.float32), mask, imgs[i]))

  if not frames:
    logging.warning('No usable metric calibration points for transect %s', transect_id)
    return CalibState(None, None, None, exp, do_calibrate, False)

  farthest_dist, farthest_depth, farthest_mask, farthest_img = max(frames, key=lambda item: item[0])
  farthest_raw = np.ma.masked_where(farthest_mask, farthest_depth)
  pred_by_dist = {}
  for dist, depth, mask, img in frames:
    try:
      aligned = _align_metric_depth(
        depth,
        _ignore_mask(img, mask, config),
        farthest_raw,
        do_calibrate,
        config.calibration_regression_method,
        exp,
        eps=eps,
      )
    except Exception as e:
      logging.warning(
        'Reference alignment failed for %.1fm calibration frame in transect %s: %s',
        dist, transect_id, exception_to_str(e),
      )
      aligned = np.asarray(depth, dtype=np.float32)
    subject_depths = np.asarray(aligned)[mask]
    subject_depths = subject_depths[np.isfinite(subject_depths) & (subject_depths > eps)]
    if subject_depths.size == 0:
      logging.warning(
        'No predicted depth inside the subject mask for %.1fm frame in transect %s',
        dist, transect_id,
      )
      continue
    pred_by_dist[dist] = float(np.median(subject_depths))

  pred_by_dist = OrderedDict(sorted(pred_by_dist.items(), key=lambda kv: kv[0]))
  if not pred_by_dist:
    logging.warning('No usable metric calibration points for transect %s', transect_id)
    return CalibState(None, None, None, exp, do_calibrate, False)

  true_depths = np.array(list(pred_by_dist.keys()), dtype=np.float64)
  pred_depths = np.array(list(pred_by_dist.values()), dtype=np.float64)
  try:
    metric_correct = calibrate_metric_model(
      pred_depths,
      true_depths,
      config.metric_calibration_method,
      exp=exp,
      eps=eps,
    )
  except Exception as e:
    logging.warning(
      'Failed metric calibration of transect %s: %s',
      transect_id, exception_to_str(e),
    )
    return CalibState(None, None, None, exp, do_calibrate, False)

  logging.info(
    'Metric reference frame for transect %s is the %.1fm calibration image',
    transect_id, farthest_dist,
  )
  return CalibState(
    None, None, farthest_raw, exp, do_calibrate, True,
    metric_pred_depths=pred_depths,
    metric_true_depths=true_depths,
    metric_calibration_method=config.metric_calibration_method,
    metric_correct=metric_correct,
    reference_image=farthest_img,
    reference_ignore=_ignore_mask(farthest_img, farthest_mask, config),
  )


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

  if config.depth_estimation_model in METRIC_DEPTH_ESTIMATION_MODELS:
    if config.metric_calibration_method == MetricCalibrationMethod.NONE:
      return CalibState(None, None, None, exp, do_calibrate, True)
    return _calibrate_metric_transect(
      transect_dir, transect_id, manifest, config, depth_estimation_model, sam,
      do_calibrate, bbox_audit,
    )

  _preprocess_calibration_masks_from_manifest(
    transect_dir, manifest, config, sam, bbox_audit,
  )

  calibration_frames = {}
  calibration_images = {}
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
        calibration_images[dist] = imgs[i]

  calibration_frames = OrderedDict(sorted(calibration_frames.items(), key=lambda kv: kv[0]))
  farthest_calibration_frame_disp_raw = (
    list(calibration_frames.values())[-1] if len(calibration_frames) > 0 else None
  )

  if farthest_calibration_frame_disp_raw is None:
    return CalibState(None, None, None, exp, do_calibrate, False)

  try:
    reference_shape = farthest_calibration_frame_disp_raw.shape
    reference_img = calibration_images[list(calibration_frames.keys())[-1]]
    reference_ignore = _ignore_mask(reference_img, farthest_calibration_frame_disp_raw.mask, config)
    x, y = [], []
    for dist, disp in calibration_frames.items():
      disp = resize(disp, reference_shape)
      if config.calibrate_metric:
        disp = np.clip(disp, eps, np.inf)
      fit_src = disp ** exp
      fit_ref = farthest_calibration_frame_disp_raw ** exp
      if config.mask_info_bar:
        bar = resize(info_bar_mask(calibration_images[dist]).astype(np.uint8) * 255, reference_shape) > 127
        fit_src = np.ma.masked_where(np.ma.getmaskarray(fit_src) | bar, fit_src)
        fit_ref = np.ma.masked_where(reference_ignore, fit_ref)
      disp_calibrated = do_calibrate(
        fit_src,
        fit_ref,
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
      reference_image=reference_img,
      reference_ignore=reference_ignore,
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

  if config.depth_estimation_model in METRIC_DEPTH_ESTIMATION_MODELS:
    depths = _predict_metric_depths(depth_estimation_model, imgs)
    reference = calib.farthest_calibration_frame_disp_raw
    if reference is not None:
      aligned_depths = []
      for img, depth, animal_mask in zip(imgs, depths, animal_masks):
        try:
          if config.register_to_reference:
            reference_fit = np.ma.masked_array(
              _metric_fit_space(np.ma.getdata(reference), exp, eps),
              np.ma.getmaskarray(reference),
            )
            depth_fit = _metric_fit_space(depth, exp, eps)
            align = _fit_to_reference(
              depth_fit, img, animal_mask, reference_fit, calib, config, do_calibrate,
            )
            aligned_depths.append(_metres_from_fit_space(align(depth_fit), exp, eps))
            continue
          aligned_depths.append(_align_metric_depth(
            depth,
            animal_mask,
            reference,
            do_calibrate,
            config.calibration_regression_method,
            exp,
            eps=eps,
          ))
        except Exception as e:
          logging.warning(
            'Reference alignment failed for a trap image: %s',
            exception_to_str(e),
          )
          aligned_depths.append(depth)
      depths = aligned_depths
    if calib.metric_correct is not None:
      depths = [calib.metric_correct(depth) for depth in depths]
    return [np.clip(depth, config.min_depth, config.max_depth) ** -1 for depth in depths]

  if calib.farthest_calibration_frame_disp is None or calib.calibration_map is None:
    return [None for _ in imgs]

  disps = [None for _ in imgs]
  computed_disps = depth_estimation_model(imgs)
  for i, disp in enumerate(computed_disps):
    if config.calibrate_metric:
      disp = np.clip(disp, eps, np.inf)

    if config.register_to_reference:
      reference_fit = np.ma.masked_array(
        np.ma.getdata(calib.farthest_calibration_frame_disp_raw) ** exp,
        np.ma.getmaskarray(calib.farthest_calibration_frame_disp_raw),
      )
      try:
        align = _fit_to_reference(
          disp ** exp,
          imgs[i],
          animal_masks[i] if config.calibration_mask_animals else None,
          reference_fit,
          calib,
          config,
          do_calibrate,
        )
      except Exception as e:
        logging.warning('Reference alignment failed for a trap image: %s', exception_to_str(e))
        continue
      disps[i] = calib.calibration_map(align(disp ** exp) ** exp)
      continue

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

  uses_metric_model = config.depth_estimation_model in METRIC_DEPTH_ESTIMATION_MODELS
  needs_metric_calibration = (
    uses_metric_model
    and config.metric_calibration_method != MetricCalibrationMethod.NONE
  )
  if needs_metric_calibration and (not calib.ok or calib.metric_correct is None):
    calibration_failed = True
  elif (not uses_metric_model) and not calib.ok:
    calibration_failed = True
  else:
    calibration_failed = False

  if calibration_failed:
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

  # Registered frames are processed at their own size and aspect ratio. Only
  # the legacy pixel-to-pixel alignment stretches them onto the reference.
  resize_shape = (
    calib.farthest_calibration_frame_disp.shape
    if calib.farthest_calibration_frame_disp is not None
    and (not config.register_to_reference or config.sample_from == SampleFrom.REFERENCE)
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
  session = None
  try:
    session = prepare_transect_session(
      data_dir,
      transect_id,
      config=config,
      collect_bbox_audit=collect_bbox_audit,
      cached_calib_path=cached_calib_path,
      calib_cache_path=calib_cache_path,
    )
    results = estimate_transect_traps(session)
    if collect_bbox_audit:
      return {'results': results, 'bbox_audit': session.get('bbox_audit')}
    return results
  finally:
    release_transect_session(session)


def prepare_transect_session(
  data_dir: str,
  transect_id: str,
  config: Optional[Config] = None,
  collect_bbox_audit: bool = False,
  cached_calib_path: Optional[str] = None,
  calib_cache_path: Optional[str] = None,
) -> dict:
  '''
  Initialize models and calibration once for a transect job.

  Returns a session dict that estimate_transect_traps() can reuse across
  trap download chunks (caller rewrites detection_frames/ + manifest.json
  between calls).
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
      and (not config.register_to_reference or config.sample_from == SampleFrom.REFERENCE)
      else None
    )
    bbox_audit['job_config']['resize_shape'] = resize_shape

  return {
    'data_dir': data_dir,
    'transect_id': transect_id,
    'transect_dir': transect_dir,
    'config': config,
    'depth_estimation_model': depth_estimation_model,
    'sam': sam,
    'calib': calib,
    'bbox_audit': bbox_audit,
    'collect_bbox_audit': collect_bbox_audit,
  }


def estimate_transect_traps(session: dict) -> dict:
  '''
  Estimate distances for trap detections currently listed in the transect
  manifest. Intended to be called repeatedly after swapping detection_frames/.

  Returns:
    {str(detection_id): {"distance": float|None, "error": str|None}}
  '''
  transect_dir = session['transect_dir']
  manifest = _load_manifest(transect_dir)
  return _estimate_trap_detections(
    transect_dir,
    manifest,
    session['config'],
    session['depth_estimation_model'],
    session['sam'],
    session['calib'],
    bbox_audit=session.get('bbox_audit'),
  )


def release_transect_session(session: Optional[dict]) -> None:
  '''
  Unload ONNX/torch models from a transect session and free GPU memory.

  Call from a finally block after each Celery depth job so VRAM does not
  accumulate across tasks on a long-lived worker process.
  '''
  if not session:
    return

  from utils import free_gpu_memory

  for key in ('depth_estimation_model', 'sam'):
    model = session.pop(key, None)
    if model is None:
      continue
    try:
      if hasattr(model, 'close'):
        model.close()
    except Exception as e:
      logging.warning('Failed to close depth model %s: %s', key, e)
    del model

  calib = session.pop('calib', None)
  if calib is not None and calib.registration is not None:
    calib.registration.close()
  session.clear()
  free_gpu_memory()
