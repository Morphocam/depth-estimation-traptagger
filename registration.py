'''
Geometric registration of trap images to the reference calibration frame.

Calibration aligns every trap image's depth map to the depth map of the farthest
calibration frame pixel by pixel. That only holds if the camera did not move.
In the field cameras are regularly re-aimed between the calibration session and
trapping, and some cameras switch between 4:3 and 16:9 capture modes. Each trap
image is therefore registered to the reference frame with a homography (exact
for a camera rotated about its centre) before the depth maps are compared.

Matching uses DISK + LightGlue (kornia) when available, which is robust to the
lighting and vegetation changes between calibration and trap images, and falls
back to SIFT. The camera's info strip and logo badge are excluded from matching
because they are identical in every frame and would pin the result to identity.
'''

import logging
from typing import Optional

import cv2
import numpy as np


INFO_BAR_DARK_LEVEL = 40  # grey level below which a pixel counts as part of the info strip
INFO_BAR_MAX_FRACTION = 0.08  # never treat more than this fraction of the height as the strip
OVERLAY_MARGIN_FRACTION = 0.06  # extra band above the strip holding logo badges and icons
MIN_INLIERS = 15
MAX_SCALE_CHANGE = 0.25  # reject registrations zooming more than this
MAX_PERSPECTIVE = 5e-4  # reject strongly degenerate homographies


def info_bar_rows(img: np.ndarray) -> int:
  '''Height in pixels of the dark camera info strip at the bottom of the frame.'''
  gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  height = gray.shape[0]
  dark = (gray < INFO_BAR_DARK_LEVEL).mean(axis=1)
  rows = 0
  for r in range(height - 1, int(height * (1 - INFO_BAR_MAX_FRACTION)), -1):
    if dark[r] > 0.6:
      rows = height - r
  return rows


def info_bar_mask(img: np.ndarray) -> np.ndarray:
  '''Boolean mask of the info strip, to exclude it from depth alignment.'''
  mask = np.zeros(img.shape[0:2], dtype=bool)
  rows = info_bar_rows(img)
  if rows:
    mask[-rows:] = True
  return mask


def _feature_rows(img: np.ndarray) -> int:
  '''Number of rows from the top that may be used for feature matching.'''
  return img.shape[0] - info_bar_rows(img) - int(round(img.shape[0] * OVERLAY_MARGIN_FRACTION))


def _validate(H, inliers) -> Optional[np.ndarray]:
  if H is None or inliers is None or int(np.sum(inliers)) < MIN_INLIERS:
    return None
  det = float(np.linalg.det(H[:2, :2]))
  if not ((1 - MAX_SCALE_CHANGE) ** 2 < det < (1 + MAX_SCALE_CHANGE) ** 2):
    return None
  if abs(H[2, 0]) > MAX_PERSPECTIVE or abs(H[2, 1]) > MAX_PERSPECTIVE:
    return None
  return H


class ImageRegistration:
  '''Estimate homographies mapping images onto one reference image.'''

  def __init__(self, reference_img: np.ndarray, use_learned: bool = True, max_keypoints: int = 2048):
    self.reference_img = reference_img
    self.max_keypoints = max_keypoints
    self._learned = None
    self._ref_learned = None
    self._ref_sift = None
    if use_learned:
      try:
        self._init_learned()
      except Exception as e:
        logging.warning('DISK/LightGlue unavailable (%s); registering with SIFT', e)
        self._learned = None

  # learned features --------------------------------------------------------
  def _init_learned(self):
    import torch
    import kornia.feature as KF
    self._torch = torch
    self._KF = KF
    self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
    disk = KF.DISK.from_pretrained('depth').to(self._device).eval()
    matcher = KF.LightGlueMatcher('disk').to(self._device).eval()
    self._learned = (disk, matcher)

  def _learned_features(self, img):
    torch = self._torch
    rows = _feature_rows(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:rows]
    tensor = torch.from_numpy(gray).float().div(255)[None, None].repeat(1, 3, 1, 1).to(self._device)
    disk, _ = self._learned
    with torch.no_grad():
      feats = disk(tensor, self.max_keypoints, pad_if_not_divisible=True)[0]
    lafs = self._KF.laf_from_center_scale_ori(
      feats.keypoints[None],
      torch.ones(1, len(feats.keypoints), 1, 1, device=self._device),
    )
    return feats, lafs, torch.tensor(gray.shape, device=self._device)

  def _learned_matches(self, img):
    if self._ref_learned is None:
      self._ref_learned = self._learned_features(self.reference_img)
    f1, lafs1, hw1 = self._learned_features(img)
    f2, lafs2, hw2 = self._ref_learned
    _, matcher = self._learned
    with self._torch.no_grad():
      _, idx = matcher(f1.descriptors, f2.descriptors, lafs1, lafs2, hw1=hw1, hw2=hw2)
    idx = idx.cpu().numpy()
    p1 = f1.keypoints.cpu().numpy()[idx[:, 0]]
    p2 = f2.keypoints.cpu().numpy()[idx[:, 1]]
    return p1.astype(np.float32), p2.astype(np.float32)

  # SIFT fallback -----------------------------------------------------------
  def _sift_features(self, img):
    if not hasattr(self, '_sift'):
      self._sift = cv2.SIFT_create(nfeatures=5000)
      self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = self._clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[:_feature_rows(img)] = 255
    return self._sift.detectAndCompute(gray, mask)

  def _sift_matches(self, img):
    if self._ref_sift is None:
      self._ref_sift = self._sift_features(self.reference_img)
    k1, d1 = self._sift_features(img)
    k2, d2 = self._ref_sift
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
      return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    p2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return p1, p2

  # public ------------------------------------------------------------------
  def __call__(self, img: np.ndarray) -> Optional[np.ndarray]:
    '''Homography mapping pixel coordinates of img onto the reference, or None.'''
    matchers = ([self._learned_matches] if self._learned is not None else []) + [self._sift_matches]
    for match in matchers:
      try:
        p1, p2 = match(img)
      except Exception as e:
        logging.warning('Image registration matcher failed: %s', e)
        continue
      if len(p1) < MIN_INLIERS:
        continue
      H, inliers = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 4.0)
      H = _validate(H, inliers)
      if H is not None:
        return H
    return None

  def close(self):
    self._learned = None
    self._ref_learned = None


def warp_to_reference(data: np.ndarray, H: np.ndarray, reference_shape, nearest=False, border_value=0):
  '''Warp a per-pixel map of the registered image into reference coordinates.'''
  return cv2.warpPerspective(
    data,
    H,
    (int(reference_shape[1]), int(reference_shape[0])),
    flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=border_value,
  )
