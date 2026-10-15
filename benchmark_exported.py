'''
Benchmark the TrapTagger pipeline on the exported field validation set.

Stages exported_comp/ into the TrapTagger job layout (transects/<id>/manifest.json,
calibration_frames/, detection_frames/), runs traptagger_api.run_transect_job()
per station and compares the estimated distances against the ground truth CSVs.

Usage:
  python benchmark_exported.py                      # default TrapTagger configuration
  python benchmark_exported.py --model dpt --out results_dpt.csv
  python benchmark_exported.py --model unidepth_v2 --legacy-calibration
'''

import argparse
import glob
import json
import logging
import os
import re
import shutil

import numpy as np
import pandas as pd

from config import Config
from custom_types import (
  DepthEstimationModel,
  DetectionSamplingMethod,
  MetricCalibrationMethod,
  RegressionMethod,
  SampleFrom,
)
import traptagger_api

# RMSE reported in the paper for the stations it covers
PAPER_RMSE = {'XA30': 2.21, 'XA62': 2.04, 'XA71': 2.54, 'XA76': 2.17}


def _read_image_size(path):
  import cv2
  img = cv2.imread(path)
  return img.shape[0], img.shape[1]


def load_exported(export_dir):
  '''Return {station: {'cal': [...], 'trap': [...], 'truth': DataFrame}}.'''
  cal_df = pd.read_csv(os.path.join(export_dir, 'bounding_box_csvs', 'calibration_boxes.csv'))
  trap_df = pd.read_csv(os.path.join(export_dir, 'bounding_box_csvs', 'trap_images-boxes.csv'))
  trap_df['station'] = trap_df['File'].str.extract(r'/(XA\d+)/')[0]
  trap_df['filename'] = trap_df['File'].str.split('/').str[-1]

  stations = {}
  for truth_path in sorted(glob.glob(os.path.join(export_dir, 'truth_csvs', '*.csv'))):
    truth = pd.read_csv(truth_path)
    station = str(truth['station'].iloc[0])
    truth = truth[['station', 'filename', 'species', 'depth']].copy()
    truth['depth'] = pd.to_numeric(truth['depth'], errors='coerce')
    stations[station] = {'truth': truth, 'cal': [], 'trap': []}

    cal_dir = glob.glob(os.path.join(export_dir, 'survey', '_calibration_', station, '*'))[0]
    for _, row in cal_df[cal_df['cameragroup'] == station].iterrows():
      path = os.path.join(cal_dir, row['filename'].split('/')[-1])
      stations[station]['cal'].append({
        'path': path,
        'distance': float(row['distance_m']),
        'box_norm': [row['left'], row['top'], row['right'], row['bottom']],
      })

    trap_dir = glob.glob(os.path.join(export_dir, 'survey', 'trap_images_compressed', station, '*'))[0]
    for _, row in trap_df[trap_df['station'] == station].iterrows():
      stations[station]['trap'].append({
        'path': os.path.join(trap_dir, row['filename']),
        'filename': row['filename'],
        'score': float(row['Score']),
        'box_norm': [row['Left'], row['Top'], row['Right'], row['Bottom']],
      })
  return stations


def _box_pixels(box_norm, hw):
  h, w = hw
  left, top, right, bottom = box_norm
  return [float(left) * w, float(top) * h, float(right) * w, float(bottom) * h]


def stage_station(data_dir, station, items):
  '''Write the TrapTagger job layout for one station. Returns detection_id -> trap item.'''
  transect_dir = os.path.join(data_dir, 'transects', station)
  if os.path.isdir(transect_dir):
    shutil.rmtree(transect_dir)
  os.makedirs(os.path.join(transect_dir, 'calibration_frames'))
  os.makedirs(os.path.join(transect_dir, 'detection_frames'))

  manifest = {'cameragroup_id': station, 'cam_name': station, 'calibration': [], 'trap': []}
  for cal in items['cal']:
    # name by distance so get_calibration_frame_dist() also works
    ext = os.path.splitext(cal['path'])[1]
    name = '{:g}{}'.format(cal['distance'], ext)
    shutil.copy(cal['path'], os.path.join(transect_dir, 'calibration_frames', name))
    manifest['calibration'].append({
      'known_distance': cal['distance'],
      'relative_path': 'calibration_frames/' + name,
      'bbox_pixels': _box_pixels(cal['box_norm'], _read_image_size(cal['path'])),
    })

  by_id = {}
  for det_id, trap in enumerate(items['trap']):
    ext = os.path.splitext(trap['path'])[1]
    name = '{}{}'.format(det_id, ext)
    shutil.copy(trap['path'], os.path.join(transect_dir, 'detection_frames', name))
    manifest['trap'].append({
      'detection_id': det_id,
      'relative_path': 'detection_frames/' + name,
      'bbox_pixels': _box_pixels(trap['box_norm'], _read_image_size(trap['path'])),
      'download_ok': True,
    })
    by_id[str(det_id)] = trap

  with open(os.path.join(transect_dir, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
  return by_id


def match_truth(station, truth, by_id, results):
  '''One estimate per ground-truth image: the highest-scoring detection.'''
  rows = []
  for det_id, trap in by_id.items():
    res = results.get(det_id, {})
    rows.append({
      'filename': trap['filename'],
      'score': trap['score'],
      'pred': res.get('distance'),
      'error': res.get('error'),
    })
  pred = pd.DataFrame(rows)
  pred = pred.sort_values('score', ascending=False).drop_duplicates('filename')
  merged = truth.merge(pred, on='filename', how='left')
  merged['station'] = station
  return merged


def rmse(df):
  ok = df.dropna(subset=['depth', 'pred'])
  return float(np.sqrt(np.mean((ok['pred'] - ok['depth']) ** 2))) if len(ok) else float('nan')


def summarize(merged):
  lines = []
  for station, df in merged.groupby('station'):
    paper = PAPER_RMSE.get(station)
    lines.append({
      'station': station,
      'n': int(df.dropna(subset=['depth', 'pred']).shape[0]),
      'rmse': rmse(df),
      'paper_rmse': paper,
    })
  summary = pd.DataFrame(lines)
  summary.loc[len(summary)] = {
    'station': 'ALL', 'n': int(summary['n'].sum()), 'rmse': rmse(merged), 'paper_rmse': None,
  }
  paper_stations = merged[merged['station'].isin(PAPER_RMSE)]
  summary.loc[len(summary)] = {
    'station': 'PAPER_STATIONS', 'n': int(paper_stations.dropna(subset=['depth', 'pred']).shape[0]),
    'rmse': rmse(paper_stations), 'paper_rmse': float(np.sqrt(np.mean(np.square(list(PAPER_RMSE.values()))))),
  }
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--export-dir', default='exported_comp')
  parser.add_argument('--work-dir', default=os.path.join('exported_comp', '_benchmark_job'))
  parser.add_argument('--model', default='unidepth_v2_large', type=str.lower,
                      choices=[m.name.lower() for m in DepthEstimationModel])
  parser.add_argument('--metric-calibration', default='piecewise_linear', type=str.lower,
                      choices=[m.name.lower() for m in MetricCalibrationMethod])
  parser.add_argument('--regression', default='ransac', type=str.lower,
                      choices=[m.name.lower() for m in RegressionMethod])
  parser.add_argument('--sampling', default='sam', type=str.lower,
                      choices=[m.name.lower() for m in DetectionSamplingMethod])
  parser.add_argument('--max-depth', type=float, default=None)
  parser.add_argument('--legacy-calibration', action='store_true',
                      help='pixel-to-pixel reference alignment without registration, info-strip or animal masking')
  parser.add_argument('--stations', nargs='*', default=None)
  parser.add_argument('--out', default=None, help='per-image CSV output')
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  config = traptagger_api.traptagger_default_config()
  config.depth_estimation_model = DepthEstimationModel[args.model.upper()]
  config.metric_calibration_method = MetricCalibrationMethod[args.metric_calibration.upper()]
  config.calibration_regression_method = RegressionMethod[args.regression.upper()]
  config.detection_sampling_method = DetectionSamplingMethod[args.sampling.upper()]
  config.sample_from = SampleFrom.DETECTION
  if args.legacy_calibration:
    config.register_to_reference = False
    config.mask_info_bar = False
    config.calibration_mask_animals = False
  if args.max_depth is not None:
    config.max_depth = args.max_depth

  stations = load_exported(args.export_dir)
  if args.stations:
    stations = {k: v for k, v in stations.items() if k in args.stations}

  all_merged = []
  for station, items in stations.items():
    by_id = stage_station(args.work_dir, station, items)
    results = traptagger_api.run_transect_job(args.work_dir, station, config=config)
    merged = match_truth(station, items['truth'], by_id, results)
    all_merged.append(merged)
    print('{}: RMSE {:.3f} (n={})'.format(station, rmse(merged), merged['pred'].notna().sum()), flush=True)

  merged = pd.concat(all_merged, ignore_index=True)
  if args.out:
    merged.to_csv(args.out, index=False)
  print(summarize(merged).to_string(index=False, float_format='%.3f'))


if __name__ == '__main__':
  main()
