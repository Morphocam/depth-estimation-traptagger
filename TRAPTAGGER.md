## TrapTagger integration

TrapTagger calls `traptagger_api.run_transect_job(data_dir, transect_id)` after staging
images and writing `manifest.json` under each transect folder.

See `traptagger_api.py` and `traptagger_default_config()` for headless server defaults.

## Example manifest

{
  "cameragroup_id": 73,
  "cam_name": "XA20_images",
  "calibration": [
    {
      "known_distance": 20.0,
      "relative_path": "calibration_frames/20.jpg",
      "bbox_pixels": [120, 80, 400, 600]
    }
  ],
  "trap": [
    {
      "detection_id": 2409,
      "relative_path": "detection_frames/2409.jpg",
      "bbox_pixels": [100, 200, 300, 450],
      "download_ok": true
    }
  ]
}

## Manifest rules

* One trap row per detection; filename = `{detection_id}.jpg`
* `download_ok: false` → API returns `error: 's3_download_failed'`
* Trap row without `bbox_pixels` → API returns `error: 'missing_bbox'`
* Cal images: distance in filename (`20.jpg`) used by `get_calibration_frame_dist()`
* Cal and trap `bbox_pixels` are `[xmin, ymin, xmax, ymax]` in full-image pixel coordinates (from TrapTagger)
* Calibration masks are built with SAM from manifest cal bboxes (MegaDetector is not used)