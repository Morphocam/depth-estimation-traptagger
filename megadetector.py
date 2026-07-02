import sys
import os
import logging
import numpy as np
import cv2
import onnxruntime
from utils import get_onnxruntime_providers, DownloadableWeights, dirs


class MegaDetectorLabel:
    ANIMAL = 0
    PERSON = 1
    VEHICLE = 2


class MegaDetector(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        weights_url = "https://github.com/timmh/MegaDetectorLite/releases/download/v0.2/md_v5a.0.0.onnx"
        weights_md5 = "c2c93e4ed7e297eb650562df74341a25"
        weights_path = self.get_weights(weights_url, weights_md5)

        providers = get_onnxruntime_providers(enable_coreml=False)
        try:
            self.session = onnxruntime.InferenceSession(
                weights_path,
                providers=providers,
            )
        except Exception as e:
            providers_str = ",".join(providers)
            logging.warn(f"Failed to create onnxruntime inference session with providers '{providers_str}', trying 'CPUExecutionProvider'")
            self.session = onnxruntime.InferenceSession(
                weights_path,
                providers=["CPUExecutionProvider"],
            )

        self.common_size = None


    def __call__(self, imgs):
        # ensure model is loaded
        self._load_model()

        if not isinstance(imgs, list):
            imgs = [imgs]
            was_list = False
        else:
            was_list = True

        results = []
        for img in imgs:
            # BGR to RGB
            img_rgb = img[..., ::-1]

            # convert into 0..1 range
            img_rgb = img_rgb / 255.

            # resize
            if self.common_size is not None:
                img_input = cv2.resize(img_rgb, self.common_size, cv2.INTER_AREA)
            else:
                img_input = img_rgb

            # transpose from HWC to CHW
            img_input = img_input.transpose(2, 0, 1)

            # add batch dimension
            img_input = img_input[None, ...]

            # compute
            scores, labels, boxes = self.session.run(
                ["scores", "labels", "boxes"],
                {self.session.get_inputs()[0].name: img_input.astype(np.float32)}
            )

            if self.common_size is not None:
                for box in boxes:
                    box[0] = box[0] * img.shape[1] / self.common_size[0]
                    box[1] = box[1] * img.shape[0] / self.common_size[1]
                    box[2] = box[2] * img.shape[1] / self.common_size[0]
                    box[3] = box[3] * img.shape[0] / self.common_size[1]
            
            results.append((scores, labels, boxes))

        if not was_list:
            return results[0]
        else:
            return results


class MegaDetectorV2(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        # Try local weights folder first, then cache dir
        weights_path = os.path.join("weights", "md_v5b.0.0.pt")
        if not os.path.exists(weights_path):
            download_dir = os.path.join(dirs.user_cache_dir, "weights")
            weights_path = os.path.join(download_dir, "md_v5b.0.0.pt")

        if not os.path.exists(weights_path):
            raise RuntimeError(f"MegaDetector v5b weights not found at {weights_path}. Please ensure md_v5b.0.0.pt is in the weights folder.")

        try:
            import torch
            import yolov5
            import functools
            
            # Newer torch versions (2.6+) default weights_only=True which fails for YOLOv5 models
            # We temporarily monkeypatch torch.load to allow loading the model
            original_torch_load = torch.load
            torch.load = functools.partial(original_torch_load, weights_only=False)
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = yolov5.load(weights_path, device=device)
            finally:
                torch.load = original_torch_load
                
            self.model.eval()
        except Exception as e:
            logging.error(f"Failed to load MegaDetectorV2: {e}")
            raise RuntimeError(f"Failed to load MegaDetectorV2 using yolov5 package. Error: {e}")

    def __call__(self, imgs):
        # ensure model is loaded
        self._load_model()

        if not isinstance(imgs, list):
            imgs = [imgs]
            was_list = False
        else:
            was_list = True

        # YOLOv5 expects RGB images. Input imgs are BGR from cv2.
        imgs_rgb = [img[..., ::-1] for img in imgs]

        # Run inference
        results = []
        import torch
        with torch.no_grad():
            model_results = self.model(imgs_rgb)

        # Parse results
        # model_results.xyxy is a list of tensors [x1, y1, x2, y2, confidence, class]
        for det in model_results.xyxy:
            det = det.cpu().numpy()
            scores = det[:, 4]
            labels = det[:, 5].astype(int)
            boxes = det[:, :4]
            results.append((scores, labels, boxes))

        if not was_list:
            return results[0]
        else:
            return results


class MegaDetectorV6(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        try:
            from PytorchWildlife.models import detection as pw_detection
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # We use yolov9-c as default compact version as it's a good balance.
            # It will download weights to ~/.cache/torch/hub/checkpoints or similar.
            self.model = pw_detection.MegaDetectorV6(version='MDV6-yolov9-c', device=device)
        except Exception as e:
            logging.error(f"Failed to load MegaDetectorV6: {e}")
            raise RuntimeError(f"Failed to load MegaDetectorV6 using PytorchWildlife package. Error: {e}")

    def __call__(self, imgs):
        # ensure model is loaded
        self._load_model()

        if not isinstance(imgs, list):
            imgs = [imgs]
            was_list = False
        else:
            was_list = True

        results = []
        for img in imgs:
            # PytorchWildlife MegaDetectorV6 expects RGB image (numpy array)
            img_rgb = img[..., ::-1]

            # single_image_detection returns a dict with 'detections' which is a supervision.Detections object
            res = self.model.single_image_detection(img_rgb)
            detections = res['detections']

            scores = detections.confidence
            labels = detections.class_id.astype(int)
            boxes = detections.xyxy # [x1, y1, x2, y2]

            results.append((scores, labels, boxes))

        if not was_list:
            return results[0]
        else:
            return results
