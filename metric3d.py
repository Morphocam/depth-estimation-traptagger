
import sys
import os
import json
import logging
import numpy as np
import cv2
import onnxruntime
from utils import get_onnxruntime_providers, DownloadableWeights


class Metric3D(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        weights_url = "https://github.com/timmh/Metric3D/releases/download/v0.1/metric3d_vit_small.onnx"
        weights_md5 = "f620d1b8d70dd3cd8652b82cfe9f9a77"
        weights_path = self.get_weights(weights_url, weights_md5)

        providers = get_onnxruntime_providers()
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

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.net_w, self.net_h = json.loads(metadata["ImageSize"])
        normalization = json.loads(metadata["Normalization"])
        self.prediction_factor = float(metadata["PredictionFactor"])
        self.mean = np.array(normalization["mean"])
        self.std = np.array(normalization["std"])
    
    def __call__(self, imgs):
        # ensure model is loaded
        self._load_model()

        if not isinstance(imgs, list):
            imgs = [imgs]
            was_list = False
        else:
            was_list = True

        predictions = []
        for img in imgs:
            original_shape = img.shape
            preprocessed_img = self.preprocess(img)

            # add batch dimension
            img_input = preprocessed_img[None, ...]

            # compute
            prediction = self.session.run(["pred_depth"], {"image": img_input.astype(np.float32)})[0][0][0]

            # post-process
            resized_prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]), cv2.INTER_CUBIC)
            resized_prediction *= self.prediction_factor

            # into disparity
            resized_prediction = np.clip(resized_prediction, 1e-6, np.inf) ** -1
            predictions.append(resized_prediction)

        if not was_list:
            return predictions[0]
        else:
            return predictions

    def preprocess(self, img):
        # BGR to RGB
        img = img[..., ::-1]

        # resize
        img_input = cv2.resize(img, (self.net_w, self.net_h), cv2.INTER_LINEAR)

        # normalize
        img_input = (img_input - self.mean) / self.std

        # transpose from HWC to CHW
        img_input = img_input.transpose(2, 0, 1)

        return img_input