import torch
import numpy as np
from transformers import DepthProImageProcessor, DepthProForDepthEstimation
from utils import DownloadableWeights, condition_disparity

class DepthPro(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # DepthPro requires transformers and accelerate
        self.processor = DepthProImageProcessor.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.device)
        self.model.eval()

    def __call__(self, imgs):
        self._load_model()
        
        if not isinstance(imgs, list):
            imgs = [imgs]
            was_list = False
        else:
            was_list = True

        # Convert BGR (cv2) to RGB and make contiguous to avoid negative strides issue
        imgs_rgb = [np.ascontiguousarray(img[..., ::-1]) for img in imgs]
        
        predictions = []
        with torch.no_grad():
            for img_rgb in imgs_rgb:
                inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                
                # Post-process to get depth and focal length
                post_processed = self.processor.post_process_depth_estimation(
                    outputs, target_sizes=[img_rgb.shape[:2]]
                )
                # DepthPro outputs metric depth
                predicted_depth = post_processed[0]["predicted_depth"]
                predictions.append(predicted_depth.cpu().numpy())

        if not was_list:
            return predictions[0]
        else:
            return predictions
