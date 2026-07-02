import os
import logging
import numpy as np
import cv2
import torch
import zipfile
import io
import urllib.request
from utils import DownloadableWeights, condition_disparity
from monodepth2_lib.networks import ResnetEncoder, DepthDecoder

class MonoDepth2(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Official weights URL (mono+stereo 640x192)
        weights_url = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip"
        
        # Download and extract weights if not already present
        download_dir = os.path.join(os.path.expanduser("~"), ".cache", "monodepth2")
        
        if not os.path.exists(os.path.join(download_dir, "encoder.pth")):
            os.makedirs(download_dir, exist_ok=True)
            logging.info(f"Downloading monodepth2 weights from {weights_url}...")
            try:
                response = urllib.request.urlopen(weights_url)
                with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                    z.extractall(download_dir)
            except Exception as e:
                logging.error(f"Failed to download/extract monodepth2 weights: {e}")
                raise RuntimeError(f"Failed to load MonoDepth2 weights: {e}")

        # Loading models
        encoder_path = os.path.join(download_dir, "encoder.pth")
        decoder_path = os.path.join(download_dir, "depth.pth")

        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # Extract training resolution
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dict_dec = torch.load(decoder_path, map_location=self.device)
        self.decoder.load_state_dict(loaded_dict_dec)
        self.decoder.to(self.device)
        self.decoder.eval()

    def __call__(self, imgs):
        self._load_model()
        
        if not isinstance(imgs, list):
            imgs = [imgs]
            was_list = False
        else:
            was_list = True

        predictions = []
        with torch.no_grad():
            for img in imgs:
                original_height, original_width = img.shape[:2]
                
                # Preprocess: BGR to RGB, resize, normalize, to tensor
                img_rgb = img[..., ::-1]
                img_resized = cv2.resize(img_rgb, (self.feed_width, self.feed_height), cv2.INTER_LANCZOS4)
                img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().to(self.device) / 255.0
                img_tensor = img_tensor.unsqueeze(0)
                
                # Normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                img_tensor = (img_tensor - mean) / std
                
                # Inference
                features = self.encoder(img_tensor)
                outputs = self.decoder(features)
                
                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False
                )
                
                # Condition disparity
                disp_np = disp_resized.squeeze().cpu().numpy()
                disp_np = condition_disparity(disp_np)
                
                predictions.append(disp_np)

        if not was_list:
            return predictions[0]
        else:
            return predictions
