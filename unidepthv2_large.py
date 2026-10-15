import logging
import numpy as np
from utils import DownloadableWeights


class UniDepthv2Large(DownloadableWeights):
    """
    UniDepth V2 ViT-L (PyTorch). Predicts metric depth in metres.

    Requires the `unidepth` package (pip install --no-deps
    git+https://github.com/lpiccinelli-eth/UniDepth.git). Weights are fetched
    from the Hugging Face hub on first use and cached there.
    """

    hf_repo = "lpiccinelli/unidepth-v2-vitl14"

    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return

        import torch
        from unidepth.models import UniDepthV2

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniDepthV2.from_pretrained(self.hf_repo).to(self.device).eval()
        self._model_loaded = True

    def __call__(self, imgs):
        self._load_model()
        torch = self._torch

        if not isinstance(imgs, list):
            imgs = [imgs]
            was_list = False
        else:
            was_list = True

        predictions = []
        with torch.no_grad():
            for img in imgs:
                # BGR uint8 HWC -> RGB CHW
                rgb = torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).permute(2, 0, 1).to(self.device)
                depth = self.model.infer(rgb)["depth"][0, 0]
                predictions.append(depth.float().cpu().numpy())

        if not was_list:
            return predictions[0]
        else:
            return predictions
