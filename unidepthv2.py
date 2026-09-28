import logging
import numpy as np
import cv2
import onnxruntime
from utils import get_onnxruntime_providers, DownloadableWeights


# ibaiGorordo's unidepthv2_vits14_simp.onnx has no custom metadata.
# Input is fixed at 364x644; these are the fallback if the graph shape is dynamic.
_DEFAULT_NET_H = 364
_DEFAULT_NET_W = 644
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _static_dim(value, fallback):
    try:
        dim = int(value)
    except (TypeError, ValueError):
        return fallback
    if dim <= 0:
        return fallback
    return dim


class UniDepthv2(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return

        # Filename must stay unidepthv2_vits14_simp.onnx: that is the cache key.
        # Bake that file into ~/.cache/depth-estimation-traptagger/weights/ on the
        # depth AMI. md5 is omitted because the published checksum is SHA256 and
        # get_weights compares MD5, which would reject a correctly placed file.
        weights_url = "https://huggingface.co/ibaiGorordo/unidepth-v2-vits14-onnx/resolve/main/unidepthv2_vits14_simp.onnx"
        weights_path = self.get_weights(weights_url, None)

        providers = get_onnxruntime_providers()
        try:
            self.session = onnxruntime.InferenceSession(
                weights_path,
                providers=providers,
            )
        except Exception:
            providers_str = ",".join(providers)
            logging.warn(
                f"Failed to create onnxruntime inference session with providers '{providers_str}', trying "
                f"'CPUExecutionProvider'")
            self.session = onnxruntime.InferenceSession(
                weights_path,
                providers=["CPUExecutionProvider"],
            )

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        # NCHW. H and W are static on this export (364, 644); batch is dynamic.
        shape = model_input.shape
        self.net_h = _static_dim(shape[2] if len(shape) > 2 else None, _DEFAULT_NET_H)
        self.net_w = _static_dim(shape[3] if len(shape) > 3 else None, _DEFAULT_NET_W)
        self.mean = _IMAGENET_MEAN
        self.std = _IMAGENET_STD
        output_names = [o.name for o in self.session.get_outputs()]
        if "depth" not in output_names:
            raise RuntimeError(
                "UniDepth ONNX is missing a 'depth' output (found: {})".format(output_names)
            )
        self.depth_output_name = "depth"
        self._model_loaded = True

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

            # Depth is already metres. out_K and confidence are unused.
            prediction = self.session.run(
                [self.depth_output_name],
                {self.input_name: img_input.astype(np.float32)},
            )[0]
            prediction = np.squeeze(prediction)
            if prediction.ndim != 2:
                raise RuntimeError(
                    "Unexpected UniDepth depth shape {}".format(prediction.shape)
                )

            resized_prediction = cv2.resize(
                prediction, (original_shape[1], original_shape[0]), cv2.INTER_CUBIC
            )
            predictions.append(resized_prediction)

        if not was_list:
            return predictions[0]
        else:
            return predictions

    def preprocess(self, img):
        # BGR to RGB
        img = img[..., ::-1]

        # convert into 0..1 range
        img = img / 255.

        # resize
        img_input = cv2.resize(img, (self.net_w, self.net_h), cv2.INTER_AREA)

        # ImageNet normalize
        img_input = (img_input - self.mean) / self.std

        # transpose from HWC to CHW
        img_input = img_input.transpose(2, 0, 1)

        return img_input
