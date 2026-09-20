from enum import Enum, auto


class DetectionSamplingMethod(Enum):
    BBOX_BOTTOM = auto()
    BBOX_PERCENTILE = auto()
    SAM = auto()


class SampleFrom(Enum):
    REFERENCE = auto()
    DETECTION = auto()


class MultipleAnimalReduction(Enum):
    NONE = auto()
    MEDIAN = auto()
    ONLY_CENTERMOST = auto()


class RegressionMethod(Enum):
    RANSAC = auto()
    LEASTSQUARES = auto()
    POLY = auto()
    RANSAC_POLY = auto()
    PIECEWISE_LINEAR = auto()


class MetricCalibrationMethod(Enum):
    """Correction applied on top of models which already predict metric depth"""
    NONE = auto()  # trust the raw metric prediction
    SCALE = auto()  # single multiplicative factor
    AFFINE = auto()  # scale and shift
    PIECEWISE_LINEAR = auto()  # monotonic piecewise-linear map through all calibration points


class DepthEstimationModel(Enum):
    DPT = auto()
    DEPTH_AHYTHING_METRIC = auto()
    METRIC_3D_V2_VIT_S = auto()
    DPT_PYTORCH = auto()
    MONODEPTH2 = auto()
    DEPTH_PRO = auto()
    UNIDEPTH_V2 = auto()


# Models whose raw output is already in metric units and which therefore do not
# require the per-image scale/shift alignment used for relative depth models.
# METRIC_3D_V2_VIT_S is intentionally absent: its wrapper converts the metric
# prediction back into disparity, so it is consumed by the relative pipeline.
METRIC_DEPTH_ESTIMATION_MODELS = frozenset({
    DepthEstimationModel.DEPTH_AHYTHING_METRIC,
    DepthEstimationModel.DEPTH_PRO,
    DepthEstimationModel.UNIDEPTH_V2,
})


class DetectionModel(Enum):
    MEGADETECTOR_V5A = auto()
    MEGADETECTOR_V5B = auto()
    MEGADETECTOR_V6 = auto()