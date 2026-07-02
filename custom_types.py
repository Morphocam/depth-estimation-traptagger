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


class DepthEstimationModel(Enum):
    DPT = auto()
    DEPTH_AHYTHING_METRIC = auto()
    METRIC_3D_V2_VIT_S = auto()
    DPT_PYTORCH = auto()
    MONODEPTH2 = auto()
    DEPTH_PRO = auto()


class DetectionModel(Enum):
    MEGADETECTOR_V5A = auto()
    MEGADETECTOR_V5B = auto()
    MEGADETECTOR_V6 = auto()