from ..modules.annotation import AnnotationStatic
from ..source.config import GLCar50X
from ..modules.loader import LoaderStaticObstacle
from ..modules.image import ImageStatic

class_names = [
    "traffic cone",
    "crash_bar",
    "water barrier",
    "traffic drum",
    "construction sign",
    "triangular road sign",
]

_CAMERA_LIST = [
    "cam_front_120",   # 有地图时需要这里放【第1个】，用来和 rv 监督对齐
]

_SENSOR_NAMES = dict(camera_names=_CAMERA_LIST)

_CAR = GLCar50X

_PIPELINE = dict()

image_cfg = dict(
    type=ImageStatic,
    car=_CAR,
    camera_names=_CAMERA_LIST,
    target_resolution="200w",
)

annotation_cfg = dict(
    type=AnnotationStatic,
    class_names = class_names
)

_CAR = GLCar50X
static_dataset_cfg = dict(
    car=dict(type=_CAR),
    mode="train",
    sensor_names=_SENSOR_NAMES,
    loader=dict(
        type=LoaderStaticObstacle,
        car=_CAR,
        camera_names=_CAMERA_LIST,
        datasets_names=["Static"],
        # datasets_names=["debug"],
        only_key_frame=True,
        rebuild=False,
        use_redis=False,
    ),
    image=image_cfg,
    annotation=annotation_cfg,
    pipeline=_PIPELINE,
)

