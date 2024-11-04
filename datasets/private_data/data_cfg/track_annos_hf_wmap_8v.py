from ..modules.annotation import AnnotationDet, AnnotationADMap
from ..source.config import HFCar9
from ..modules.loader import LoaderBase
from ..modules.image import ImageBase
from ..modules.pipelines.transformation import MultiFrameImageAffineTransformation, ObjectRangeFilter

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "motorcycle",
    "bicycle",
    "tricycle",
    "cyclist",
    "pedestrian",
    # "masked_area",
]
point_cloud_range = [-40, -70.0, -5.0, 40, 70.0, 5.0]
category_map = {
    "小汽车": "car",
    "汽车": "car",
    "货车": "truck",
    "工程车": "construction_vehicle",
    "巴士": "bus",
    "摩托车": "motorcycle",
    "自行车": "bicycle",
    "三轮车": "tricycle",
    "骑车人": "cyclist",
    "骑行的人": "cyclist",
    "人": "pedestrian",
    "行人": "pedestrian",
    "其它": "other",
    "其他": "other",
    "残影": "ghost",
    "蒙版": "masked_area",
    "car": "car",
    "truck": "truck",
    "construction_vehicle": "construction_vehicle",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "tricycle": "tricycle",
    "cyclist": "cyclist",
    "pedestrian": "pedestrian",
    "other": "other",
    "ghost": "ghost",
    "masked_area": "masked_area",
}

_CAMERA_LIST = [
    "cam_front_left_120",
    "cam_front_120",   # 有地图时需要这里放【第1个】，用来和 rv 监督对齐
    "cam_front_right_120",
    "cam_back_right_120",
    "cam_back_120",
    "cam_back_left_120",
]

_SENSOR_NAMES = dict(camera_names=_CAMERA_LIST)

_CAR = HFCar9

_PIPELINE_MULTIFRAME = dict(
    object_range_filter=dict(
        type=ObjectRangeFilter,
        point_cloud_range=point_cloud_range,
    ),
    # bda_aug: need to be adapted
    ida_aug=dict(
        type=MultiFrameImageAffineTransformation,
        aug_conf=dict(
            final_dim=(224, 400),
            # resize_lim=((0.472, 0.5), (0.472, 0.5)),
            resize_lim=(0.2084, 0.2084),#((0.5, 0.5), (0.5, 0.5)), # zy check
            bot_pct_lim=(0.0, 0.0),
            H=1080,
            W=1920,
            rand_flip=False,
            rot_lim=(-0.0, 0.0),
        ),
        camera_names=_CAMERA_LIST,
        img_norm=False,
        img_conf={"img_mean": [123.675, 116.28, 103.53], "img_std": [58.395, 57.12, 57.375], "to_rgb": False},
    ),
)

image_cfg = dict(
    type=ImageBase,
    car=_CAR,
    camera_names=_CAMERA_LIST,
    target_resolution="200w",
)

annotation_cfg = dict(
    box=dict(
        type=AnnotationDet,
        label_key="pre_labels",
        category_map=category_map,
        class_names=class_names,
        occlusion_threshold=-1,
        filter_outlier_boxes=True,
        filter_outlier_frames=True,
        filter_empty_2d_bboxes=False,
        roi_range=point_cloud_range,
        HF=False,
    ),
    # admap=dict(
    #     type=AnnotationADMap,
    # ),
)

_CAR = HFCar9
base_dataset_cfg = dict(
    car=dict(type=_CAR),
    mode="train",
    sensor_names=_SENSOR_NAMES,
    num_frames_per_sample=24,
    loader=dict(
        type=LoaderBase,
        car=_CAR,
        camera_names=_CAMERA_LIST,
        datasets_names=["track_good_2738json"],
        only_key_frame=False,
        rebuild=False,
    ),
    image=image_cfg,
    annotation=annotation_cfg,
    pipeline=_PIPELINE_MULTIFRAME,
)

