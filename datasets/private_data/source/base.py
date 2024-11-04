from abc import ABCMeta
from collections import namedtuple

SENSORS = namedtuple(
    "sensors",
    [
        # 环视
        "cam_front_190",
        "cam_back_190",
        "cam_left_190",
        "cam_right_190",
        # 前视
        "cam_front_120",
        "cam_front_60",
        "cam_front_30",
        "cam_front_70_left",
        "cam_front_70_right",
        "cam_front_120_backup",
        "cam_front_left_120",
        "cam_front_right_120",
        # 后视
        "cam_back_120",
        "cam_back_left_120",
        "cam_back_right_120",
        # 主雷达
        "main_lidar",
    ],
)

IMAGE_RESOLUTION = {"200w": (1920, 1080), "800w": (3840, 2160), "800w_org": (3840, 2165)}


class Sensor:
    """base class of sensor.

    Args:
        name (str): sensor name
        resolution (tuple[int]): camera original resolution

    """

    def __init__(self, name="", resolution=str):
        self.name = name
        self.resolution = resolution


class BaseCar(metaclass=ABCMeta):
    name = None
    trainset, trainset_partial = None, None
    benchmark, benchmark_partial = None, None

    sensors = SENSORS(
        cam_front_190=Sensor("cam_front_190", "200w"),
        cam_back_190=Sensor("cam_back_190", "200w"),
        cam_left_190=Sensor("cam_left_190", "200w"),
        cam_right_190=Sensor("cam_right_190", "200w"),
        cam_front_120=Sensor("cam_front_120", "800w"),
        cam_front_60=Sensor("cam_front_60", "800w"),
        cam_front_30=Sensor("cam_front_30", "800w"),
        cam_front_120_backup=None,
        cam_front_70_left=None,
        cam_front_70_right=None,
        cam_front_left_120=Sensor("cam_front_left_120", "200w"),
        cam_front_right_120=Sensor("cam_front_right_120", "200w"),
        cam_back_120=Sensor("cam_back_120", "800w"),
        cam_back_left_120=Sensor("cam_back_left_120", "200w"),
        cam_back_right_120=Sensor("cam_back_right_120", "200w"),
        main_lidar=Sensor("main_lidar", ()),
    )
