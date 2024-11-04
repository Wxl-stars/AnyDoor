from . import geely_car_1, geely_car_2, geely_car_50X, wm_car_1, wm_car_2, wm_car_3, wm_car_4, wm_car_5, hf_car_9, hf_car_9_tos, hf_car_9_bos
from .base import SENSORS, BaseCar, Sensor
from datasets.private_data.utils.functional import is_volcano_platform


class WMCar1(BaseCar):
    name = "wm_car_1"
    trainset, trainset_partial = wm_car_1.TRAINSET, wm_car_1.TRAINSET_PARTIAL
    benchmark, benchmark_partial = wm_car_1.BENCHMARK, wm_car_1.BENCHMARK_PARTIAL

    sensors = SENSORS(
        cam_front_190=Sensor("camera_4", "200w"),
        cam_back_190=Sensor("camera_6", "200w"),
        cam_left_190=Sensor("camera_7", "200w"),
        cam_right_190=Sensor("camera_1", "200w"),
        cam_front_120=Sensor("camera_13", "200w"),
        cam_front_60=Sensor("camera_15", "200w"),
        cam_front_30=Sensor("camera_12", "200w"),
        cam_front_120_backup=Sensor("camera_14", "200w"),
        cam_front_70_left=None,
        cam_front_70_right=None,
        cam_front_left_120=Sensor("camera_10", "200w"),
        cam_front_right_120=Sensor("camera_11", "200w"),
        cam_back_120=Sensor("camera_3", "200w"),
        cam_back_left_120=Sensor("camera_5", "200w"),
        cam_back_right_120=Sensor("camera_2", "200w"),
        main_lidar=Sensor("middle_lidar", "200w"),
    )


class WMCar2(BaseCar):
    name = "wm_car_2"
    trainset, trainset_partial = wm_car_2.TRAINSET, wm_car_2.TRAINSET_PARTIAL
    benchmark, benchmark_partial = wm_car_2.BENCHMARK, wm_car_2.BENCHMARK_PARTIAL

    sensors = SENSORS(
        cam_front_190=Sensor("camera_0_0", None),
        cam_back_190=Sensor("camera_0_2", None),
        cam_left_190=Sensor("camera_1_2", None),
        cam_right_190=Sensor("camera_1_0", None),
        cam_front_120=Sensor("camera_0_6", "800w"),
        cam_front_60=Sensor("camera_1_6", None),
        cam_front_30=Sensor("camera_0_7", None),
        cam_front_120_backup=None,
        cam_front_70_left=None,
        cam_front_70_right=None,
        cam_front_left_120=Sensor("camera_1_7", "200w"),
        cam_front_right_120=Sensor("camera_0_5", "200w"),
        cam_back_120=Sensor("camera_1_4", "200w"),
        cam_back_left_120=Sensor("camera_0_4", "200w"),
        cam_back_right_120=Sensor("camera_1_5", "200w"),
        main_lidar=Sensor("middle_lidar", None),
    )


class WMCar3(BaseCar):
    name = "wm_car_3"
    trainset, trainset_partial = wm_car_3.TRAINSET, wm_car_3.TRAINSET_PARTIAL
    benchmark, benchmark_partial = wm_car_3.BENCHMARK, wm_car_3.BENCHMARK_PARTIAL


class WMCar4(BaseCar):
    name = "wm_car_4"
    trainset, trainset_partial = wm_car_4.TRAINSET, wm_car_4.TRAINSET_PARTIAL
    benchmark, benchmark_partial = wm_car_4.BENCHMARK, wm_car_4.BENCHMARK_PARTIAL


class WMCar5(BaseCar):
    name = "wm_car_5"
    trainset, trainset_partial = wm_car_5.TRAINSET, wm_car_5.TRAINSET_PARTIAL
    benchmark, benchmark_partial = wm_car_5.BENCHMARK, wm_car_5.BENCHMARK_PARTIAL


class GeelyCar1(BaseCar):
    name = "geely_car_1"
    trainset, trainset_partial = geely_car_1.TRAINSET, geely_car_1.TRAINSET_PARTIAL
    benchmark, benchmark_partial = geely_car_1.BENCHMARK, geely_car_1.BENCHMARK_PARTIAL

    sensors = SENSORS(
        cam_front_190=None,
        cam_back_190=None,
        cam_left_190=None,
        cam_right_190=None,
        cam_front_120=None,
        cam_front_60=None,
        cam_front_30=Sensor("cam_front_30", "800w_org"),
        cam_front_120_backup=None,
        cam_front_70_left=Sensor("cam_front_70_left", "800w_org"),
        cam_front_70_right=Sensor("cam_front_70_right", "800w_org"),
        cam_front_left_120=Sensor("cam_front_left_120", "800w_org"),
        cam_front_right_120=Sensor("cam_front_right_120", "800w_org"),
        cam_back_120=Sensor("cam_back_120", "800w_org"),
        cam_back_left_120=Sensor("cam_back_left_120", "800w_org"),
        cam_back_right_120=Sensor("cam_back_right_120", "800w_org"),
        main_lidar=Sensor("front_lidar", None),
    )


class GeelyCar2(BaseCar):
    name = "car_102"
    trainset, trainset_partial = geely_car_2.TRAINSET, geely_car_2.TRAINSET_PARTIAL
    benchmark, benchmark_partial = geely_car_2.BENCHMARK, geely_car_2.BENCHMARK_PARTIAL

    # TODO: 补全分辨率
    sensors = SENSORS(
        cam_front_190=None,
        cam_back_190=None,
        cam_left_190=None,
        cam_right_190=None,
        cam_front_120=Sensor("cam_front_120_left", "800w_org"),
        cam_front_60=None,
        cam_front_30=Sensor("cam_front_30", "800w_org"),
        cam_front_120_backup=Sensor("cam_front_120_right", "800w_org"),
        cam_front_70_left=Sensor("cam_front_70", "800w_org"),
        cam_front_70_right=None,
        cam_front_left_120=Sensor("cam_front_left_120", "800w_org"),
        cam_front_right_120=Sensor("cam_front_right_120", "800w_org"),
        cam_back_120=Sensor("cam_back_120", "800w_org"),
        cam_back_left_120=Sensor("cam_back_left_120", "800w_org"),
        cam_back_right_120=Sensor("cam_back_right_120", "800w_org"),
        main_lidar=Sensor("fuser_lidar", None),
    )

class GLCar50X(BaseCar):
    name = "gl_car_50X"
    trainset, trainset_partial = geely_car_50X.TRAINSET, geely_car_50X.TRAINSET_PARTIAL
    benchmark, benchmark_partial = geely_car_50X.BENCHMARK, geely_car_50X.BENCHMARK_PARTIAL
    sensors = SENSORS(
        cam_front_190=None,
        cam_back_190=None,
        cam_left_190=None,
        cam_right_190=None,
        cam_front_120=Sensor("cam_front_120", "200w"),
        cam_front_60=None,
        cam_front_30=None,
        cam_front_120_backup=None,
        cam_front_70_left=None,
        cam_front_70_right=None,
        cam_front_left_120=None,
        cam_front_right_120=None,
        cam_back_120=None,
        cam_back_left_120=None,
        cam_back_right_120=None,
        main_lidar=Sensor("front_lidar", None),
    )

class HFCar9(BaseCar):
    name = "car_9"
    volcano = is_volcano_platform()
    if volcano:
        trainset, trainset_partial = hf_car_9_tos.TRAINSET, hf_car_9_tos.TRAINSET_PARTIAL
        benchmark, benchmark_partial = hf_car_9_tos.BENCHMARK, hf_car_9_tos.BENCHMARK_PARTIAL
        sensors = SENSORS(
            cam_front_190=None,
            cam_back_190=Sensor("cam_back_190", "200w"),
            cam_left_190=Sensor("cam_left_190", "200w"),
            cam_right_190=Sensor("cam_right_190", "200w"),
            cam_front_120=Sensor("cam_front_120", "200w"),
            cam_front_60=None,
            cam_front_30=Sensor("cam_front_30", "200w"),
            cam_front_120_backup=None,
            cam_front_70_left=None,
            cam_front_70_right=None,
            cam_front_left_120=Sensor("cam_front_left_120", "200w"),
            cam_front_right_120=Sensor("cam_front_right_120", "200w"),
            cam_back_120=Sensor("cam_back_120", "200w"),
            cam_back_left_120=Sensor("cam_back_left_120", "200w"),
            cam_back_right_120=Sensor("cam_back_right_120", "200w"),
            main_lidar=Sensor("front_lidar", None),
        )
    else:
        trainset, trainset_partial = hf_car_9_bos.TRAINSET, hf_car_9_bos.TRAINSET_PARTIAL
        benchmark, benchmark_partial = hf_car_9_bos.BENCHMARK, hf_car_9_bos.BENCHMARK_PARTIAL

        sensors = SENSORS(
            cam_front_190=None,
            cam_back_190=Sensor("cam_back_190", "200w"),
            cam_left_190=Sensor("cam_left_190", "200w"),
            cam_right_190=Sensor("cam_right_190", "200w"),
            cam_front_120=Sensor("cam_front_120", "800w"),
            cam_front_60=None,
            cam_front_30=Sensor("cam_front_30", "800w"),
            cam_front_120_backup=None,
            cam_front_70_left=None,
            cam_front_70_right=None,
            cam_front_left_120=Sensor("cam_front_left_120", "200w"),
            cam_front_right_120=Sensor("cam_front_right_120", "200w"),
            cam_back_120=Sensor("cam_back_120", "800w"),
            cam_back_left_120=Sensor("cam_back_left_120", "200w"),
            cam_back_right_120=Sensor("cam_back_right_120", "200w"),
            main_lidar=Sensor("front_lidar", None),
        )
