from collections import defaultdict
from megfile.errors import S3Exception
import time

import cv2
import numpy as np
import refile
from loguru import logger
from tqdm import tqdm

from datasets.private_data.source.base import IMAGE_RESOLUTION
from datasets.private_data.utils.functional import SensorCalibrationInterface, camera_filter, get_lidar_to_pixel, is_master

def s3_retry(func, args=[], kwargs={}, retries=3, delay=1):
    """  
    上下文管理器，提供重试机制。
    
    :param retries: 最大重试次数  
    :param delay: 每次重试之间的延迟（秒）  
    """  
    attempt = 0
    while attempt < retries:
        try:
            return func(*args, **kwargs)  # 如果没有抛出异常，退出重试循环  
        except S3Exception as e:
            attempt += 1
            if attempt == retries:
                print(f"操作失败，已达到最大重试次数 {retries}.")  
                raise  # 抛出最后一次捕获的异常  
            print(f"第 {attempt} 次重试失败: {e}")  
            time.sleep(delay)  # 等待指定时间后重试

class ImageBase:
    """
    The transformation for image undistort is generate by calibration parameters in initialisation, and decode the original image from the nori address, which is used as the input for the subsequent pipeline.
    The image output in this stage is a list sorted by camera_names, regardless of whether the image sizes of different cameras are the same or not.

    Options:
        undistort: Switch of all undistortion related functions

        target_resolution: Determine the size of the undistorted image

        undistort_func: Calculate the transformation relations from the calibration parameters. SimFov is used to project the physical camera to a hypothetical standard camera with the same intrinsics

        postpone_undistort: Option for sending original image and undistortion mapping to pipeline, such as gpu undistort
    """

    def __init__(
        self,
        car,
        camera_names,
        loader_output,
        mode,
        target_resolution="200w",
    ) -> None:
        self.car = car
        self.camera_names = camera_names
        self.loader_output = loader_output
        self.mode = mode
        self.nori_fetcher = None
        assert target_resolution in list(IMAGE_RESOLUTION.keys()) + ["original"]
        if target_resolution in IMAGE_RESOLUTION:
            self.target_resolution = IMAGE_RESOLUTION[target_resolution]
        else:
            self.target_resolution = (None, None)

        self.calibrated_sensors = self.loader_output["calibrated_sensors"]
        self.camera_name_mapping_s2h = self.loader_output["camera_name_mapping"]["standard"]
        self.raw_names = [self.camera_name_mapping_s2h[k]["hidden_name"] for k in self.camera_names]
        self._init_camera_translation_matrix()

    def _init_sensors_info(self):
        pass

    def _init_camera_translation_matrix(self):
        """用于预处理了去畸变的图像"""
        camera_info_cache = {
            "calib_param_to_index": defaultdict(int),
            "new_k": [],
            "lidar_to_pix": [],
        }
        self._init_sensors_info()
        for json_idx, sensors_info in tqdm(
            self.calibrated_sensors.items(),
            disable=(not is_master()),
            desc="[Init lidar to pixel translation]",
        ):
            for camera_name in self.camera_names:
                hidden_name = self._get_hidden_name(camera_name)
                sensor_info = sensors_info[hidden_name]

                if str(sensor_info) in camera_info_cache["calib_param_to_index"]:
                    map_id = camera_info_cache["calib_param_to_index"][str(sensor_info)]
                    cur_cam_info = self.calibrated_sensors[json_idx][hidden_name]
                    cur_cam_info["intrinsic"]["K"] = camera_info_cache["new_k"][map_id]
                    cur_cam_info["T_lidar_to_pixel"] = camera_info_cache["lidar_to_pix"][map_id]
                else:
                    cache_idx = len(camera_info_cache["lidar_to_pix"])
                    camera_info_cache["calib_param_to_index"][str(sensor_info)] = cache_idx
                    cur_cam_info = self.calibrated_sensors[json_idx][hidden_name]
                    intrinsic_k = np.array(cur_cam_info["intrinsic"]["K"]).reshape(3, 3)
                    intrinsic_k[:2] /= cur_cam_info["intrinsic"]["resolution"][0] / 1920
                    lidar2pix = get_lidar_to_pixel(sensor_info, intrinsic_k)

                    lidar2pix = np.array(lidar2pix)
                    lidar2pix = lidar2pix.tolist()
                    self.calibrated_sensors[json_idx][hidden_name]["intrinsic"]["K"] = intrinsic_k.tolist()
                    cur_cam_info["T_lidar_to_pixel"] = lidar2pix

                    camera_info_cache["new_k"].append(intrinsic_k.tolist())
                    camera_info_cache["lidar_to_pix"].append(lidar2pix)

    def _get_hidden_name(self, camera_name):
        hidden_name = self.camera_name_mapping_s2h[camera_name]["hidden_name"]
        return hidden_name

    def _get_single_img(self, sensor_data, camera_name):
        hidden_name = self.camera_name_mapping_s2h[camera_name]["hidden_name"]
        s3_path = sensor_data[hidden_name]["s3_path"]
        try:
            img = s3_retry(refile.smart_load_image, [s3_path], retries=5, delay=1)
        except Exception:
            return None
        return img

    @staticmethod
    def _able_to_get_image(name, sensor_data):

        Flag = True
        if name not in sensor_data:
            logger.warning(f"{name} is not in {sensor_data.keys()}")
            Flag = False
        elif sensor_data[name] is None:
            logger.warning(f"{name} in {sensor_data.keys()}, but has no info")
            Flag = False
        elif sensor_data[name]["nori_id"] is None:
            logger.warning(f"The Nori id of {name} is None")
            Flag = False
        return Flag

    def get_images(self, idx, data_dict):
        """
        Loading image and intrinsic mats with given sample index
        """
        imgs, lidar2imgs = [], []
        sensors_info = self.calibrated_sensors[self.loader_output["calibrated_sensors_id"][idx]]
        sensor_data = self.loader_output["frame_data_list"][idx]["sensor_data"]

        if camera_filter(self.raw_names, sensor_data):
            return None

        calibrator = SensorCalibrationInterface(sensors_info)
        trans_ego2cam, trans_lidar2cam, trans_cam2img, trans_lidar2ego, resolution = [], [], [], [], []
        for camera_name in self.camera_names:
            hidden_name = self._get_hidden_name(camera_name)
            img = self._get_single_img(sensor_data, camera_name)
            if img is None:
                return None
            image_resolution = self.target_resolution

            T_lidar_to_pixel = sensors_info[hidden_name]["T_lidar_to_pixel"]
            lidar2img = np.concatenate([np.array(T_lidar_to_pixel), np.array([[0.0, 0.0, 0.0, 1.0]])])
            assert lidar2img is not None, "Incorrect camera name in image info"
            imgs.append(img)
            lidar2imgs.append(np.array(lidar2img, dtype=np.float32))

            trans_ego2cam.append(calibrator.get_ego2cam_trans(hidden_name, inverse=False))
            trans_lidar2cam.append(calibrator.get_lidar2cam_trans(hidden_name))
            trans_cam2img.append(calibrator.get_cam2img_trans(hidden_name))
            trans_lidar2ego.append(calibrator.get_lidar2ego_trans(inverse=False))
            resolution.append(image_resolution)

        data_dict["imgs"] = np.stack(imgs)
        data_dict["lidar2imgs"] = np.stack(lidar2imgs)
        data_dict["cam2img"] = np.stack(trans_cam2img)
        data_dict["ego2cam"] = np.stack(trans_ego2cam)
        data_dict["lidar2ego"] = np.stack(trans_lidar2ego)
        data_dict["lidar2cam"] = np.stack(trans_lidar2cam)
        data_dict["cam_resolution"] = np.stack(resolution)

        return data_dict
