import numpy as np

from datasets.private_data.utils.functional import SensorCalibrationInterface, camera_filter
from .base import ImageBase


class ImageStatic(ImageBase):
    def get_images(self, idx, data_dict):
        """
        Loading image and intrinsic mats with given sample index
        """
        imgs = []
        sensors_info = self.calibrated_sensors[self.loader_output["calibrated_sensors_id"][idx]]
        sensor_data = self.loader_output["frame_data_list"][idx]["sensor_data"]

        if camera_filter(self.raw_names, sensor_data):
            return None
        
        calibrator = SensorCalibrationInterface(sensors_info)
        resolution, trans_ego2cam, trans_lidar2cam, lidar2imgs = [], [], [], []
        for camera_name in self.camera_names:
            hidden_name = self._get_hidden_name(camera_name)
            img = self._get_single_img(sensor_data, camera_name)
            if img is None:
                return None
            image_resolution = self.target_resolution

            T_lidar_to_pixel = sensors_info[hidden_name]["T_lidar_to_pixel"]
            lidar2img = np.concatenate([np.array(T_lidar_to_pixel), np.array([[0.0, 0.0, 0.0, 1.0]])])
            imgs.append(img)
            resolution.append(image_resolution)
            trans_ego2cam.append(calibrator.get_ego2cam_trans(hidden_name, inverse=False))
            trans_lidar2cam.append(calibrator.get_lidar2cam_trans(hidden_name))
            lidar2imgs.append(np.array(lidar2img, dtype=np.float32))

        data_dict["imgs"] = np.stack(imgs)
        data_dict["cam_resolution"] = np.stack(resolution)
        data_dict["ego2cam"] = np.stack(trans_ego2cam)
        data_dict["lidar2imgs"] = np.stack(lidar2imgs)
        data_dict["lidar2cam"] = np.stack(trans_lidar2cam)

        return data_dict
