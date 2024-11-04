import numpy as np

from datasets.private_data.utils.functional import camera_filter
from .base import ImageBase


class ImageStatic(ImageBase):
    def get_images(self, idx, data_dict):
        """
        Loading image and intrinsic mats with given sample index
        """
        imgs = []
        sensor_data = self.loader_output["frame_data_list"][idx]["sensor_data"]

        if camera_filter(self.raw_names, sensor_data):
            return None
        resolution = []
        for camera_name in self.camera_names:
            img = self._get_single_img(sensor_data, camera_name)
            if img is None:
                return None
            image_resolution = self.target_resolution
            imgs.append(img)
            resolution.append(image_resolution)

        data_dict["imgs"] = np.stack(imgs)
        data_dict["cam_resolution"] = np.stack(resolution)

        return data_dict
