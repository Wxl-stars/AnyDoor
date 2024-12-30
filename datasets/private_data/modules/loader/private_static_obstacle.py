import json

import numpy as np
from torch.utils.data.dataset import ConcatDataset, Subset
from tqdm import tqdm

from datasets.private_data.utils.redis_cache import RedisCachedData, OSSEtagHelper
from datasets.private_data.utils.functional import is_master
from .private_base import LoaderBase
import refile

hf9_lidar2ego = {
    "transform": {
        "translation": {
            "x": 0.0,
            "y": 0.0,
            "z": -0.33000001311302185
        },
        "rotation": {
            "w": -0.7071067811865474,
            "x": -0.0,
            "y": -0.0,
            "z": 0.7071067811865477
        }
    },
    "euler_degree": {
        "RotX": -0.0,
        "RotY": 0.0,
        "RotZ": -90.0
    },
    "calib_status": 0,
    "information": "ego_tf_rfu",
    "calib_time": "2024-10-25 03:16:02"
}

class LoaderStaticObstacle(LoaderBase):
    def islabeled(self, frame):
        labeld = True
        for box in frame["labels"]["boxes"]:
            if box["category"] in ["unlabeled_frame", "null_frame"]:
                labeld = False
                break
        return labeld

    def __call__(self):
        json_collection = self._parse_name_to_json_path()
        assert len(json_collection) > 0, "There should be more one Json Files! Please check!"

        frame_data_list, frame_index = [[]], [[-1]]
        calibrated_sensors, calibrated_sensors_id = {}, []

        cumsum = 0
        oss_etag_helper = OSSEtagHelper(check_etag=is_master())
        bar_name = "[Load Redis Dataset]" if self.use_redis else "[Load OSS Dataset]"
        for json_idx, json_path in enumerate(tqdm(json_collection, disable=(not is_master()), desc=bar_name)):
            if self.use_redis:
                json_data = RedisCachedData(json_path, oss_etag_helper, **self.redis_param)
                frames, idx = json_data["frames"], json_data["key_frame_idx"]
            else:
                json_data = json.load(refile.smart_open(json_path))
                frames = json_data["frames"]
                # 静态障碍物标注时不区分key_frame，这里借用key_frame的逻辑判断一帧是否有效
                idx = [
                    i
                    for i, x in enumerate(frames)
                    if x.get("rect_label_result", True) and not x.get("labeled_wrong", False) and self.islabeled(x) and "sensor_data" in x
                ]

            calibrated_sensors[json_idx] = {
                "cam_front_120": json_data["frames"][0]["cam_calib"],
                "lidar2ego": hf9_lidar2ego,
            }
            
            if len(idx) == 0:
                print(f"filter out json_path {json_path} because of no key frames")
                continue
            if self.only_key_frame:
                frame_data_list.append(Subset(frames, idx))
                idx_len = len(idx)
                frame_index.append(np.arange(idx_len) + frame_index[-1][-1] + 1)
            else:
                frame_data_list.append(frames)
                idx_len = len(frames)
                frame_index.append(np.arange(idx_len) + cumsum)
                cumsum += idx_len
            calibrated_sensors_id += [json_idx] * idx_len
        oss_etag_helper.join()

        self._loader_out = {
            "frame_data_list": ConcatDataset(frame_data_list[1:]),
            "frame_index": ConcatDataset(frame_index[1:]),
            "calibrated_sensors": calibrated_sensors,
            "calibrated_sensors_id": calibrated_sensors_id,
            "json_collection": json_collection,
            "camera_name_mapping": self.camera_name_mapping,
            "cummulative_sizes": ConcatDataset(frame_data_list[1:]).cumulative_sizes,
        }