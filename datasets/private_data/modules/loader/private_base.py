import bisect
import json
import os
from collections import defaultdict

import numpy as np
from torch.utils.data.dataset import ConcatDataset, Subset
from tqdm import tqdm

from datasets.private_data.utils.redis_cache import RedisCachedData, OSSEtagHelper
from datasets.private_data.utils.functional import is_master
from datasets.private_data.utils.file_io import get_latest_version
import refile


class LoaderBase:
    def __init__(self, car, camera_names, mode, datasets_names, only_key_frame=True, rebuild=False, use_redis=True) -> None:
        self.car = car
        self.camera_names = camera_names
        self.mode = mode
        self.datasets_names = datasets_names
        self.only_key_frame = only_key_frame
        self.redis_param = {"rebuild": rebuild}
        self.use_redis = use_redis
        self._parse_camera_name_mapping()

    @property
    def output(self):
        return self._loader_out

    @property
    def camera_name_mapping(self):
        return self._camera_name_mapping

    def _load_primary_paths(self, dataset_name):
        if self.mode == "train":
            json_path = self.car.trainset_partial[dataset_name]
        else:
            json_path = self.car.benchmark_partial[dataset_name]

        if isinstance(json_path, str):
            json_data = json.load(refile.smart_open(json_path))
            if isinstance(json_data, list):
                json_paths = json_data
            else:
                json_paths = json_data["paths"]
                assert "information" in json_data and json_data["information"], "dataset should be described."
                assert "author" in json_data and json_data["author"], "author should be declared."
        elif isinstance(json_path, list):
            json_paths = json_path
        else:
            raise TypeError("Only json file (str type) and json list (list type) are supported!")

        assert len(json_paths) > 0, f"There should be more than one json in {json_path}"
        return json_paths

    def _parse_name_to_json_path(self):
        json_collection = []
        for dataset_name in self.datasets_names:
            primary_paths = self._load_primary_paths(dataset_name)  # [:10]
            for primary_path in primary_paths:
                # assert refile.smart_exists(primary_path), "dataset path don't exists."
                if primary_path.endswith("json"):
                    json_collection.append(primary_path)
                else:
                    json_paths = refile.smart_glob(os.path.join(primary_path, "**[0-9].json"))
                    json_collection.extend(json_paths)
        return get_latest_version(json_collection)

    def _parse_camera_name_mapping(self):
        camera_name_mapping_h2s = defaultdict(dict)
        camera_name_mapping_s2h = defaultdict(dict)

        for sensor_name in self.camera_names:
            if not sensor_name.startswith("cam"):
                continue
            sensor_ins = getattr(self.car.sensors, sensor_name)
            camera_name_mapping_h2s[sensor_ins.name] = {
                "standard_name": sensor_name,
                "resolution": sensor_ins.resolution,
            }
            camera_name_mapping_s2h[sensor_name] = {
                "hidden_name": sensor_ins.name,
                "resolution": sensor_ins.resolution,
            }
        self._camera_name_mapping = {"hidden": camera_name_mapping_h2s, "standard": camera_name_mapping_s2h}

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
                idx = [
                    i
                    for i, x in enumerate(frames)
                    if x["is_key_frame"] and x.get("fetched_back", True) and not x.get("skip_label", False)
                ]
            calibrated_sensors[json_idx] = json_data["calibrated_sensors"]

            if self.only_key_frame:
                frame_data_list.append(Subset(frames, idx))
                idx_len = len(idx)
                if idx_len == 0:
                    print(f"filter out json_path {json_path} because of no key frames")
                    continue
                frame_index.append(np.arange(idx_len) + frame_index[-1][-1] + 1)
            else:
                frame_data_list.append(frames)
                idx_len = len(frames)
                frame_index.append(np.arange(idx_len) + cumsum)
                cumsum += idx_len
            calibrated_sensors_id += [json_idx] * idx_len
            break
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

    def get_scene_name(self, idx):
        # 定位当前idx数据存在于哪个json文件，json文件地址作为 scene_name。
        json_idx = bisect.bisect_right(self.output["frame_data_list"].cumulative_sizes, idx)
        json_path = self.output["json_collection"][json_idx][1]
        return json_path
