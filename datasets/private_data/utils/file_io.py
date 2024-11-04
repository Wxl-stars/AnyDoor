import json
import os
import pickle
from collections import defaultdict

import numpy as np
import refile


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.void)):
            return None
        if obj.__class__.__name__ == "ObjectId":
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def load_pkl(file):
    with refile.smart_open(file, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, file):
    with refile.smart_open(file, "wb") as f:
        pickle.dump(obj, f)


def load_json(file):
    with refile.smart_open(file, "r") as f:
        return json.loads(f.read())


def dump_json(obj, file, encoder=None):
    with refile.smart_open(file, "w") as f:
        json.dump(obj, f, indent=2, cls=encoder)


def get_latest_version(json_paths):
    """Given a list of file paths, only contain the latest version files. such as "s3://tf-22q4-shared-data/labeled_data/car_101/20221211_dp-track_yueying_checked/ppl_bag_20221211_134306_det/" only contain version of "v1_230118_144046".

    Args:
        json_paths (list[str]): A list of json paths.

    Returns:
        list[str]: A list of json paths containing only the latest version.
    """
    ppl_bag_mapping = defaultdict(lambda: defaultdict(list))
    re_json_paths = []
    for json_path in json_paths:
        item = json_path.split("/")
        path_1, path_2, json_name = "/".join(item[:-2]), item[-2], item[-1]
        if "ppl_bag" in path_2:
            re_json_paths.append(json_path)
        else:
            ppl_bag_mapping[path_1][path_2].append(json_name)

    for ppl_bag_path, version_dict in ppl_bag_mapping.items():
        versions = list(version_dict.keys())
        latest_version = sorted(versions)[-1]
        re_json_paths.extend(
            [
                os.path.join(ppl_bag_path, latest_version, json_name)
                for json_name in ppl_bag_mapping[ppl_bag_path][latest_version]
            ]
        )
    return re_json_paths


def find_all_json(data_paths):
    return_json_path = []
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    for data_dir_item in data_paths:
        if refile.smart_isdir(data_dir_item):
            json_paths = refile.smart_glob(refile.smart_path_join(data_dir_item, "**.json"))
            return_json_path.extend(json_paths)
        elif data_dir_item.endswith(".json"):
            return_json_path.append(data_dir_item)
        else:
            raise ValueError(f"Json Files / Json files dir expected, but got <{data_dir_item}>")
    return return_json_path
