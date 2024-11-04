# 存储数据集筛选的常用脚本
import refile
import numpy as np
from loguru import logger
from data3d.datasets.private import PrivateDataset
from collections import defaultdict
from ControlNetSDXL.data.private_data.utils.file_io import dump_json, find_all_json


def example():
    dataset = [
        "s3://tf-labeled-res/20220612_det_yueying_checked/ppl_bag_20220612_052914_det",
    ]
    partial_dataset_info = {
        "paths": dataset,
        "filter": None,
        "information": "WM car2 for test",
        "author": "liyadong",
    }
    dump_json(partial_dataset_info, "s3://lyd-tf-share/wm_car2_test.json")


def WMCar4():
    DataDir = ["s3://zf-tf-share/Perceptron/data-zoo/car_4/"]

    all_json = find_all_json(DataDir)
    TestData, TrainData = [], []
    for js in all_json:
        if "bmk" in js:
            TestData.append(js)
        else:
            TrainData.append(js)
    TrainData.sort()

    partial_dataset_info = {
        "paths": TrainData,
        "filter": None,
        "information": "WM car4",
        "author": "zhengfei",
    }
    dump_json(partial_dataset_info, "s3://zf-tf-share/temp/wm_car4_all.json")


WMCar4()


class DataCollect:
    r"""
    Collect original json list for fusion and lidar perception.
    """

    def __init__(
        self,
        car_id=[3, 4],
        data_type=[
            "dp-det_yueying_checked",
            "dp-tracking_yueying_checked",
            "dp-track_yueying_checked",
            "3dbmk-tracking_yueying_checked",
        ],
        quater_list=["22q3", "22q4", "23q1"],
        stop_time="20230201",
        data_prefix=set(["gs", "gszd"] + ["cq", "cqlk"] + ["hd", "qx", "sd"]),
        bmk_type=["GAOSU", "CHENGQU", "OTHER"],
    ) -> None:
        self.car_id = car_id
        self.data_type = data_type
        self.quater_list = quater_list
        self.stop_time = stop_time
        self.data_prefix = data_prefix
        self.bmk_type = bmk_type

    def filter_data_by_skip_label(self, data_paths):
        """
        Returns:
            keep_json_list, skip_json_list
        """
        dataset = PrivateDataset(data_paths=data_paths, only_key_frame=True)
        skip_dicts_list = defaultdict(list)
        for i in range(len(dataset)):
            frame = dataset.frame_data_list[i]
            json_path = dataset.get_scene_name(i)

            if "skip_label" in frame:
                skip_dicts_list[json_path].append(frame["skip_label"])
            else:
                skip_dicts_list[json_path].append(False)
        skip_dicts_list = dict(skip_dicts_list)
        skip_json_list = []
        keep_json_list = []
        for k, v in skip_dicts_list.items():
            if any(v):
                skip_json_list.append(k)
            else:
                keep_json_list.append(k)
        return keep_json_list, skip_json_list

    def dump_standard_paths(self, save_path, all_data, flag="det"):
        """dump data paths

        Args:
            all_data dict(): collected data paths
            save path str: save path
        """
        for key, value in all_data.items():
            partial_dataset_info = {
                "paths": value,
                "filter": None,
                "information": f"WM car{key}",
                "author": "wangpan",
            }
            dump_json(partial_dataset_info, f"{save_path}/car{key}-{flag}.json")

    def collect_bmk_data(self, bmk_list):
        bmk_list_type = {k: {t: [] for t in self.car_id} for k in self.bmk_type}

        bmk_all_list = [p for v in bmk_list.values() for p in v]

        keys = [str.split(x, "/")[-3] for x in bmk_all_list]
        keys_uniq = np.unique(keys)
        logger.info(keys_uniq)
        assert set(keys_uniq) == self.data_prefix, f"key uniq not same, {keys_uniq}"

        # ['cq' 'cqlk' 'gs' 'gszd' 'hd' 'qx' 'sd']
        for car_idx in bmk_list.keys():
            for x in bmk_list[car_idx]:
                cur_key = str.split(x, "/")[-3]
                for t_idx, t in enumerate([["gs", "gszd"], ["cq", "cqlk"], ["hd", "qx", "sd"]]):
                    if cur_key in t:
                        bmk_list_type[self.bmk_type[t_idx]][car_idx].append(x)

        return bmk_list_type

    def collect_origin_data(self, quater_list, car_id_list, type_list):
        det_list = {k: [] for k in car_id_list}
        tracking_list = {k: [] for k in car_id_list}
        bmk_list = {k: [] for k in car_id_list}
        t_list = [det_list, tracking_list, tracking_list, bmk_list]

        for q in quater_list:
            for t_idx, t in enumerate(type_list):
                for car_id in car_id_list:
                    if t_idx == 3:  # bmk
                        path_list = refile.s3_glob(f"s3://tf-{q}-shared-data/labeled_data/car_{car_id}/*/*/ppl*")
                    else:
                        path_list = refile.s3_glob(f"s3://tf-{q}-shared-data/labeled_data/car_{car_id}/*/ppl*")

                    for path in path_list:
                        # filted_by_time
                        time_str = str.split(path, "/")[-1]
                        time_str = time_str[len("ppl_bag_") : len("ppl_bag_") + 8]
                        if time_str >= self.stop_time:
                            continue
                        if t in path:
                            # get newest version
                            version_list = refile.s3_glob(refile.smart_path_join(path, "/*"))

                            if len(version_list) == 0:
                                logger.info(f"empty path: {path}")
                                continue

                            v_list = [
                                str.split(str.rsplit(x, "/", maxsplit=1)[-1], sep="_", maxsplit=1)[0]
                                for x in version_list
                            ]
                            for v in v_list:
                                assert "v" == v[0], f"{version_list}"
                            max_v = max(v_list)
                            idx = v_list.index(max_v)
                            newest_path = version_list[idx]
                            t_list[t_idx][car_id].append(newest_path)

        return (det_list, tracking_list, bmk_list)

    def __call__(self, save_path):
        """Collect original json path, filter empty jsons and save to s3.

        Args:
            save_path (str): s3 path.
        """
        det_list, tracking_list, bmk_list = self.collect_origin_data(self.quater_list, self.car_id, self.data_type)
        bmk_list_type = self.collect_bmk_data(bmk_list)

        # collect all det and tracking data paths
        for v in [det_list, tracking_list]:
            for car_id in self.car_id:
                v[car_id] = sorted([y for x in v[car_id] for y in refile.s3_glob(refile.smart_path_join(x, "*.json"))])

        # collect all bmk data paths
        for bmk_type_i in self.bmk_type:
            for car_id in self.car_id:
                bmk_list_type[bmk_type_i][car_id] = sorted(
                    [
                        y
                        for x in bmk_list_type[bmk_type_i][car_id]
                        for y in refile.s3_glob(refile.smart_path_join(x, "*.json"))
                    ]
                )

        # remove json with all skip_label
        for v in [det_list, tracking_list]:
            for car_id in self.car_id:
                if len(v[car_id]) == 0:
                    continue
                keep_json, skip_json = self.filter_data_by_skip_label(v[car_id])
                v[car_id] = sorted(keep_json)
                logger.info(f"car_id {car_id} skip len", len(skip_json))

        for bmk_type_i in self.bmk_type:
            for car_id in self.car_id:
                if len(bmk_list_type[bmk_type_i][car_id]) == 0:
                    continue
                keep_json, skip_json = self.filter_data_by_skip_label(bmk_list_type[bmk_type_i][car_id])
                bmk_list_type[bmk_type_i][car_id] = keep_json
                logger.info(f"car_id {car_id} bmk_type {bmk_type_i} skip len", len(skip_json))

        # dump_to_json
        self.dump_standard_paths(save_path, det_list, flag="det")
        self.dump_standard_paths(save_path, tracking_list, flag="tracking")
        for key, value in bmk_list_type.items():
            self.dump_standard_paths(save_path, value, flag=f"bmk-{key.lower()}")


data_collect = DataCollect()
data_collect("s3://e2emodel-data/data-collect")
