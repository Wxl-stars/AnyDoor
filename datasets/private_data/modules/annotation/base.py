import numpy as np
from typing import Dict
from abc import abstractmethod
from datasets.private_data.utils.functional import (
    load_angle_anno,
    outlier_filter,
)


class AnnotationBase:
    """
    loader_output: reformat data output , provided by class LoaderBase
    mode: train or val
    """

    def __init__(self, loader_output, mode, label_key="labels") -> None:
        self.loader_output = loader_output
        self.mode = mode
        self.label_key = label_key

    @abstractmethod
    def __getitem__(self, idx) -> Dict:
        raise NotImplementedError


class AnnotationDet(AnnotationBase):
    """
    遮挡定义参考wiki: https://wiki.megvii-inc.com/pages/viewpage.action?pageId=307327004
    对应遮挡阈值：1/2/3/4
    对应遮挡程度：完全可见/轻微遮挡/中度遮挡/严重遮挡
    对应可见比例：100% / 50%～100% / 20%～100% / 0～100%
    """

    def __init__(
        self,
        loader_output,
        mode,
        category_map,
        class_names,
        occlusion_threshold=4,
        filter_outlier_boxes=True,
        filter_outlier_frames=False,
        filter_empty_2d_bboxes=True,
        roi_range=None,
        label_key="labels",
        HF=False,
    ) -> None:
        super(AnnotationDet, self).__init__(loader_output, mode)

        self.category_map = category_map
        self.class_names = class_names

        self.occlusion_threshold = occlusion_threshold
        self.filter_outlier_frames = filter_outlier_frames
        self.filter_outlier_boxes = filter_outlier_boxes
        self.filter_empty_2d_bboxes = filter_empty_2d_bboxes
        self.roi_range = roi_range
        self.label_key = label_key
        self.HF = HF

    @staticmethod
    def _load_single_box(label):
        coor = [
            label["xyz_lidar"]["x"],
            label["xyz_lidar"]["y"],
            label["xyz_lidar"]["z"],
            label["lwh"]["l"],
            label["lwh"]["w"],
            label["lwh"]["h"],
            load_angle_anno(label),
        ]
        return np.array(coor, dtype=np.float32)

    def _judge_whether_outlier_box(self, box_anno, cat_anno):
        if cat_anno not in self.category_map:
            return False
        cur_anno = self.category_map[cat_anno]
        is_outlier = False
        if cur_anno in ["pedestrian"] and (box_anno[3:6] > np.array([3, 3, 3])).any():
            is_outlier = True
        elif cur_anno in ["car", "bus", "bicycle"] and (box_anno[3:6] > np.array([30, 6, 10])).any():
            is_outlier = True
        return is_outlier

    def _get_occlusion_attr(self, anno, camera_keys, around_occluded_mode=False):
        if "2d_bboxes" not in anno:
            if_visible = True
            return if_visible
        mapping = {"严重遮挡": 1, "不可见": 2, "正常": 0, 0: 0, 1: 1, 2: 2, "0": 0, "1": 1, "2": 2, 3: 3}
        if self.HF:
            mapping = {"严重遮挡": 1, "不可见": 2, "正常": 0, 0: 0, 1: 1, 2: 2, 3: 3, "0": 3, "1": 0, "2": 1, "3": 2}
        if_visible = anno["2d_visibility"] if "2d_visibility" in anno else False
        assert "2d_bboxes" in anno, "Illegal anno for occlusion attributes"
        if not self.filter_empty_2d_bboxes and len(anno["2d_bboxes"]) == 0:
            if_visible = True
        for cam_anno in anno["2d_bboxes"]:
            if (around_occluded_mode or cam_anno["sensor_name"] in camera_keys) and int(
                mapping[cam_anno["occluded"]]
            ) < self.occlusion_threshold:
                if_visible = True
                break

        return if_visible

    @property
    def camera_keys(self):
        if "_camera_keys" not in self.__dict__:
            _camera_keys = []
            for item in self.loader_output["camera_name_mapping"]["standard"].values():
                _camera_keys.append(item["hidden_name"])
            self._camera_keys = _camera_keys
        return self._camera_keys

    def _get_single_anno(self, anno):
        # load 3d box annotation
        box_anno = self._load_single_box(anno)
        # load category annotation, considering occlusion or outliers
        cat_anno = anno["category"]
        track_id = anno.get("track_id", -1)
        if track_id == -1:
            return None

        if self.occlusion_threshold > 0 and not self._get_occlusion_attr(anno, self.camera_keys):
            cat_anno = "蒙版"
        if self.filter_outlier_boxes and self._judge_whether_outlier_box(box_anno, cat_anno):
            cat_anno = "蒙版"
        category = self.category_map[cat_anno] if cat_anno in self.category_map else "other"
        # load num_lidar_pt
        num_lidar_info = anno["num_lidar_pts"] if "num_lidar_pts" in anno else 20
        return box_anno, category, num_lidar_info, track_id

    def __getitem__(self, idx):
        data_dict = dict()
        result = {}
        boxes = []
        cats = []
        num_lidar_pts = []
        track_ids = []
        frame_data_list = self.loader_output["frame_data_list"]

        if self.label_key == "label_first":
            label_key = "labels" if "labels" in frame_data_list[idx] else "pre_labels"
        else:
            label_key = self.label_key
        annos = frame_data_list[idx].get(label_key, None)
        if not annos:
            return None
        
        for anno in annos:
            flags = self._get_single_anno(anno)
            if flags is None:
                return None
            box_anno, category, num_lidar_info, track_id = flags
            boxes.append(box_anno)
            cats.append(category)
            num_lidar_pts.append(num_lidar_info)
            track_ids.append(track_id)

        gt_boxes = np.stack(boxes) if len(boxes) > 0 else boxes
        result["gt_boxes"] = np.array(gt_boxes, dtype=np.float32)
        gt_labels = np.stack(cats) if len(cats) > 0 else cats
        result["gt_labels"] = np.array(
            [self.class_names.index(i) if i in self.class_names else -1 for i in gt_labels], dtype=np.int64
        )
        result["labels"] = np.stack(cats) if len(cats) > 0 else cats  # for evaluation
        num_lidar_points = np.stack(num_lidar_pts) if len(boxes) > 0 else num_lidar_pts
        result["num_lidar_points"] = np.array(num_lidar_points, dtype=np.float32)
        result["instance_inds"] = np.array(track_ids, dtype=np.int64)

        if self.filter_outlier_boxes and outlier_filter(
            gt_boxes=result["gt_boxes"], gt_labels=result["gt_labels"], class_names=self.class_names
        ):
            print("ffffffff")
            return None

        # gt_labels = np.zeros((0,), dtype=np.float32)
        # gt_boxes = np.zeros((0, 7), dtype=np.float32)
        data_dict["gt_labels"] = result["gt_labels"]
        data_dict["gt_boxes"] = result["gt_boxes"]
        data_dict["instance_inds"] = result["instance_inds"]

        return data_dict