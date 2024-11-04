import bisect
import math
import warnings
from collections import abc
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..modules.annotation import E2EAnnotations
from ..private.base import DatasetBase

class PrivateMultiModalData(DatasetBase):
    r"""
    Dataset for orin 7v.
    """
    @property
    def is_train(self):
        return self.mode == "train"

    def _sensor_list_check(self) -> None:
        sensor_names = self.sensor_names.keys()
        assert "camera_names" in sensor_names, "sensor_name dict must include camera_names "
        # assert (
        #     "lidar_names" in sensor_names or "radar_names" in sensor_names
        # ), "sensor_name dict must include lidar_names or radar_names"

    def __getitem__(self, index):
        """
        Return dataset's final outputs.
        Note:
            "frame_id":       list of frame index.
            "gt_boxes":       (num_boxes, 7) numpy array, 7 means (x,y,z,l,w,h,angle).
            "gt_labels":      (N, 1) numpy array, box's labels.
            "imgs":           list of multi-view image (in order to compatible with inputs of different shapes).
            "lidar2imgs":     (N, 4, 4) numpy array, camera intrinsic matrix.
            "points":         (N, 4) numpy array, concated lidar points.
            "lidar2ego:       (4, 4) numpy array, lidar to ego extrinsics.
            "ego2global":     (4, 4) numpy array, ego to world extrinsics.
            "bda_mat":        (N, 4, 4) numpy array, loaded by pipeline forward
            "ida_mats":       (N, 4, 4) numpy array, loaded by pipeline forward
        """
        while True:
            frame_idx = self.loader_output["frame_index"][index]
            data_dict = {
                "frame_id": frame_idx,
            }
            # annotation
            annos = self.annotation(index, data_dict)
            if annos is None:
                index = self._rand_index()
                continue

            # image
            if hasattr(self, "image") and "camera_names" in self.sensor_names:
                img_info = self.image.get_images(index, data_dict)
                if img_info is None:
                    index = self._rand_index()
                    continue
            break

        # apply data augment
        data_dict = self.pipeline(data_dict)

        # 这里写法需要改进，需要支持非对称的roi_mask
        data_dict["roi_mask"] = np.asarray(self.roi_mask)
        return data_dict

    @staticmethod
    def collate_fn_fill_batch(data: dict, max_radar_num=200):
        def fill_batch_tensor(batch_data: list):
            if max([len(x) for x in batch_data]) == min([len(x) for x in batch_data]):
                return np.stack([data for data in batch_data])
            else:
                batch_size = len(batch_data)
                batch_length = max([len(x) for x in batch_data])
                for data in batch_data:
                    if data.size != 0:
                        data_shape = data.shape
                        break
                batch_data_shape = (batch_size, batch_length, *data_shape[1:])
                batch_tensor = np.zeros(batch_data_shape)
                for i, data in enumerate(batch_data):
                    if data.size != 0:
                        batch_tensor[i, : len(data)] = data
                return batch_tensor

        def padding_radar(radar_points):
            radar_feats_dim = radar_points.shape[1]
            padding_radar_points = np.zeros((max_radar_num, radar_feats_dim))
            padding_radar_points[: radar_points.shape[0], :] = radar_points[:, :]
            return padding_radar_points

        batch_collection = dict()
        batch_collection["mats_dict"] = dict()
        mats_shape_list = [(3, 3), (4, 3), (4, 4)]
        for key, value in data[0].items():
            if "gt_forecasting" in key or "instance_inds" in key:
                continue
            if key == "gt_boxes":
                data_list = [np.hstack((iter_data[key], np.zeros((iter_data[key].shape[0], 2)))) for iter_data in data]
                batch_collection[key] = fill_batch_tensor(data_list)
            elif key in ["gt_labels", "points"]:
                data_list = [iter_data[key] for iter_data in data]
                batch_collection[key] = fill_batch_tensor(data_list)
            elif key == "radar_points":
                data_list = [padding_radar(iter_data[key]) for iter_data in data]
                batch_collection[key] = fill_batch_tensor(data_list)
            else:
                data_list = [iter_data[key] for iter_data in data]
                if isinstance(value, (list, int, np.int64, np.int32)):
                    batch_collection[key] = np.stack(data_list)
                elif isinstance(value, np.ndarray) and value.shape[-2:] in mats_shape_list:
                    batch_collection["mats_dict"][key] = np.stack(data_list)
                elif isinstance(value, np.ndarray):
                    batch_collection[key] = np.stack(data_list)
                else:
                    batch_collection[key] = data_list
        return batch_collection


class PrivateE2EDataset(PrivateMultiModalData):
    r"""
    Dataset for orin 7v.
    """

    def __init__(
        self,
        num_frames_per_sample=5,
        postcollate_tensorize=False,
        seq_mode=False,
        seq_split_num=1,
        **kwargs
    ):

        for key, transform in kwargs["pipeline"].items():
            transform.update({"multiframe": True})
        if "annotation" in kwargs and isinstance(kwargs["annotation"], list):
            warnings.warn(
                "Task list annotations is deprecated for e2e dataset, should use E2EAnnotations instead.",
                DeprecationWarning,
            )
            super(PrivateE2EDataset, self).__init__(**kwargs)
        else:
            annotations_e2e = kwargs.pop("annotation")
            kwargs["annotation"] = annotations_e2e["box"]
            super(PrivateE2EDataset, self).__init__(**kwargs)
            self.annotation = E2EAnnotations(self.loader_output, self.mode, annotations_e2e)

        self.num_frames_per_sample = num_frames_per_sample

        if isinstance(self.annotation, list):
            for task in self.annotation:
                task.loader_output["calibrated_sensors"] = self.image.loader_output["calibrated_sensors"]
        else:
            self.annotation.loader_output["calibrated_sensors"] = self.image.loader_output["calibrated_sensors"]

        self._init_scene_flag()
        self.seq_mode = seq_mode
        self.seq_split_num = seq_split_num
        if self.seq_mode:
            self._set_sequence_group_flag()

        self.postcollate_tensorize = postcollate_tensorize

    def _init_scene_flag(self):
        """
        organize data in scene for multi gpu eval
        """
        cumulative_size = self.loader_output["cummulative_sizes"]
        scene_len = np.array([0] + cumulative_size)
        scene_len = scene_len[1:] - scene_len[:-1]
        self.scene_flag = []
        self.scene_frame_idx = []
        for i, s in enumerate(scene_len):
            self.scene_flag.extend([i] * s)
            self.scene_frame_idx.extend(list(range(s)))
        self.scene_flag = np.array(self.scene_flag)  # must be np.array
        self.scene_frame_idx = np.array(self.scene_frame_idx)
        assert len(self.scene_flag) == len(self.scene_frame_idx) == len(self)
        self.scene_order = (
            True  # default as False, need to set True when using GroupEachSampleInBatchSampler and multi gpu eval
        )

    def _set_sequence_group_flag(self):
        """Set each sequence to be a different group"""

        assert self.seq_split_num > 1, "each group must be longer than only one frame!"
        bin_counts = np.bincount(self.scene_flag)

        new_flags = []
        group_frame_idx = []

        curr_new_flag = 0
        for curr_flag in range(len(bin_counts)):
            curr_sequence_length = np.array(
                list(range(0, bin_counts[curr_flag], math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                + [bin_counts[curr_flag]]
            )
            for sub_seq_idx in curr_sequence_length[1:] - curr_sequence_length[:-1]:
                for frame_idx in range(sub_seq_idx):
                    new_flags.append(curr_new_flag)
                    group_frame_idx.append(frame_idx)
                curr_new_flag += 1

        assert len(new_flags) == len(self.scene_flag)
        assert len(group_frame_idx) == len(self.scene_frame_idx)
        assert len(np.bincount(new_flags)) == len(np.bincount(self.scene_flag)) * self.seq_split_num
        self.scene_flag = np.array(new_flags, dtype=np.int64)
        self.scene_frame_idx = np.array(group_frame_idx, dtype=np.int64)

    @staticmethod
    def convert_data2tensor(batch_dict) -> None:
        def _cast_to_tensor(inputs: Any, dtype: Optional[np.dtype] = None) -> Any:

            """Recursively convert np.ndarray in inputs into torch.Tensor
            Args:
                inputs: Inputs that to be casted.
                dtype (numpy.dtype): dtype before conversion


            Returns:
                The same recursive structure are remained as inputs, with all contained ndarrays cast to Tensors.
            """

            if isinstance(inputs, torch.Tensor):
                return inputs
            elif isinstance(inputs, np.ndarray):
                # if isinstance(inputs, np.ndarray):
                if dtype is not None:
                    inputs = inputs.astype(dtype)
                if inputs.dtype in [np.uint16]:
                    # unsupported numpy types in torch.tensor
                    inputs = inputs.astype(np.int32)
                return torch.from_numpy(inputs).contiguous()
            elif isinstance(inputs, abc.Mapping):
                return {k: _cast_to_tensor(v, dtype) for k, v in inputs.items()}
            elif isinstance(inputs, abc.Iterable):
                return [_cast_to_tensor(item, dtype) for item in inputs]
            else:
                return inputs

        skip_conversions = ["frame_id", "img_metas", "calib", "image_shape", "gt_bboxes_3d"]

        for key, val in batch_dict.items():
            if key in skip_conversions:
                continue
            batch_dict[key] = _cast_to_tensor(val)

    def get_single_data(self, index: int) -> dict:
        """
        Return dataset's final outputs.
        Note:
            "frame_id":       list of frame index.
            "gt_boxes":       (num_boxes, 7) numpy array, 7 means (x,y,z,l,w,h,angle).
            "gt_labels":      (N, 1) numpy array, box's labels.
            "imgs":           list of multi-view image (in order to compatible with inputs of different shapes).
            "lidar2imgs":     (N, 4, 4) numpy array, camera intrinsic matrix.
            "points":         (N, 4) numpy array, concated lidar points.
            "lidar2ego:       (4, 4) numpy array, lidar to ego extrinsics.
            "ego2global":     (4, 4) numpy array, ego to world extrinsics.
            "bda_mat":        (N, 4, 4) numpy array, loaded by pipeline forward
            "ida_mats":       (N, 4, 4) numpy array, loaded by pipeline forward
        """

        frame_idx = self.loader_output["frame_index"][index]
        data_dict = {
            "frame_id": frame_idx,
        }
        # annotation

        if isinstance(self.annotation, list):
            for task in self.annotation:
                annos = task[index]
                if annos is None:
                    return None
        else:
            annos = self.annotation[index]
            if annos is None:
                # print('88888', self.annotation[index])
                return None

        if annos is not None:  # self.is_train:
            data_dict.update(annos)

        # image
        if hasattr(self, "image") and "camera_names" in self.sensor_names:
            img_info = self.image.get_images(index, data_dict)
            if img_info is None and self.is_train:
                return None

        if hasattr(self, "scene_flag"):
            group_idx = self.scene_flag[index]
            data_dict["scene_idx"] = group_idx
        if hasattr(self, "scene_frame_idx"):
            scene_frame_idx = self.scene_frame_idx[index]
            data_dict["scene_frame_idx"] = scene_frame_idx
        # remove single frame pipeline for E2E dataset
        data_dict["tags"] = self.loader_output["frame_data_list"][index]["tags"]
        return data_dict

    def generate_track_data_indexes(self, index: int) -> list:
        """Choose the track indexes that are within the same sequence"""
        index_list = [i for i in range(index - self.num_frames_per_sample + 1, index + 1)]

        scene_tokens = [bisect.bisect_right(self.loader_output["cummulative_sizes"], i) for i in index_list]

        tgt_scene_token, earliest_index = scene_tokens[-1], index_list[-1]
        for i in range(self.num_frames_per_sample)[::-1]:
            if scene_tokens[i] == tgt_scene_token:
                earliest_index = index_list[i]
            else:
                index_list[i] = earliest_index
        return index_list

    @staticmethod
    def multiframe_to_batch_basic(sample_queue):
        batch = dict()
        for key in sample_queue[0].keys():
            if (
                key in ["imgs", "lane_seg_rv", "sequence_data", "valid_mask_seq", "sequence_data_noise", 
                        "cam2img", "ego2cam", "lidar2cam", "lidar2imgs", "lidar2ego", "ida_mats", "bda_mat"] or key[-5:] == "_imgs"):  # 
                batch[key] = np.stack(
                        [
                            np.stack(
                                sample_queue[i][key],
                                axis=0,
                            )
                            for i in range(len(sample_queue))
                        ],
                        axis=0,
                    )
            elif key in [
                "gt_labels", "gt_boxes", "timestamp",
                "sequence_pe", "fork_idx", "end_idx", 
                "instance_inds", "group_idx", "scene_frame_idx",
                "frame_id", "scene_idx", "tags"
            ]:
                batch[key] = [sample_queue[i][key] for i in range(len(sample_queue))]
        return batch

    def __getitem__(self, index: int) -> dict:
        while True:
            index_list = self.generate_track_data_indexes(index)
            sample_queue = []
            for i in range(len(index_list)):
                if index_list[i] < 0:  # 仅有一个 json 用于训练时，会出现负数 index
                    index = self._rand_index()
                    # print('22222')
                    break
                data_dict = self.get_single_data(index_list[i])
                if data_dict is None:
                    index = self._rand_index()
                    break
                else:
                    sample_queue.append(data_dict)
            if len(sample_queue) == len(
                index_list
            ):  # and sum([len(sample_queue[i]["gt_labels"]) for i in range(len(sample_queue))]) != 0:
                # try:
                sample_queue = self.pipeline(sample_queue)
                break
                # except:
                #     index = self._rand_index()
                #     del sample_queue
                #     continue
            else:
                del sample_queue
        # print(index, index_list)
        assert len(sample_queue) == len(index_list), "len(sample_queue) != len(index_list)"
        # sample_queue = self.pipeline(sample_queue)
        batch = {}  # reorg multi-frame in dict
        batch.update(self.multiframe_to_batch_basic(sample_queue))
        batch["roi_mask"] = np.asarray(self.roi_mask, dtype=np.float32)[np.newaxis]  # (1, 4)
        # NOTE: Method 1: 压缩语义分割数据(int64(8 Byte) -> bool(1 Byte))

        return batch