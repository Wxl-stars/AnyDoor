
import random

from typing import Any, Dict, Optional

import numpy as np

from loguru import logger
from torch.utils.data import Dataset

from datasets.private_data.modules.pipelines.compose import Compose
from datasets.private_data.source.base import BaseCar
from datasets.private_data.modules.annotation.base import AnnotationBase
from datasets.private_data.utils.functional import (
    initialize_object,
)


class DatasetBase(Dataset):
    def __init__(
        self,
        car: BaseCar,
        mode: str = "train",
        sensor_names: list = None,
        loader: Optional[Dict[str, Any]] = None,
        annotation: Optional[Dict[str, Any]] = None,
        pipeline: Optional[Dict[str, Any]] = None,
        evaluator: Optional[Dict[str, Any]] = None,
        image: Optional[Dict[str, Any]] = None,
        roi_mask: list = [-25.6, -85, 25.6, 85],
    ):
        self.mode = mode
        self.car = car
        self.sensor_names = sensor_names
        self._sensor_list_check()

        # Load frame data from Json file
        loader.update({"mode": self.mode})
        self.loader = initialize_object(loader)
        self.loader()
        self.loader_output = self.loader.output

        if isinstance(annotation, dict):
            # Load annotation
            annotation.update(
                {
                    "loader_output": self.loader_output,
                    "mode": self.mode,
                }
            )
            self.annotation: AnnotationBase = initialize_object(annotation)
        else:
            raise NotImplementedError

        # Prepare camera data
        if image is not None and "camera_names" in sensor_names:
            self.camera_names = [camera_name for camera_name in sensor_names["camera_names"]]
            image.update(
                {
                    "loader_output": self.loader_output,
                    "mode": self.mode,
                }
            )
            self.image = initialize_object(image)

        # Process data pipeline
        for key, transform in pipeline.items():
            transform.update({"mode": self.mode})
        self.pipeline = Compose(pipeline)
        if self.mode != "train":
            evaluator.update(
                {
                    "loader_output": self.loader_output,
                    "annotation": self.annotation,
                }
            )
            self.evaluator = initialize_object(evaluator)

        self.roi_mask = roi_mask

    def _sensor_list_check(self) -> None:
        assert "camera_names" in self.sensor_names.keys(), "sensor_name dict must include camera_names "

    def __len__(self):
        return len(self.loader.output["frame_index"])

    def _rand_index(self):
        return random.randint(0, self.__len__() - 1)

    def __getitem__(self, index):
        while True:
            frame_idx = self.loader_output["frame_index"][index]
            data_dict = {
                "frame_id": frame_idx,
            }
            # annotation
            anno_info = self.annotation.get_annos(frame_idx, data_dict)
            if anno_info is None:
                index = self._rand_index()
                logger.warning("Anno in this frame is not valid! Pick another one.")
                continue

            # image
            img_info = self.image.get_images(frame_idx, data_dict)
            if img_info is None:
                index = self._rand_index()
                logger.warning("Image in this frame is not valid! Pick another one.")
                continue
            else:
                break

        # preocess image
        data_dict = self.pipeline(data_dict)

        data_dict["roi_mask"] = np.asarray(self.roi_mask)
        return data_dict

    @staticmethod
    def collate_fn(data: dict):
        """merges a list of samples to form a mini-batch of Tensor(s).

        Args:
            data_dict (dict): samples contain all elements about inputs of network

        Returns:
            dict: mini-batch of network input tensors
        """

        batch_collection = dict()
        batch_collection["mats_dict"] = dict()
        mats_shape_list = [(3, 3), (4, 3), (4, 4)]

        for key, value in data[0].items():
            data_list = [iter_data[key] for iter_data in data]
            if isinstance(value, (list, int, np.int64, np.int32)):
                batch_collection[key] = np.stack(data_list)
            elif isinstance(value, np.ndarray) and value.shape[-2:] in mats_shape_list:
                batch_collection["mats_dict"][key] = np.stack(data_list)
            else:
                batch_collection[key] = data_list

        return batch_collection
