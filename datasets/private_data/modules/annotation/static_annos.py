from .base import AnnotationBase
import numpy as np


class AnnotationStatic(AnnotationBase):
    def __init__(
            self, 
            loader_output, 
            mode,
            class_map = [
                "cone", 
                "crash_bar", 
                "water_horse", 
                "bucket", 
                "construction_sign", 
                "triangle"],
            class_names = [
                "traffic cone",
                "crash_bar",
                "water barrier",
                "traffic drum",
                "construction sign",
                "triangular road sign",
            ]
        ):
        super(AnnotationStatic, self).__init__(loader_output, mode)
        self.class_map = class_map
        self.class_names = class_names

    def __getitem__(self, idx):
        data_dict = dict()
        frame_info = self.loader_output["frame_data_list"][idx]
        annos = frame_info["labels"]["boxes"]
        boxes = []
        cats = []
        for anno in annos:
            cat = anno["category"]
            imagine = anno["imagine_frame"]
            with_track = anno["with_track"]
            occluded = anno["occluded"]
            # 抄的这里 https://git-core.megvii-inc.com/tukai/fake3d/-/blob/ptq_dev_static/dataset/utils.py#L1074
            if imagine == "no" and with_track != "":
                continue
            if occluded == "occluded_moderate":
                continue
            try:
                cat = self.class_map.index(cat)
            except:
                continue
            # box是按照(2160，3840)标的，这里归一化，再从dataset里返回去
            box = np.array([
                anno["rects"]["xmin"]/3840,
                anno["rects"]["ymin"]/2160,
                anno["rects"]["xmax"]/3840,
                anno["rects"]["ymax"]/2160,
            ])
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            boxes.append(box)
            cats.append(cat)

        gt_boxes = np.stack(boxes) if len(boxes) > 0 else boxes
        gt_labels = np.stack(cats) if len(cats) > 0 else cats

        data_dict["gt_boxes"] = gt_boxes
        data_dict["gt_labels"] = gt_labels

        return data_dict