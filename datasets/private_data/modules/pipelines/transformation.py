# -------------------------------------------------------------------
# *Author       : tanfeiyang tanfeiyang@megvii.com
# *Date         : 2023-02-16 18:28:07
# *LastEditors  : tanfeiyang tanfeiyang@megvii.com
# *LastEditTime : 2023-02-23 18:18:55
# *FilePath     : /Perceptron/perceptron/data/multimodal/pipelines/transformation.py
# *Description  : TODO
# *Copyright (c) 2023 by tanfeiyang@megvii.com, All Rights Reserved.
# -------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Any, Dict, NewType, Optional

import cv2
import numpy as np

from torch.nn import Module

from ControlNetSDXL.data.private_data.utils.functional import robust_crop_img, imnormalize
from ControlNetSDXL.data.private_data.utils.box_np_ops import center_to_corner_box3d


_Affine_Matrix = NewType("_Affine_Matrix", np.array)


class BaseAugmentation(ABC, Module):
    r"""This Base Augmentation class is used to define the interface for all.

    Example:
        >>> class CustomObject(BaseAugmentation):
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def camera_aug(self, data_dict : Dict, *args : Any, **kwargs : Any):
        ...         ...
        ...     def lidar_aug(self, data_dict : Dict, *args : Any, **kwargs : Any):
        ...         ...
        ...     def radar_aug(self, data_dict : Dict, *args : Any, **kwargs : Any):
        ...         ...
        ...     def forward(self, data_dict : Dict, *args : Any, **kwargs : Any):
        ...         ....
        >>> custom_object = CustomObject()
    """

    def __init__(self, **kwargs: Any):
        ABC.__init__(self)
        Module.__init__(self)
        self.aug_conf: Optional[Dict] = kwargs

    @abstractmethod
    def camera_aug(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("You must rewrite this method!")


    def gt_aug(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: @tanfeiyang  -> For Multi-Task.
        raise NotImplementedError("TODO!")

    def forward(self, data_dict: Optional[Dict], *args: Any, **kwargs: Any) -> Any:
        if "imgs" in data_dict.keys():
            self.camera_aug(data_dict)

    def __repr__(self):
        return "Class name is : {}. ".format(self._get_name()) + "Its aug method paras are : {}.".format(self.aug_conf)


class BevAffineTransformation(BaseAugmentation):
    r"""This class is used to bird view augmentation only for camera.

    Example:

        >>> bda_aug_cfg=dict(
        ...     rot_lim=(-22.5 * 2, 22.5 * 2), # Bev Ratation
        ...     scale_lim=(0.90, 1.10),        # Bev Scale
        ...     trans_lim=(-4, 4),             # Bev Translation
        ...     flip_dx_ratio=0.5,             # Bev Flip of X
        ...     flip_dy_ratio=0.5,             # Bev Flip of Y
        ... )
        >>> bda_aug = BevAffineTransformation(aug_conf=bda_aug_cfg)
    """

    def __init__(
        self,
        aug_conf: Dict[str, float],
        mode: str,
        with_trans_z: bool = False,
        multiframe=False,
    ):
        r"""This method is used to initialize the class.

        Args:
            aug_conf {Dict}:  Augmentation configuration.
            mode {str}: "train" or "val"
            with_trans_z {bool}: use z-trans aug.

        Return:
            None
        """
        super().__init__(aug_conf=aug_conf)
        self.aug_conf = aug_conf
        self.mode = mode
        self.with_trans_z = with_trans_z
        self.multiframe = multiframe

    def sample_augs(self):
        rotate_bda, scale_bda = 0, 1.0
        trans_x, trans_y = 0, 0
        if self.with_trans_z:
            trans_z = 0
        flip_dx, flip_dy = False, False

        if self.mode == "train":
            try:
                rotate_bda = np.random.uniform(*self.aug_conf["rot_lim"])
                scale_bda = np.random.uniform(*self.aug_conf["scale_lim"])
                flip_dx = np.random.uniform() < self.aug_conf["flip_dx_ratio"]
                flip_dy = np.random.uniform() < self.aug_conf["flip_dy_ratio"]
                if "trans_lim" in self.aug_conf:
                    low, high = self.aug_conf["trans_lim"]
                    trans_x = np.random.rand() * (high - low) + low
                    trans_y = np.random.rand() * (high - low) + low
                    if self.with_trans_z:
                        trans_z = np.random.rand() * (high - low) + low
            except KeyError:
                raise KeyError(
                    f"You must set the aug_conf for {self._get_name()} correctly! \
                    Current keys are : {self.aug_conf.keys()}"
                )
        if self.with_trans_z:
            return rotate_bda, scale_bda, (trans_x, trans_y, trans_z), flip_dx, flip_dy
        return rotate_bda, scale_bda, (trans_x, trans_y), flip_dx, flip_dy

    def _bev_transform(self, gt_boxes, rotate_angle, scale_ratio, trans_bda, flip_dx, flip_dy):
        rotate_angle = np.array(rotate_angle / 180 * np.pi, dtype=np.float32)
        rot_sin = np.sin(rotate_angle)
        rot_cos = np.cos(rotate_angle)
        rot_mat = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]], dtype=np.float32)
        # scale_mat = np.eye((3)) * scale_ratio
        scale_mat = np.array([[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]])
        flip_mat = np.eye((3), dtype=np.float32)
        if flip_dx:
            flip_mat = flip_mat @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        if flip_dy:
            flip_mat = flip_mat @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        rot_mat = flip_mat @ (scale_mat @ rot_mat)

        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (rot_mat @ np.expand_dims(gt_boxes[:, :3], -1)).squeeze(-1)
            gt_boxes[:, 0] += trans_bda[0]
            gt_boxes[:, 1] += trans_bda[1]
            if self.with_trans_z:
                gt_boxes[:, 2] += trans_bda[2]
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * np.arcsin(np.array(1.0, dtype=np.float32)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7:] = (rot_mat[:2, :2] @ np.expand_dims(gt_boxes[:, 7:], -1)).squeeze(-1)
        bda_mat = np.zeros((4, 4), dtype=np.float32)
        bda_mat[3, 3] = 1
        bda_mat[0, 3] = trans_bda[0]
        bda_mat[1, 3] = trans_bda[1]
        if self.with_trans_z:
            bda_mat[2, 3] = trans_bda[2]
        bda_mat[:3, :3] = rot_mat

        return gt_boxes, bda_mat

    def camera_aug(self, data_dict: Dict, transform_mat: _Affine_Matrix) -> Any:
        pass

    def lidar_aug(self, data_dict: Dict, transform_mat: _Affine_Matrix) -> Any:
        homogeneous_point = np.ones((data_dict["points"].shape[0], 4))
        homogeneous_point[:, :3] = data_dict["points"][:, :3]
        homogeneous_point_transform = (transform_mat @ homogeneous_point.T).T
        data_dict["points"][:, :3] = homogeneous_point_transform[:, :3]

    def radar_aug(self, data_dict: Dict, transform_mat: _Affine_Matrix) -> Any:
        if data_dict["radar_mode"] == "mlp":
            homogeneous_radar_point = np.ones((data_dict["radar_points"].shape[0], 4))
            homogeneous_radar_point[:, :3] = data_dict["radar_points"][:, :3]
            homogeneous_radar_point_transform = (transform_mat @ homogeneous_radar_point.T).T
            data_dict["radar_points"][:, :3] = homogeneous_radar_point_transform[:, :3]
            # torch.nn.Moduleel trans
            data_dict["radar_points"][:, 6:8] = (transform_mat[:2, :2] @ data_dict["radar_points"][:, 6:8].T).T
        else:
            radar_points = data_dict["radar_points"]
            radar_points[:, :3] = (transform_mat[:3, :3] @ radar_points[:, :3].T).T
            radar_points[:, 6:8] = (transform_mat[:2, :2] @ radar_points[:, 6:8].T).T
            radar_points[:, 8:10] = (transform_mat[:2, :2] @ radar_points[:, 8:10].T).T
            data_dict["radar_points"] = radar_points

    def forward_single(self, data_dict: Dict) -> Dict:
        rotate_bda, scale_bda, trans_bda, flip_dx, flip_dy = self.sample_augs()
        gt_boxes = data_dict["gt_boxes"]
        gt_boxes, transform_mat = self._bev_transform(gt_boxes, rotate_bda, scale_bda, trans_bda, flip_dx, flip_dy)
        if data_dict.get("imgs", None) is not None:
            self.camera_aug(data_dict, transform_mat)
        if data_dict.get("points", None) is not None:
            self.lidar_aug(data_dict, transform_mat)
        if data_dict.get("radar_points", None) is not None:
            self.radar_aug(data_dict, transform_mat)
        data_dict["bda_mat"] = transform_mat
        return data_dict

    def forward(self, data_dict: Dict) -> Dict:
        if not self.multiframe:
            return self.forward_single(data_dict)
        elif isinstance(data_dict, list):
            data_seq = []
            for frame in data_dict:
                data_seq.append(self.forward_single(frame))
            return data_seq
        else:
            raise NotImplementedError


class ImageAffineTransformation(BaseAugmentation):
    r"""This class is used to bird view augmentation for different modal.

    Example:

        >>> ida_aug_conf = {
        ...    "final_dim": final_dim,         # Final image shape
        ...    "resize_lim": (0.772, 1.10),    # Image resize scale
        ...    "H": H,                         # Original image height
        ...    "W": W,                         # Original image width
        ...    "rand_flip": True,              # Image flip
        ...    "bot_pct_lim": (0.0, 0.0),      # Image crop
        ...    "rot_lim": (-5.4, 5.4),         # Image Rotation
        >>> }
        ida_aug = ImageAffineTransformation(aug_conf=ida_aug_cfg)
    """

    def __init__(
        self,
        aug_conf: Dict[str, float],
        camera_names: list,
        mode: str,
        img_norm=False,
        img_conf={},
        gpu_aug: bool = False,
        multiframe=False,
    ):
        r"""This method is used to initialize the class.

        Args:
            aug_conf {Dict}:  Augmentation configuration.

        Return:
            None
        """
        super().__init__(aug_conf=aug_conf)
        self.ida_aug_conf_dict = self._init_ida_aug_conf(aug_conf, camera_names)
        self.mode = mode
        self.img_norm = img_norm
        self.camera_names = camera_names
        self.gpu_aug = gpu_aug
        self.multiframe = multiframe
        if self.img_norm:
            assert (
                img_conf and "img_mean" in img_conf and "img_std" in img_conf and "to_rgb" in img_conf
            ), "invalid image conf input format."
            self.img_norm = img_norm
            self.img_mean = np.array(img_conf["img_mean"], np.float32)
            self.img_std = np.array(img_conf["img_std"], np.float32)
            self.to_rgb = img_conf["to_rgb"]

    def _init_ida_aug_conf(self, aug_conf, camera_names):
        """This function designed for different camera equipped with different aug setting.

        Args:
            >>> camera_key = ["camera_front_120", "camera_front_left_120"]

            >>> ida_aug_conf = {
            ...    "final_dim": (512, 1408),                        # Final image shape
            ...    "resize_lim": [(0.772, 1.10), (0.386, 0.55)],    # Image resize scale: index 0 for camera_key[0]...
            ...    "H": 1080,                                       # Original image height
            ...    "W": 1960,                                       # Original image width
            ...    "rand_flip": True,                               # Image flip
            ...    "bot_pct_lim": [(0.0, 0.0), (0.0, 0.2)],         # Image crop: index 0 for camera_key[0]...
            ...    "rot_lim": (-5.4, 5.4),                          # Image Rotation
            >>> }
        Return:
            ida_aug_conf = {
                "camera_front_120" :{
                    "final_dim": (512, 1408),
                    "resize_lim": (0.772, 1.10),
                    ...
                }
                "camera_front_left_120":{
                    "final_dim": (512, 1408),
                    "resize_lim": (0.386, 0.55),
                }
            }
        """
        ida_aug_conf_dict = {}
        for camera_idx, camera_name in enumerate(camera_names):
            cur_camera_aug = {}
            for aug_key in aug_conf:
                if isinstance(aug_conf[aug_key], list):
                    print(aug_conf[aug_key], camera_names)
                    assert len(aug_conf[aug_key]) == len(
                        camera_names
                    ), f"The lenth of {aug_key} should be equal with camera_names'! "
                    cur_camera_aug[aug_key] = aug_conf[aug_key][camera_idx]
                elif isinstance(aug_conf[aug_key], (tuple, int, float, bool)):
                    cur_camera_aug[aug_key] = aug_conf[aug_key]
                else:
                    raise TypeError(
                        "{} in ida_aug_conf should be list or tuple! But got {} instead!".format(
                            aug_key, type(aug_conf[aug_key])
                        )
                    )
            ida_aug_conf_dict[camera_name] = cur_camera_aug
        return ida_aug_conf_dict

    def _get_crop_hw(self, newH, fH, newW, fW, bot_pct_lim):
        if self.mode == "train":
            crop_h = int((1 - np.random.uniform(*bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        else:
            crop_h = int((1 - np.mean(bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)

        return crop_h, crop_w

    def sample_augs(self, camera_name):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf_dict[camera_name]["H"], self.ida_aug_conf_dict[camera_name]["W"]
        fH, fW = self.ida_aug_conf_dict[camera_name]["final_dim"]
        resize_lim = self.ida_aug_conf_dict[camera_name]["resize_lim"]
        bot_pct_lim = self.ida_aug_conf_dict[camera_name]["bot_pct_lim"]
        if self.mode == "train":
            resize = np.random.uniform(*resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h, crop_w = self._get_crop_hw(newH, fH, newW, fW, bot_pct_lim)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = True if self.ida_aug_conf_dict[camera_name]["rand_flip"] and np.random.choice([0, 1]) else False
            rotate_ida = np.random.uniform(*self.ida_aug_conf_dict[camera_name]["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h, crop_w = self._get_crop_hw(newH, fH, newW, fW, bot_pct_lim)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida
    
    def sample_augs_hw(self, camera_name):  # 处理 h w 方向 resize ratio 不一样的情况
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf_dict[camera_name]["H"], self.ida_aug_conf_dict[camera_name]["W"]
        fH, fW = self.ida_aug_conf_dict[camera_name]["final_dim"]
        resize_lim_h, resize_lim_w = self.ida_aug_conf_dict[camera_name]["resize_lim"]
        
        bot_pct_lim = self.ida_aug_conf_dict[camera_name]["bot_pct_lim"]
        if self.mode == "train":
            resize_h = np.random.uniform(*resize_lim_h)
            resize_w = np.random.uniform(*resize_lim_w)
            resize = (resize_h, resize_w)
            resize_dims = (int(W * resize_w), int(H * resize_h))
            newW, newH = resize_dims
            crop_h, crop_w = self._get_crop_hw(newH, fH, newW, fW, bot_pct_lim)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = True if self.ida_aug_conf_dict[camera_name]["rand_flip"] and np.random.choice([0, 1]) else False
            rotate_ida = np.random.uniform(*self.ida_aug_conf_dict[camera_name]["rot_lim"])
        else:
            # resize = max(fH / H, fW / W)
            # resize_dims = (int(W * resize), int(H * resize))
            resize_h = (resize_lim_h[0] + resize_lim_h[1]) / 2
            resize_w = (resize_lim_w[0] + resize_lim_w[1]) / 2
            resize = (resize_h, resize_w)
            resize_dims = (int(W * resize_w), int(H * resize_h))
            newW, newH = resize_dims
            crop_h, crop_w = self._get_crop_hw(newH, fH, newW, fW, bot_pct_lim)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def get_ida_mat(self, resize, resize_dims, crop, flip, rotate):
        ida_rot = np.eye(2, dtype=np.float32)
        ida_tran = np.zeros(2, dtype=np.float32)
        # post-homography transformation

        # import ipdb
        # ipdb.set_trace()
        
        ida_rot *= resize
        # if not (isinstance(resize, list) or isinstance(resize, tuple)): 
        #     # print('!'*10, resize)
        #     ida_rot *= resize
        # else:
        #     h_ratio, w_ratio = resize
        #     new_resize = (w_ratio, h_ratio)
        #     ida_rot *= new_resize
        ida_tran -= np.array(crop[:2], dtype=np.float32)

        if flip:
            A = np.array([[-1, 0], [0, 1]], dtype=np.float32)
            b = np.array([crop[2] - crop[0], 0], dtype=np.float32)
            ida_rot = np.matmul(A, ida_rot)
            ida_tran = np.matmul(A, ida_tran) + b

        def _get_rot(h):
            return np.array(
                [
                    [np.cos(h), np.sin(h)],
                    [-np.sin(h), np.cos(h)],
                ],
                dtype=np.float32,
            )

        A = _get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]], dtype=np.float32) / 2
        b = np.matmul(A, -b) + b
        ida_rot = np.matmul(A, ida_rot)
        ida_tran = np.matmul(A, ida_tran) + b
        ida_mat = np.zeros((4, 4), dtype=np.float32)
        ida_mat[3, 3] = 1
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 3] = ida_tran
        return ida_mat

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        if self.mode == "train":
            interpolate_candidate = [
                cv2.INTER_LINEAR,
                cv2.INTER_NEAREST,
                cv2.INTER_AREA,
                cv2.INTER_CUBIC,
                cv2.INTER_LANCZOS4,
            ]
            index = np.random.randint(0, 5)
            interpolation = interpolate_candidate[index]
        else:
            interpolation = cv2.INTER_LINEAR
        img = cv2.resize(img, resize_dims, interpolation=interpolation)
        img = robust_crop_img(img, crop)
        if flip:
            img = cv2.flip(img, 1)
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, rotate, 1.0)
        img = cv2.warpAffine(img, M, (width, height))
        if self.img_norm:
            img = imnormalize(img, self.img_mean, self.img_std, self.to_rgb)
            return img.astype(np.float32)
        else:
            return img

    def camera_aug(self, data_dict):
        aug_imgs, ida_mats = [], []
        ida_mats_detail = []
        assert len(data_dict["imgs"]) == len(self.camera_names)
        for img, camera_name in zip(data_dict["imgs"], self.camera_names):
            resize, resize_dims, crop, flip, rotate_ida = self.sample_augs(camera_name)
            # resize, resize_dims, crop, flip, rotate_ida = self.sample_augs_hw(camera_name)
            if not self.gpu_aug:
                img = self._img_transform(
                    img, resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate_ida
                )
            ida_mat = self.get_ida_mat(resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate_ida)
            aug_imgs.append(img)
            ida_mats.append(ida_mat)
            ida_mats_detail.append(
                {
                    "resize": resize,
                    "resize_dims": resize_dims,
                    "crop": crop,
                    "flip": flip,
                    "rotate": rotate_ida,
                }
            )
        return np.stack(aug_imgs), np.stack(ida_mats), ida_mats_detail

    def lidar_aug(self):
        pass

    def radar_aug(self):
        pass

    def forward_single(self, data_dict):
        if "imgs" in data_dict:
            aug_imgs, ida_mats, ida_mats_detail = self.camera_aug(data_dict)
            data_dict["imgs"] = aug_imgs
            data_dict["ida_mats"] = ida_mats
            data_dict["ida_mats_detail"] = ida_mats_detail
        return data_dict

    def forward(self, data_dict):
        if not self.multiframe:
            return self.forward_single(data_dict)
        elif isinstance(data_dict, list):
            data_seq = []
            for frame in data_dict:
                data_seq.append(self.forward_single(frame))
            return data_seq
        else:
            raise NotImplementedError


class MultiFrameImageAffineTransformation(ImageAffineTransformation):
    """
    This class is implemented for multi frame camera view augmentation.
    Augmentation is the same for one camera across different frames.
    """

    def camera_aug(self, sample_queue):
        queue_len = len(sample_queue)

        ida_mats = [[] for i in range(queue_len)]
        ida_mats_detail = []
        assert len(sample_queue[0]["imgs"]) == len(self.camera_names)
        for i in range(len(self.camera_names)):
            resize, resize_dims, crop, flip, rotate_ida = self.sample_augs(self.camera_names[i])
            # resize, resize_dims, crop, flip, rotate_ida = self.sample_augs_hw(self.camera_names[i])
            ida_mats_detail.append(
                {
                    "resize": resize,
                    "resize_dims": resize_dims,
                    "crop": crop,
                    "flip": flip,
                    "rotate": rotate_ida,
                }
            )
            ida_mat = self.get_ida_mat(resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate_ida)
            for j in range(queue_len):
                if not self.gpu_aug:
                    sample_queue[j]["imgs"][i] = self._img_transform(
                        sample_queue[j]["imgs"][i],
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate_ida,
                    )
                else:
                    sample_queue[j]["imgs"][i] = sample_queue[j]["imgs"][i].transpose(2, 0, 1)
                ida_mats[j].append(ida_mat)

        for j in range(queue_len):
            sample_queue[j]["ida_mats"] = np.stack(ida_mats[j])
            sample_queue[j]["ida_mats_detail"] = ida_mats_detail            
            img_semantic_seg = sample_queue[j].get("lane_seg_rv", [])
            if len(img_semantic_seg) > 0:
                lane_seg_list = []
                for lane_seg in img_semantic_seg:
                    lane_seg = self._img_transform(
                        lane_seg, resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate_ida
                    )
                    lane_seg_list.append(lane_seg)
                
                sample_queue[j]["lane_seg_rv"] = np.stack(lane_seg_list)
        return sample_queue

    def forward(self, sample_queue):
        if not self.multiframe:
            raise NotImplementedError
        if "imgs" in sample_queue[0]:
            sample_queue = self.camera_aug(sample_queue)
        return sample_queue


class ObjectRangeFilter(BaseAugmentation):
    """Filter objects by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range, mode, multiframe=False):
        super().__init__()
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.mode = mode
        self.multiframe = multiframe

    @staticmethod
    def mask_points_by_range(points, limit_range):
        mask = (
            (points[:, 0] >= limit_range[0])
            & (points[:, 0] <= limit_range[3])
            & (points[:, 1] >= limit_range[1])
            & (points[:, 1] <= limit_range[4])
        )
        return mask

    @staticmethod
    def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
        if boxes.shape[1] > 7:
            boxes = boxes[:, 0:7]
        corners = center_to_corner_box3d(boxes[:, :3], boxes[:, 3:6], boxes[:, 6], origin=(0.5, 0.5, 0.5), axis=2)
        mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)
        mask = mask.sum(axis=1) >= min_num_corners  # (N)

        return mask

    def camera_aug(self, data_dict):
        pass

    def lidar_aug(self, data_dict):
        mask = self.mask_points_by_range(data_dict["points"], self.point_cloud_range)
        data_dict["points"] = data_dict["points"][mask]

    def radar_aug(self, data_dict):
        mask = self.mask_points_by_range(data_dict["radar_points"], self.point_cloud_range)
        data_dict["radar_points"] = data_dict["radar_points"][mask]

    def forward_single(self, data_dict):
        if len(data_dict.get("gt_boxes", [])) > 0:
            mask = self.mask_boxes_outside_range_numpy(data_dict["gt_boxes"], self.point_cloud_range)
            data_dict["gt_boxes"] = data_dict["gt_boxes"][mask]
            if data_dict.get("gt_names", None) is not None:
                data_dict["gt_names"] = data_dict["gt_names"][mask]
            if data_dict.get("gt_labels", None) is not None:
                data_dict["gt_labels"] = data_dict["gt_labels"][mask]
            if data_dict.get("instance_inds", None) is not None:
                data_dict["instance_inds"] = data_dict["instance_inds"][mask]
            if data_dict.get("predict_attribute", None) is not None:
                for key in list(data_dict["predict_attribute"].keys()):
                    data_dict["predict_attribute"][key] = data_dict["predict_attribute"][key][mask]
        return data_dict

    def forward(self, data_dict):
        if not self.multiframe:
            return self.forward_single(data_dict)
        elif isinstance(data_dict, list):
            data_seq = []
            for frame in data_dict:
                data_seq.append(self.forward_single(frame))
            return data_seq
        else:
            raise NotImplementedError