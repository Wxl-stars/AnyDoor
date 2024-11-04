import copy
import inspect
import multiprocessing as mp
import time
from functools import wraps
from typing import Any, Dict
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch import distributed as dist
import torch
from pyquaternion import Quaternion

def is_master():
    if not dist.is_available():
        rank = 0
    elif not dist.is_initialized():
        rank = 0
    else:
        rank = dist.get_rank()
    return rank == 0

def check_parameters(func):
    @wraps(func)
    def wrapper(cfg: dict) -> Any:
        if not isinstance(cfg, dict):
            raise TypeError("The argument 'cfg' must be a dictionary")

        cfg_ = copy.deepcopy(cfg)
        type_ = cfg_.pop("type")
        if not type_:
            raise ValueError("You must define the type of this object!")
        if not callable(type_):
            raise TypeError("The type of this object must be callable!")

        sig = inspect.signature(type_)
        has_keyword_args = False
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_keyword_args = True
                break
            if param.default == inspect.Parameter.empty and param.name not in cfg_:
                raise ValueError(f"{param.name} is a required parameter")

        if not has_keyword_args:
            extra_params = set(cfg_) - set(sig.parameters)
            if extra_params:
                raise ValueError(f"The following parameters were not expected: {', '.join(extra_params)}")

        return func(cfg)

    return wrapper


@check_parameters
def initialize_object(cfg: Dict[str, Any]) -> Any:
    """Initialize Object from a configuration.

    Example:
        >>> cfg = Dict(
        ...     type = InitObject,
        ...     args = args
        ...     ...)
        >>> test = initialize_object(cfg=cfg) # Initialize Object
    """
    cfg_ = {}
    for key, value in cfg.items():
        if key != "loader_output":
            cfg_[key] = copy.deepcopy(value)
        else:
            cfg_[key] = value
    type_ = cfg_.pop("type")
    return type_(**cfg_)


def robust_crop_img(img, crop):
    x1, y1, x2, y2 = crop
    img = cv2.copyMakeBorder(
        img,
        -min(0, y1),
        max(y2 - img.shape[0], 0),
        -min(0, x1),
        max(x2 - img.shape[1], 0),
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img[y1:y2, x1:x2]


def non_gt_filter(gt_boxes, gt_labels, roi_range, **kwargs):
    valid_boxes = gt_boxes[gt_labels != -1]

    if roi_range is not None and len(valid_boxes) > 0:
        mask = (valid_boxes[..., :3] >= roi_range[:3]).all(1)
        mask &= (valid_boxes[..., :3] <= roi_range[3:]).all(1)
        valid_boxes = valid_boxes[mask]

    return len(valid_boxes) == 0


def outlier_filter(gt_boxes, gt_labels, class_names, **kwargs):
    ped_idx = gt_labels == class_names.index("pedestrian")
    ped_boxes = gt_boxes[ped_idx]
    if len(ped_boxes) > 0 and (ped_boxes[:, 3:6] > np.array([3, 3, 3])).any():
        return True
    vehicle_idx = (
        (gt_labels == class_names.index("car"))
        | (gt_labels == class_names.index("bus"))
        | (gt_labels == class_names.index("bicycle"))
    )
    vehicle_boxes = gt_boxes[vehicle_idx]
    if len(vehicle_boxes) > 0 and (vehicle_boxes[:, 3:6] > np.array([30, 6, 10])).any():
        return True


def camera_filter(camera_names, camera_info, **kwargs):
    for k in camera_names:
        if camera_info.get(k, None) is None:
            return True
        if camera_info[k]["s3_path"] is None:
            return True


def frame_washer(annotation, camera_names, start, end, filters, num_workers=16):
    frame_index = annotation.loader_output["frame_index"]
    class_names = annotation.class_names

    def worker(_start, _end, q):
        for dataset_idx, sub_dataset in enumerate(frame_index.datasets[_start:_end]):
            remove_list = []
            for sort_idx, frame_idx in enumerate(sub_dataset):
                annos = annotation.get_annos(frame_idx)
                gt_labels = np.array(
                    [class_names.index(i) if i in class_names else -1 for i in annos["labels"]],
                    dtype=np.float32,
                )
                gt_boxes = np.array(annos["gt_boxes"], dtype=np.float32)

                for filter in filters:
                    filter_tag = filter(
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels,
                        roi_range=annotation.roi_range,
                        class_names=annotation.class_names,
                        camera_names=camera_names,
                        camera_info=annotation.loader_output["frame_data_list"][frame_idx]["sensor_data"],
                    )
                    if filter_tag:
                        remove_list.append(sort_idx)
                        break
            sub_dataset = np.delete(sub_dataset, remove_list)
            q.put((_start + dataset_idx, sub_dataset))
        time.sleep(1)

    manager = mp.Manager()
    result_list = [0] * (end - start)
    q = manager.Queue()
    num_worker = min(num_workers, end - start)
    step = int(np.ceil((end - start) / num_worker))
    process_list = []
    for i in range(num_worker):
        _start = i * step + start
        _end = min((i + 1) * step + start, end)
        t = mp.Process(target=worker, args=(_start, _end, q))
        process_list.append(t)
        t.start()
    for t in process_list:
        t.join()
    while not q.empty():
        idx, result = q.get()
        result_list[idx - start] = result
    return start, end, result_list


# annoation utils
def load_angle_anno(anno):
    """
    四元数转欧拉角
    """
    quat = np.zeros((4,), dtype=np.float32)
    quat[0] = anno["angle_lidar"]["x"]
    quat[1] = anno["angle_lidar"]["y"]
    quat[2] = anno["angle_lidar"]["z"]
    quat[3] = anno["angle_lidar"]["w"]
    return R.from_quat(quat).as_euler("xyz")[-1]


# image utils
def get_sensor_tran_matrix(sensor_extrinsic):
    """
    相机外参转换为矩阵形式
    """
    trans = [sensor_extrinsic["transform"]["translation"][key] for key in ["x", "y", "z"]]
    quats = [sensor_extrinsic["transform"]["rotation"][key] for key in ["x", "y", "z", "w"]]
    trans_matrix = np.eye(4, 4)
    rotation = R.from_quat(quats).as_matrix()
    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = rotation
    trans_matrix[:3, 3] = trans
    return trans_matrix


def get_lidar_to_pixel(sensor_info, intrinsic_k):
    """
    相机参数矩阵，包含内参和外参
    """
    transform = get_sensor_tran_matrix(sensor_info["extrinsic"])
    lidar2pix = np.eye(4)
    lidar2pix[:3, :3] = intrinsic_k
    lidar2pix = (lidar2pix @ transform)[:3].tolist()
    return lidar2pix


def is_volcano_platform():
    if not os.environ.get("MLP_CONSOLE_HOST"):
        return False  # brainpp
    else:
        return True  # volc engine


def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    
    if to_rgb:
        img = img[..., ::-1]  # Convert BGR to RGB
    
    # Normalize each channel by subtracting mean and dividing by std
    img = (img - mean) / std
    return img


class SensorCalibrationInterface(object):
    """
    直接迁移自e5a8f4c4 perceptron/utils/map3d_utils/helpers/calibrator.py
    """

    def __init__(self, calibrated_sensors):
        self.calibrated_sensors = calibrated_sensors

    def get_lidar2ego_trans(self, inverse=False):
        if "lidar2ego" in self.calibrated_sensors:
            lidar2ego_params = self.calibrated_sensors["lidar2ego"].get(
                "transform", self.calibrated_sensors["lidar2ego"]
            )
        else:
            lidar2ego_params = self.calibrated_sensors["lidar_ego"]["extrinsic"]["transform"]
        translation = np.array(list(lidar2ego_params["translation"].values()))
        rotation = Quaternion(list(lidar2ego_params["rotation"].values()))
        trans = self.transform_matrix(translation, rotation, inverse=inverse)
        return trans

    def get_lidar2cam_trans(self, img_key, inverse=False):
        lidar2cam_params = self.calibrated_sensors[img_key]["extrinsic"].get(
            "transform", self.calibrated_sensors[img_key]["extrinsic"]
        )
        translation = np.array(list(lidar2cam_params["translation"].values()))
        rotation = Quaternion(list(lidar2cam_params["rotation"].values()))
        trans_lidar2cam = self.transform_matrix(translation, rotation, inverse=inverse)
        return trans_lidar2cam

    def get_ego2cam_trans(self, img_key, inverse=False):
        if not inverse:
            trans_ego2lidar = self.get_lidar2ego_trans(inverse=True)
            trans_lidar2cam = self.get_lidar2cam_trans(img_key, inverse=False)
            return trans_lidar2cam @ trans_ego2lidar  # 可以把 ego 坐标系下的点转换到 cam 坐标系下
        else:
            trans_lidar2ego = self.get_lidar2ego_trans(inverse=False)
            trans_cam2lidar = self.get_lidar2cam_trans(img_key, inverse=True)
            return trans_lidar2ego @ trans_cam2lidar

    def get_distortion_status(self, img_key):
        return self.calibrated_sensors[img_key]["intrinsic"]["distortion_model"]

    def get_camera_intrinsic(self, img_key):
        cam_intrinsic_params = self.calibrated_sensors[img_key]["intrinsic"]
        _K = np.array(cam_intrinsic_params["K"]).reshape(3, 3)
        _D = np.array(cam_intrinsic_params["D"])
        mode = np.array(cam_intrinsic_params["distortion_model"])
        return _K, _D, mode

    def get_cam2img_trans(self, img_key):
        return self.get_camera_intrinsic(img_key)[0]

    @staticmethod
    def transform_matrix(translation, rotation, inverse=False):
        trans_mat = np.eye(4)
        if not inverse:
            trans_mat[:3, :3] = rotation.rotation_matrix
            trans_mat[:3, 3] = np.transpose(np.array(translation))
        else:
            trans_mat[:3, :3] = rotation.rotation_matrix.T
            trans_mat[:3, 3] = trans_mat[:3, :3].dot(np.transpose(-np.array(translation)))
        return trans_mat

    def get_cam_resolution(self, img_key):
        w, h = self.calibrated_sensors[img_key]["intrinsic"]["resolution"]  # resolution一定要是w,h
        return np.array([w, h])  # (2, )

    def project_xyz2uv(self, xyz_pts, img_key, return_format="tensor"):
        """
        :param xyz_pts: (N, 3)
        :param img_key: str
        :param return_format:
        :return:
        """
        assert isinstance(xyz_pts, (np.ndarray, torch.Tensor))
        assert len(xyz_pts.shape) == 2 and xyz_pts.shape[-1] == 3
        assert return_format in ["numpy", "tensor"]
        # 通过内外参计算xyz投影到uv平面上的坐标
        xyz_pts = xyz_pts if isinstance(xyz_pts, np.ndarray) else xyz_pts.cpu().data.numpy()
        xyz_pts = np.concatenate([xyz_pts, np.ones(xyz_pts.shape[0])[:, None]], axis=-1)  # (N, 4)
        ego2cam_trans = self.get_ego2cam_trans(img_key, inverse=False)
        cam_pts = ego2cam_trans @ xyz_pts.T
        cam_pts = cam_pts[:3]
        cam2img_trans = self.get_cam2img_trans(img_key)
        uv_pts = cam2img_trans @ cam_pts
        uv_pts = uv_pts[:2, :] / uv_pts[2, :]
        uv_pts = uv_pts if return_format == "numpy" else torch.from_numpy(uv_pts)
        # 保证xyz投影到图像上的有效性
        w, h = self.calibrated_sensors[img_key]["intrinsic"]["resolution"]  # resolution一定要是w,h
        valid_indices_u = np.bitwise_and(uv_pts[0] >= 0, uv_pts[0] < w)
        valid_indices_v = np.bitwise_and(uv_pts[1] >= 0, uv_pts[1] < h)
        valid_indices_d = (cam_pts[2] > 0).astype(bool)
        valid_indices_uv = np.bitwise_and(valid_indices_u, valid_indices_v)
        valid_indices = np.bitwise_and(valid_indices_uv, valid_indices_d)
        return uv_pts.T, valid_indices
