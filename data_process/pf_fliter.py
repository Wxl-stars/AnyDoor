import time
import uuid
from loguru import logger
import refile
import json
import numpy as np
import cv2
from pyquaternion import Quaternion
from open3d import geometry
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import nori2
import torch
from torch2trt import TRTModule
from tqdm import tqdm
from IPython import embed
import rrun
import concurrent
import random


fetcher = nori2.Fetcher()


def get_2d_box(bbox, extrinsic_matrix, intrinsic_K):
    delta = 0.001

    center = bbox[0:3]
    dim = bbox[3:6]
    yaw = np.zeros(3)
    yaw[2] = bbox[6]
    rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
    box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
    box3d.color = np.clip(box3d.color, 0, 1)
    line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
    vertex = np.array(line_set.points)  # [8,3]
    p = np.dot(extrinsic_matrix, np.c_[vertex, np.ones(len(vertex))].T).T
    if all(p[:, 2] <= 0):
        return None
    p = p[:, :3] / p[:, 3:4]
    pts = p.reshape(-1, 1, 3)

    lines_3d = []
    for x, y in np.array(line_set.lines):
        line = np.stack([pts[x, 0], pts[y, 0]])
        lines_3d.append(line)

    real_lines = []
    for line in lines_3d:
        if all(line[:, -1] > 0):
            line, _ = cv2.projectPoints(line, np.zeros(3), np.zeros(3), intrinsic_K, np.zeros(5))
            real_lines.append(line[:, 0])
        elif any(line[:, -1] > 0):
            interpolate = (delta - line[line[:, -1] <= 0][0, -1]) / (
                line[line[:, -1] > 0][0, -1] - line[line[:, -1] <= 0][0, -1]
            )
            line[line[:, -1] <= 0] = (
                interpolate * (line[line[:, -1] > 0] - line[line[:, -1] <= 0]) + line[line[:, -1] <= 0]
            )
            line, _ = cv2.projectPoints(line, np.zeros(3), np.zeros(3), intrinsic_K, np.zeros(5))
            real_lines.append(line[:, 0])
    points = []
    for line in real_lines:
        pts = np.array(line, dtype=np.int32)
        points.append(pts[0])
        points.append(pts[1])

    points = np.array(points)
    x1 = np.min(points[:, 0])
    y1 = np.min(points[:, 1])
    x2 = np.max(points[:, 0])
    y2 = np.max(points[:, 1])
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


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


def get_sensor_info(calibrated_sensors):
    for sensor_name, sensor_info in calibrated_sensors.items():
        if sensor_name.startswith("cam"):
            intrinsic = sensor_info["intrinsic"]
            K = np.array(intrinsic["K"], dtype=np.float32)
            D = np.array(intrinsic["D"], dtype=np.float32)
            if intrinsic["distortion_model"] == "fisheye":
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K,
                    D,
                    np.eye(3),
                    K,
                    sensor_info["intrinsic"]["resolution"],
                    cv2.CV_16SC2,
                )
            else:
                map1, map2 = cv2.initUndistortRectifyMap(
                    K,
                    D,
                    np.eye(3),
                    K,
                    sensor_info["intrinsic"]["resolution"],
                    cv2.CV_16SC2,
                )

            sensor_info["intrinsic_K"] = K
            sensor_info["map"] = (map1, map2)

            sensor_info["extrinsic_matrix"] = get_sensor_tran_matrix(sensor_info["extrinsic"])     
    return calibrated_sensors

def is_truncated(x1, y1, x2, y2):
    delta = []
    if x1 < 0:
        delta.append(0 - x1)
    if y1 < 0:
        delta.append(0 - y1)
    if x2 > 3940:
        delta.append(x2 - 3840)
    if y2 > 1920:
        delta.append(y2 - 1920)

    thres = max((x2 - x1), (y2 - y1)) * 0.2
    if (np.array(delta) > thres).any():
        return True
    return False

def get_score(x, y, w, h, image, model):
    # cv2.rectangle(image, (int(x - w * 0.5), int(y - h * 0.5)), (int(x + w * 0.5), int(y + h * 0.5)), (0,0,0), 4, 4)
    scales = [0.45, 0.48, 0.5, 0.53, 0.55]
    trans = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
    anchor_list = []
    for tran in trans:
        for scale in scales:
            anchor_list.append(
                    (int(x + tran - w * scale),
                    int(y + 0 - h * scale),
                    int(x + tran + w * scale),
                    int(y + 0 + h * scale))
            )
            anchor_list.append(
                    (int(x + 0 - w * scale),
                    int(y + tran - h * scale),
                    int(x + 0 + w * scale),
                    int(y + tran + h * scale))
            )

    final_score = -1
    final_x1 = None
    final_y1 = None
    
    for (x1, y1, x2, y2) in anchor_list:
        if is_truncated(x1, y1, x2, y2):
            continue
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, 3840)
        y2 = min(y2, 1920)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        crop_img = cv2.resize(image[y1:y2, x1:x2], (176, 176))
        crop_img = torch.from_numpy(crop_img).unsqueeze(0).permute(0, 3, 1, 2).cuda().float()
        score = model(crop_img)[0, :, 0, 0].cpu().numpy()
        if 1.0 - score[0] > final_score:
            final_score = 1.0 - score[0]
            final_x1 = x1
            final_y1 = y1
    return final_score, final_x1, final_y1

def add_pf_score(src_json, target_json, vis_flag):
    model = TRTModule()
    state_dict = torch.load("vehicle_pf_qa.pth")
    model.load_state_dict(state_dict)

    with refile.smart_open(src_json) as f:
        json_data = json.load(f)
    
    calibrated_sensors = deepcopy(json_data["calibrated_sensors"])
    get_sensor_info(calibrated_sensors)
    for frame in tqdm(json_data["frames"]):
        if frame["is_key_frame"]:
            image_data = {}
            for label in frame["labels"]:
                if label["xyz_lidar"]["x"] != 8.034806571373087:
                    continue
                if label["category"] in ["小汽车", "汽车", "货车", "工程车", "巴士", "摩托车", "自行车", "三轮车"]:
                    coor = [
                        label["xyz_lidar"]["x"],
                        label["xyz_lidar"]["y"],
                        label["xyz_lidar"]["z"],
                        label["lwh"]["l"],
                        label["lwh"]["w"],
                        label["lwh"]["h"],
                        load_angle_anno(label),
                    ]

                    for bbox_2d in label["2d_bboxes"]:
                        sensor_name = bbox_2d["sensor_name"]

                        if "_190" in sensor_name or sensor_name == "cam_front_70_right":
                            continue
                        if sensor_name not in image_data:
                            try:
                                nori_img_id = frame["sensor_data"][sensor_name]["nori_id"]
                                img = cv2.imdecode(np.frombuffer(fetcher.get(nori_img_id), dtype=np.uint8), 1)
                                map1, map2 = calibrated_sensors[sensor_name]["map"]
                                img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
                            except Exception as e:
                                print(e)
                                img = None
                            image_data[sensor_name] = img
                        if image_data[sensor_name] is not None:
                            x, y, w, h = get_2d_box(
                                coor, 
                                calibrated_sensors[sensor_name]["extrinsic_matrix"], 
                                calibrated_sensors[sensor_name]["intrinsic_K"]
                            )
                            score, x1, y1, x2, y2 = get_score(x, y, w, h, image_data[sensor_name], model, vis_flag)
                            if score > 0:
                                bbox_2d["pf_score"] = score
                                # print(score)
                                cv2.putText(image_data[sensor_name], "{:.2f}".format(score), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)

                        else:
                            bbox_2d["pf_score"] = 0.0

                for sensor_name, image in image_data.items():
                    if image is not None:
                        cv2.imwrite("aaacc{}_{}_debug.jpg".format(sensor_name, frame["frame_id"]), image)
                        print("{}xxxxx.jpg".format(sensor_name))
                        # exit()

                # embed()
    
    with refile.smart_open(target_json, "w") as f:
        json.dump(json_data, f)
        logger.info(f"dump as {target_json}")

def process_json(json_paths, vis_flag):
    for json_path in json_paths:
        if "camera-perceptron" in json_path:
            new_json_path = json_path.replace("camera-perceptron", "camera-perceptron/pf_data")
        else:
            new_json_path = json_path.replace("s3://", "s3://camera-perceptron/pf_data")
        if refile.smart_exists(new_json_path):
            continue
        add_pf_score(json_path, new_json_path, vis_flag)
    return len(json_paths)



if __name__ == '__main__':
    import logging
    import os

    logging.getLogger("TRT").disabled = True
    logger.add("./log.txt")

    vis_flag = False

    dataset_path = [
        "s3://camera-perceptron/sub_dataset/geely_101/geely_car101_train_beijing_20230606.json",
        "s3://camera-perceptron/sub_dataset/geely_101/geely_car101_train_ningbo_20230606.json",
        "s3://camera-perceptron/sub_dataset/car_101/car101_train_det_20221120-20230522_no_bmk_daytime.json",
        "s3://camera-perceptron/sub_dataset/car_101/car101_train_det_20221120-20230522_no_bmk_daytime.json",
        "s3://camera-perceptron/sub_dataset/car_101/car101_cq_det_train_20230714-20230723.json",
    ]
    path_list = []
    for path in dataset_path:
        with refile.smart_open(path) as f:
            json_paths = json.load(f)["paths"]
            path_list += json_paths
    print(len(path_list))

    path_list = [
        "s3://wuxiaolei/pf_check/geely_cq_20230824/camera-perceptron/pf_data/labeled_data/car_5/20230326_dp-det_yueying_checked/ppl_bag_20230326_193045_det/v0_230330_124631/0012.json"
    ]
    random.shuffle(path_list)
    process_json(path_list[:1], vis_flag)
    # cmd = "cat ./log.txt"
    # os.system(cmd)
    # import IPython; IPython.embed()

    spec = rrun.RunnerSpec()

    time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    spec.name = "pf-filter-%s" % (uuid.uuid1().hex)
    spec.log_dir = f"/data/tmp/rrun_debug{time_info}"
    logger.info(f"rrun log will be save in {spec.log_dir}")
    spec.charged_group = ""
    spec.resources.cpu = 4
    spec.resources.gpu = 1
    spec.resources.memory_in_mb = 51000
    spec.max_wait_duration = "24h"
    spec.minimum_lifetime = 24 * 3600 * int(1e9)
    spec.preemptible = False


    num_jsons = len(path_list)
    num_washers = min(32, num_jsons)
    executor = rrun.RRunExecutor(spec, num_washers, 1)
    step = num_jsons // num_washers
    pbar = tqdm(total=num_jsons, desc="[process json paths]")

    features = []

    for i in range(num_washers):
        start = i * step
        if i == num_washers - 1:
            end = num_jsons
        else:
            end = (i + 1) * step
        
        features.append(executor.submit(
            process_json,
            path_list[start:end],
            vis_flag,
        ))
    
    for feature in concurrent.futures.as_completed(features):
        subset_slice = feature.result()
        pbar.update(subset_slice)