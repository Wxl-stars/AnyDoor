from datetime import date
import time
from functools import partial
import os
import re
import cv2
import refile, json, tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from loguru import logger
from multiprocessing import Pool

BEV_RANGE = [-2, 2, 0, 50]
BEV_RESOLUTION = 0.02
LABEL_KEYS = [    
    "小汽车",
    "汽车",
    "货车",
    "大货车",  # !
    "工程车",
    "巴士",
    "摩托车",
    "自行车",
    "三轮车",
    "骑三轮车的人"  # !
    "骑车人",
    "骑行的人",
    "成年人",
    "人",
    "行人",
]

def load_json(json_path):
    with refile.smart_open(json_path, "r") as rf:
        json_data = json.load(rf)
    return json_data

def dump_json(obj, file):
    with refile.smart_open(file, "w") as f:
        f.write(json.dumps(obj))

def is_visible(anno):
    if_visible = anno["2d_visibility"] if "2d_visibility" in anno else False
    assert "2d_bboxes" in anno, "Illegal anno for occlusion attributes"
    for cam_anno in anno["2d_bboxes"]:
        if int(cam_anno["occluded"]) < 3:
            if_visible = True
            break
    return if_visible

def find_all_json(dir_path):
    jsons = []
    for data_dir_item in tqdm.tqdm(dir_path,  desc="[Load Dataset]"):
        if data_dir_item.endswith(".json"):
            jsons.append(data_dir_item)
        else:
            for root, _, files in refile.smart_walk(data_dir_item):
                for file in files:
                    cur_file_path = refile.smart_path_join(root, file)
                    if file.endswith(".json") and "ppl_bag" in cur_file_path:
                        jsons.append(cur_file_path)
    return jsons

def get_date_from_json(json_path):
    match = re.search(r'(\d{8}_\d{6})', json_path)
    datetime_part = match.group(1) if match else None
    return datetime_part.split("_")[-1]

def generate_bev_mask(bboxes, bev_range=BEV_RANGE, resolution=BEV_RESOLUTION):
    """
    Generate BEV mask from 3D bounding boxes.

    Args:
        bboxes (list): List of bounding boxes [(x, y, z, width, length, height, yaw), ...].
        bev_range (tuple): BEV range (xmin, xmax, ymin, ymax).
        resolution (float): Resolution in meters/pixel.

    Returns:
        np.ndarray: BEV mask as a binary image.
    """
    # Define BEV dimensions
    xmin, xmax, ymin, ymax = bev_range
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    bev_mask = np.zeros((height, width), dtype=np.uint8)

    for bbox in bboxes:
        x, y, _, w, l, _, yaw = bbox
        
        # Calculate corners in world coordinates
        corners = np.array([
            [w / 2, l / 2],
            [-w / 2, l / 2],
            [-w / 2, -l / 2],
            [w / 2, -l / 2]
        ])
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T)
        translated_corners = rotated_corners + np.array([x, y])
        
        # Map to BEV pixel coordinates
        pixel_corners = ((translated_corners - [xmin, ymin]) / resolution).astype(np.int32)

        # Fill polygon on the mask
        cv2.fillPoly(bev_mask, [pixel_corners], 1)

    return bev_mask

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

def one_json_process(json_path, old_prefix, new_prefix):
    new_json_path = json_path.replace(old_prefix, new_prefix)
    logger.info(f"old json: {json_path}")
    logger.info(f"new json: {new_json_path}")
    # if refile.smart_exists(new_json_path):
    #     return

    json_date = get_date_from_json(json_path)
    json_data = load_json(json_path)
    new_json_data = {
        "calibrated_sensors": None,
        "frames": [],
        "json_date": json_date,
    }
    new_json_data["calibrated_sensors"] = json_data["calibrated_sensors"]["cam_front_120"]
    for frame in json_data["frames"]:
        # 由于关键帧没有3d标注，无法确定遮挡mask，过滤非关键帧
        if not frame["is_key_frame"]:
            continue
        cur_frame = dict()
        sensor_data = frame["sensor_data"]
        cur_frame["img_path"] = sensor_data["cam_front_120"]['s3_path']
        cur_frame["nori_id"] = sensor_data["cam_front_120"]['nori_id']
        

        # check box的信息
        cur_frame["bbox"] = []
        label_key = "labels" if "labels" in frame else "pre_labels"

        num = len(frame[label_key])
        img_path = sensor_data["cam_front_120"]['s3_path']
        
        img = refile.smart_load_image(img_path)
        frame_id = frame["frame_id"]
        last_name = json_path.split("/")[-1][:-5]
        cv2.imwrite(f"test/{last_name}_{frame_id}_{num}.png", img)

        # 过滤车太多的场景
        if len(frame[label_key]) > 15:
            num = len(frame[label_key])
            logger.info(f"fiter crowded! {num}")
            continue
        for label in frame[label_key]:
            # 过滤掉自车后面的目标
            if label["xyz_lidar"]["y"] < 0:
                continue
            # 过滤掉非静态障碍物
            if label["category"] not in LABEL_KEYS:
                continue
            cur_box = dict()
            coor = [
                label["xyz_lidar"]["x"],
                label["xyz_lidar"]["y"],
                label["xyz_lidar"]["z"],
                label["lwh"]["l"],
                label["lwh"]["w"],
                label["lwh"]["h"],
                load_angle_anno(label),
            ]
            cur_box["coor"] = coor
            cur_box["category"] = label['category']
            cur_frame["bbox"].append(cur_box)
        bboxes = [box["coor"] for box in cur_frame["bbox"]]
        bev_mask= generate_bev_mask(bboxes, BEV_RANGE, BEV_RESOLUTION)
        cur_frame["bev_mask"] = bev_mask.tolist()
        new_json_data["frames"].append(cur_frame)

    # if not refile.smart_exists(new_json_path):
    if True:
        dump_json(new_json_data, new_json_path)
        logger.info(f"FINISH saving to {new_json_path}")

def dump_dataset(path_list, name, info=""):
    partial_dataset_info = {
    "paths": path_list,
    "info": info,
    "filter": None,
    "information": time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),
    "author": "wuxiaolei",
    }
    dump_json(partial_dataset_info, name) 
    logger.info(f"final info save to {name}")


if __name__ == "__main__":
    dir_path = [
        # "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/e2e_city/labeld_data/car_9",  # e2e
        "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/e171/data",  # e171
    ]
    # tgt_dir_path = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/e171/"
    # key_name = "e171_for_static_-2_2_0_50"
    old_prefix = "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/"
    # new_prefix = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/bev_range_-2_2_0_50/"
    new_prefix = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/scene_imgs/e171/"  # e171
    tgt_dir_path = new_prefix

    TODAY = str(date.today())
    debug = False
    multi_process = True

    json_list = find_all_json(dir_path)
    print(len(json_list))

    if debug:
        one_json_process(json_list[0], old_prefix=old_prefix, new_prefix=new_prefix)
        exit()

    if not multi_process:
        for json_path in tqdm.tqdm(json_list):
            one_json_process(json_path)
    else:
        num_pool = int(len(os.sched_getaffinity(0)) * 0.8)
        print(num_pool)
        pool = Pool(processes=10)
        pool.map_async(partial(one_json_process, old_prefix=old_prefix, new_prefix=new_prefix), json_list)
        pool.close()
        pool.join()

    json_list = find_all_json([tgt_dir_path])
    # dump_dataset(json_list, "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/e2e_for_vlm_-2_2_0_50.json", info="只保留了自车前面的框,bev mask是[-2, 2, 0, 50]的box")

    # dump_dataset(json_list, refile.smart_path_join(tgt_dir_path, f"{TODAY}.json"), info="只保留了自车前面的框,bev mask是[-2, 2, 0, 50]的box")

    total_frame = 0
    for path in tqdm.tqdm(json_list, desc="[load data...]"):
        json_data = json.load(refile.smart_open(path))
        total_frame += len(json_data["frames"])
    logger.info(f"total frames: {total_frame}")
    dump_dataset(json_list, refile.smart_path_join(tgt_dir_path, f"{TODAY}-{total_frame}.json"), info="只保留了自车前面的框,bev mask是[-2, 2, 0, 50]的box")

    import IPython; IPython.embed()



    
