import os
import random
import cv2
import numpy as np
import refile, json
from turbojpeg import TurboJPEG

from tqdm import tqdm

json_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/vlm/data/mtx/model/yolo-world/v1/vlm_train_label_data.json"
json_data = json.load(refile.smart_open(json_path))
local_path_prefix = "./vlm_crop_random"
s3_path_prefix = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/vlm_train_label_data/crop_img/"
jpeg = TurboJPEG()


for nori_id in tqdm(json_data.keys(), desc="[process frames]"):
    img_path = json_data[nori_id]["img_path"]
    img = refile.smart_load_image(img_path)
    if not json_data[nori_id]["labels"]['boxes_label_info']["skipped"]:
        boxes = json_data[nori_id]["labels"]['boxes']
        for box in tqdm(boxes, desc="[process boxes]"):
            rects = box['rects']
            for k, v in rects.items():  # 原json中存的都是2160p，需要转变为1080p
                rects[k] = v / 2
            label = box['class']

            ymin = rects["ymin"]
            ymax = rects["ymax"]
            xmin = rects["xmin"]
            xmax = rects["xmax"]
            lenth = ymax - ymin
            width = xmax - xmin
            area = lenth * width
            if lenth > 180 or width > 180:
                _x = (xmax + xmin) / 2.0
                _y = (ymax + ymin) / 2.0
                x = int(_x)
                y = int(_y)
                target_mask = np.zeros_like(img)
                target_mask[int(ymin) : int(ymax), int(xmin) : int(xmax),...] = 1
                thres = int(max(lenth, width) / 2.0)
                target_mask = target_mask[y-thres:y+thres, x-thres:x+thres, :]
                crop_img = img[y-thres:y+thres, x-thres:x+thres, :]
                crop_img *= target_mask

                try:
                    flag = random.choice([True, False, False, False])
                    s3_save_path = refile.smart_path_join(s3_path_prefix, f"{nori_id}_{label}_{int(area)}.png")
                    with refile.smart_open(s3_save_path, 'wb') as file:
                        file.write(jpeg.encode(crop_img))
                    if flag:  # 随机存一部分用来可视化
                        if not os.path.exists(local_path_prefix):
                            os.mkdir(local_path_prefix)
                        local_save_path = refile.smart_path_join(local_path_prefix, f"{nori_id}_{label}_{int(area)}.png")
                        cv2.imwrite(local_save_path, crop_img)

                except:
                    pass