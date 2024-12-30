import random
import cv2
import refile, json

from tqdm import tqdm

json_path = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/anydoor_generated/random/2024-11-29/_2024-11-29_fill_fake_random_0.json"

json_data = json.load(refile.smart_open(json_path))

import os

if not os.path.exists("test_check"):
    os.makedirs("test_check")

for nori_id in tqdm(json_data.keys()):
    labels = json_data[nori_id]["labels"]
    img_path = json_data[nori_id]["img_path"]

    img = refile.smart_load_image(img_path)

    boxes = labels["boxes"]
    mask_list = []
    if len(boxes) > 1:
        if random.choice([True, False]):
            for box in boxes:  # 远的在前，近的在后
                xmin = box["rects"]["xmin"]
                ymin = box["rects"]["ymin"]
                xmax = box["rects"]["xmax"]
                ymax = box["rects"]["ymax"]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax),  (0, 255, 0), 2, 2)
                cv2.putText(img, box['occluded'], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.imwrite(f"test_check/{nori_id}.png", img)