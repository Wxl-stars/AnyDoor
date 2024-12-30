import refile, json
import cv2
import numpy as np
import random

from tqdm import tqdm

json_path = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/e2e_for_vlm_-2_2_0_50_2024-11-21_fill_fake_total.json"

json_data = json.load(refile.smart_open(json_path))

import os

if not os.path.exists("test"):
    os.makedirs("test")

for nori_id in tqdm(json_data.keys()):
    labels = json_data[nori_id]["labels"]
    img_path = json_data[nori_id]["img_path"]

    img = refile.smart_load_image(img_path)

    boxes = labels["boxes"]
    mask_list = []
    if len(boxes) > 1:
        i = 0
        for box in boxes:  # 远的在前，近的在后
            mask = np.zeros_like(img)[:, :, 0]
            xmin = box["rects"]["xmin"]
            ymin = box["rects"]["ymin"]
            xmax = box["rects"]["xmax"]
            ymax = box["rects"]["ymax"]
            mask[ymin:ymax, xmin:xmax] = 1
            mask_list.append(mask)
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax),  (0, 255, 0), 2, 2)

            # # cv2.imwrite(img, "test.png")
            # cv2.imwrite("test.png", img)

        for i in range(len(boxes)):
            cur_box = boxes[i]
            src_mask = mask_list[i]
            ratio = 0
            for j in range(i+1, len(boxes)):
                if i == j:
                    continue
                tgt_mask = mask_list[j]
                # x = src_mask & tgt_mask
                # cv2.imwrite(f"test_{i}_{j}.png", x * 255)
                ratio = max((src_mask & tgt_mask).sum() / src_mask.sum(), ratio)
            # print(i, j, ratio)
            if ratio == 0:
                cur_box['occluded'] = "occluded_none"
            elif ratio < 0.3:
                cur_box['occluded'] = "occluded_mild"
            elif ratio >= 0.65:
                cur_box['occluded'] = "occluded_moderate"
            else:
                cur_box['occluded'] = "occluded_full"
                
        if random.choice([True, False]):
            for box in boxes:  # 远的在前，近的在后
                xmin = box["rects"]["xmin"]
                ymin = box["rects"]["ymin"]
                xmax = box["rects"]["xmax"]
                ymax = box["rects"]["ymax"]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax),  (0, 255, 0), 2, 2)
                cv2.putText(img, box['occluded'], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.imwrite(f"test/{nori_id}.png", img)

import IPython; IPython.embed()

