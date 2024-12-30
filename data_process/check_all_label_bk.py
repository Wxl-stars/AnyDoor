import argparse
from datetime import date
import refile, json
import pandas as pd

from tqdm import tqdm




# json_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/vlm/data/mtx/model/yolo-world/v1/vlm_train_label_data.json"
# # PRE_FIX = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/"
# SEARCH_KEY = "e2e_for_vlm_-2_2_0_50_2024-11-21_fill_fake"

# # PRE_FIX = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/static/bev_range_-2_2_0_50"
# # SEARCH_KEY = "2024-11-27_2024-11-28_fill_fake_random"

# PRE_FIX = f"s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/anydoor_generated/{SEARCH_KEY}/{DATE}" 
# # TODAY = str(date.today())

# # search json_path:
# print(PRE_FIX)
# json_paths = list(refile.smart_glob(refile.smart_path_join(PRE_FIX, "*.json")))
# print("all json_path")
# for path in json_paths:
#     print(path)
# print("--------------------------------")


# # save_info_path = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/e2e_for_vlm_-2_2_0_50_fill_fake_label_info.json"
# save_info_path = refile.smart_path_join(PRE_FIX, f"{SEARCH_KEY}_total_label_info.json")
# save_total_path = refile.smart_path_join(PRE_FIX, f"{SEARCH_KEY}_total.json")

json_paths = ["s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/e2e_for_vlm_-2_2_0_50_2024-11-21_fill_fake_total_fix.json"]

# load data info
json_data = dict()
for path in json_paths:
    # if "total" not in path:
    json_data.update(json.load(refile.smart_open(path)))

label_info = dict()
label_info_key = ['class', 'description', 'influence', 'occluded', 'blurred']
# init label_info
for key in label_info_key:
    label_info[key] = dict()

for nori_id in tqdm(json_data.keys()):
    # 1. 修改img_path的名字
    # img_path = json_data[nori_id]["img_path"]
    # img_path.replace("+baiduyun", "")
    # 2. 统计label
    try:
        flag = not json_data[nori_id]["labels"]['boxes_label_info']["skipped"]
    except:
        flag = True
    if flag:
        boxes = json_data[nori_id]["labels"]['boxes']
        for box in boxes:
            for k, v in box.items():
                if k in label_info_key:
                    if v not in label_info[k]:
                        label_info[k][v] = 0
                    else:
                        label_info[k][v] += 1

# print
print("--------------------------------")
for key in label_info:
    print(f"{key}: ")
    if isinstance(label_info[key], dict):
        for k, v in label_info[key].items():
            print(f"{k}: {v}")
    else:
        print(label_info[key])
    print("--------------------------------")

# save label info 
with refile.smart_open(save_info_path, "w") as f:
    json.dump(label_info, f, indent=2)
print(f"label info path: {save_info_path}")


# save merged data
if len(json_paths) > 1:
    with refile.smart_open(save_total_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"json_path: {save_total_path}")
else:
    print(f"json_path: {json_paths[0]}")

print("peocess over")

