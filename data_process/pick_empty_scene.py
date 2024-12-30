import refile, json

from tqdm import tqdm

json_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/vlm/data/mtx/model/yolo-world/v1/vlm_train_label_data.json"
save_path = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/vlm_empty_scene.json"  # 24198帧
json_data = json.load(refile.smart_open(json_path))
new_json_data = dict()
new_json_data["json_info"] = "empty scene for vlm_train_label_data: s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/vlm/data/mtx/model/yolo-world/v1/vlm_train_label_data.jsonn"

for nori_id in tqdm(json_data.keys()):
    # 1. 修改img_path的名字
    img_path = json_data[nori_id]["img_path"]
    if img_path.startswith("s3+baiduyun"):
        img_path.replace("+baiduyun", "")
        json_data[nori_id]["img_path"] = img_path
        
    # 2. 统计label
    if not json_data[nori_id]["labels"]['boxes_label_info']["skipped"]:
        if "boxes" not in json_data[nori_id]["labels"]:
            new_json_data[nori_id] = json_data[nori_id]
        else:
            boxes = json_data[nori_id]["labels"]['boxes']
            if len(boxes) == 0:
                new_json_data[nori_id] = json_data[nori_id]

    else:
        new_json_data[nori_id] = json_data[nori_id]


# save
print("total: ", len(new_json_data.keys()))
with refile.smart_open(save_path, "w") as f:
    json.dump(new_json_data, f, indent=2)
print("peocess over")


