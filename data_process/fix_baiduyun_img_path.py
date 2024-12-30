import refile, json

from tqdm import tqdm

json_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/vlm/data/mtx/model/yolo-world/v1/vlm_train_label_data.json"
json_data = json.load(refile.smart_open(json_path))

for nori_id in tqdm(json_data.keys()):
    # 1. 修改img_path的名字
    img_path = json_data[nori_id]["img_path"]
    img_path = img_path.replace("s3+baiduyun", "s3")
    json_data[nori_id]["img_path"] = img_path

# save
with refile.smart_open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)
print("peocess over")


