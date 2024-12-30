import refile, json

from tqdm import tqdm

json_path = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/anydoor_generated/random/2024-12-13/random_total.json"
img_prefix = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/anydoor_generated/random/2024-12-13/imgs_800w"

json_data = json.load(refile.smart_open(json_path))

for nori in tqdm(json_data.keys()):
    img_path = json_data[nori]["img_path"]
    if "test" in img_path:
        nori_png = img_path.split("/")[-1]
        nori_png = nori_png.replace(".png", "_fake.png")
        new_img_path = refile.smart_path_join(img_prefix, nori_png)
        if refile.smart_exists(new_img_path):
            json_data[nori]["img_path"] = new_img_path
        else:
            print(new_img_path)

# save
with refile.smart_open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)