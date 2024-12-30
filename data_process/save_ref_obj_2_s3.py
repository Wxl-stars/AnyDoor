import refile, cv2
import os

classes = ["cone", "construction_sign", "fence", "water_horse", "collision_bar", "anti_collision_barrel"] 
img_paths = refile.smart_glob("/gpfs/shared_files/wheeljack/wuxiaolei/projs/AnyDoor/hf_test/pick/*.png")
prefix = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/generated/ref_imgs/"
for path in img_paths:
    name = path.split("/")[-1]
    for c in classes:
        if c in path:
            save_path = refile.smart_path_join(prefix, c, name)
            bos = "aws s3 --endpoint-url http://s3.bj.bcebos.com"
            cmd = f"{bos} cp {path} {save_path}"
            print(cmd)
            os.system(cmd)
