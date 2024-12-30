import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录
parent_dir = os.path.dirname(current_dir)
# 将上层目录添加到 sys.path
sys.path.insert(0, parent_dir)
from datasets.private_data.utils.file_io import dump_json


obj_info = {
    "cone": {
        "lwh": [0.45, 0.45, 0.75],
        "ref_path": [
            "/gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/test_0_image_mask.png",
        ]
    }, 
    "fzt": {
        "lwh": [0.63, 0.63, 0.58],
        "ref_path": [
            "/gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/test_1730710957_image_mask.png",
        ]
        },
    "shuima": {
        "lwh": [0.72, 0.3, 1.2],
        "ref_path": [
            "/gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/test_1730710863_image_mask.png",
        ]
        },
    "fangzhuangzhu": {
        "lwh": [0.17, 0.17, 0.67],
        "ref_path": [
            "/gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/test_1730711068_image_mask.png",
        ]
        },
}
json_path = "/gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/ref_obj_info.json"
dump_json(obj_info, json_path)
print(f"save as {json_path}")