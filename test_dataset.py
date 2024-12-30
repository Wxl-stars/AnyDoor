import os
from tqdm import tqdm
from datasets.private_data.controlnet_sd_private_dataset_static import ControlPrivateStatic

from datasets.private_data.data_cfg.E171_image_static_1v import static_dataset_cfg as dataset_cfg


crop_path = "./crop_obj_new"
if not os.path.exists(crop_path):
    os.mkdir(crop_path)

dataset_cfg["loader"]["datasets_names"] = ["debug"]
dataset = ControlPrivateStatic(**dataset_cfg)

for idx in tqdm(range(len(dataset))):
    data = dataset[idx]
print("process over!!!!")
