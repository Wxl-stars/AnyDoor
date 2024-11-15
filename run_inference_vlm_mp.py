import copy
import os
import cv2
import einops
import numpy as np
import torch
import random
import argparse
import refile, json
import random
import re
from pytorch_lightning import seed_everything
from tqdm import tqdm
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
from pyquaternion import Quaternion
from open3d import geometry
from datetime import date
from turbojpeg import TurboJPEG

"""
--scene_json s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/vlm_empty_scene.json
# --ref_json s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/generated/ref_obj_info.json
--ref_json /gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/ref_obj_info.json
--save_path s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/vlm_empty_scene_fill.json
"""


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

RFU_CORE_BOX = [-2, 2, 10, 50]  # 左右后前
CYCLE_NUMS = 3
HALF_LANE = 1.875
# Y_THREH_MIN = 30
IMG_H = 1080
IMG_W = 1920
PREFIX = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/anydoor_generated/"
TODAY = str(date.today())
BOX_TEMP = {
    'track_index': 1,
    'class': 'construction_sign',  # need to change
    'description': 'non_pass|static',
    'influence': 'keep',  # need to change
    'occluded': 'occluded_none',  # image editing 贴出来先都默认无遮挡（TODO:可以根据2d mask， 3dmask的关系来判断）
    'blurred': '',
    'rects': {  # need to change
        'xmin': 1619.411376953125,
        'ymin': 1125.208740234375,
        'xmax': 1698.4967041015625,
        'ymax': 1184.5228271484375
        }
  }

cam_front_120_intrinsic = {
    "resolution": [
        3840,
        2160
    ],
    "distortion_model": "fisheye",
    "K": [
        [
            2431.34822198258,
            0.0,
            1915.6344415190715
        ],
        [
            0.0,
            2435.8935271608184,
            1057.1992218239238
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ],
    "D": [
        [
            -0.33283291909412016
        ],
        [
            0.18082653911290322
        ],
        [
            -0.09923141674697789
        ],
        [
            0.024348077124114128
        ]
    ]
}
 
cam_front_120_extrinsic = {
    "transform": {
        "translation": {
            "x": 0.027138140679381053,
            "y": 1.64892744897775,
            "z": -1.664214771961789
        },
        "rotation": {
            "w": -0.7041052687496392,
            "x": -0.7100094615014261,
            "y": 0.010018740197583975,
            "z": -0.0046861436498590045
        }
    },
    "euler_degree": {
        "RotX": 90.48288573101871,
        "RotY": -0.42709141644429616,
        "RotZ": 1.1933543231435044
    },
    "calib_status": 0,
    "information": "cam_front_120_tf_rfu",
    "calib_time": "2024-10-25 03:16:02"
}


def get_trans_rfu2cam(cam_front_120_extrinsic):
    translation = np.array(list(cam_front_120_extrinsic['transform']["translation"].values()))
    rotation = Quaternion(list(cam_front_120_extrinsic['transform']["rotation"].values()))
    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = rotation.rotation_matrix
    trans_matrix[:3, 3] = np.transpose(np.array(translation))
    return trans_matrix

def get_camera_intrinsic(cam_front_120_intrinsic, resize_1080p=True):
    K = np.array(cam_front_120_intrinsic["K"]).reshape(3, 3)
    D = np.array(cam_front_120_intrinsic["D"])
    resolution = np.array(cam_front_120_intrinsic["resolution"])
    # new_intrinsic_K, _ = cv2.getOptimalNewCameraMatrix(
    #     K, D, resolution, alpha=1.0
    # )
    new_intrinsic_K = K
    if resize_1080p:
        scale = resolution[0] / 1920
        new_intrinsic_K[:2] = new_intrinsic_K[:2] / scale

    return new_intrinsic_K

def get_3d_vertex(center, dim, yaw, transform, K):
    rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
    box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
    box3d.color = np.clip(box3d.color, 0, 1)
    line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
    vertex = np.array(line_set.points)
    # 转到相机系
    cur_camera_points = np.dot(transform, np.c_[vertex, np.ones(len(vertex))].T).T[:, :3]
    cur_camera_points = cur_camera_points[cur_camera_points[:, 2] > 0, :]
    uv = np.matmul(K, cur_camera_points.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    points_2d = uv

    # 找到外接矩形的最小和最大 x、y 值
    x_min = np.min(points_2d[:, 0])
    x_max = np.max(points_2d[:, 0])
    y_min = np.min(points_2d[:, 1])
    y_max = np.max(points_2d[:, 1])
    # 生成外接矩形框的左上角坐标和宽高
    top_left = (x_min, y_min)
    width = x_max - x_min
    height = y_max - y_min
    return int(x_min), int(y_min), int(x_max), int(y_max)


def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)  # 背景变白
    # cv2.imwrite("test/mask_ref_image.png", masked_ref_image)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    # 加缩放aug
    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    # print("0: ", tar_box_yyxx, tar_box_yyxx[1] - tar_box_yyxx[0], tar_box_yyxx[3] - tar_box_yyxx[2])
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])
    # print("1: ", tar_box_yyxx, tar_box_yyxx[1] - tar_box_yyxx[0], tar_box_yyxx[3] - tar_box_yyxx[2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    # print("0: ", tar_box_yyxx_crop, tar_box_yyxx_crop[1] - tar_box_yyxx_crop[0], tar_box_yyxx_crop[3] - tar_box_yyxx_crop[2])
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    # print("0: ", tar_box_yyxx_crop, tar_box_yyxx_crop[1] - tar_box_yyxx_crop[0], tar_box_yyxx_crop[3] - tar_box_yyxx_crop[2])
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    # 将ref resise到target box的大小
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale = 5.0):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    # 'extra_sizes': [512, 512, 512, 512]
    # 'tar_box_yyxx_crop': [  0, 512,   0, 512]
    ref = item['ref'] * 255  # ref object (224, 224, 3)
    tar = item['jpg'] * 127.5 + 127.5  # scene img (512, 512, 3)
    hint = item['hint'] * 127.5 + 127.5  # scene_img stich with hf_map
    # cv2.imwrite("test_hint.png", hint)

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda()   # scene_img stich with hf_map
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda()  # ref object 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    return gen_image


def get_date_from_json(json_path):
    match = re.search(r'(\d{8}_\d{6})', json_path)
    datetime_part = match.group(1) if match else None
    return datetime_part.split("_")[-1]

def get_label(coor_3d):
    label = None
    _x = min(abs(coor_3d[0] - 0.8), abs(coor_3d[0] + 0.8))  # 与自车轮廓的横向最近距离
    if abs(coor_3d[0]) > HALF_LANE:  # 目标不在自车车道
        label = "keep"
    elif abs(coor_3d[1]) > 150 or abs(coor_3d[0]) > 1:  # 离自车纵向距离过远或者横向距离过大
        label = "keep"
    elif _x > 1 and abs(coor_3d[0]) < HALF_LANE:  # 和自车轮廓最近距离超过1m且不压车道线
        label = "avoid"
    else:
        label = "stop"
    return label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scripts to generate data for vlm using AnyDoor"
    )
    parser.add_argument("--scene_json", type=str, required=True, default=None)
    parser.add_argument("--ref_json", type=str, required=True, default=None)
    parser.add_argument("--save_path", type=str, required=False, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    # 1. Preparasion 
    args = parse_args()
    if args.save_path is not None:
        json_last_name = args.save_path.split("-")[-1]
        json_last_name = json_last_name.replace(".json", "_fill_fake.json")
        json_save_path = refile.smart_path_join(PREFIX, TODAY, json_last_name)
    else:
        json_save_path = args.scene_json.replace(".json", "_fill_fake.json")

    # prepare json data
    scene_data = json.load(refile.smart_open(args.scene_json))
    scene_data.pop("json_info")
    ref_data = json.load(refile.smart_open(args.ref_json))
    # prepare transformation
    trans_rfu2cam = get_trans_rfu2cam(cam_front_120_extrinsic)
    K = get_camera_intrinsic(cam_front_120_intrinsic)

    new_scene_data = dict()
    jpeg = TurboJPEG()

    # 2. prepare(random create) input data
    ref_obj_classes = list(ref_data.keys())
    scene_keys = list(scene_data.keys())
    for nori_id in tqdm(scene_keys[:1000]):
        gen_image = None
        new_scene_data[nori_id] = copy.deepcopy(scene_data[nori_id])
        new_scene_data[nori_id]["labels"]['boxes_label_info']["skipped"] = False
        ori_json_path = new_scene_data[nori_id]['json_path']
        json_date = get_date_from_json(ori_json_path)
        # prepare scene img
        scene_img_path = scene_data[nori_id]["img_path"]
        scene_img = refile.smart_load_image(scene_img_path)  # BGR
        scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
        # randm pick obj
        # ref_obj = random.randint(1, len(ref_obj_class))  #? 随机obj的个数好像不太好
        cycle_num = random.randint(1, CYCLE_NUMS)
        coor_3d_list = []
        coor_2d_list = []
        for i in range(cycle_num):
            if gen_image is not None:
                scene_img = copy.deepcopy(gen_image)
            cur_box = copy.deepcopy(BOX_TEMP)
            # 随机选取ref obj
            #! temp: 夜间数据只用水马来贴
            if json_date >= "190000":
                ref_obj_class = "water_horse"
            else:
                ref_obj_class = random.choice(ref_obj_classes)
            ref_path = random.choice(ref_data[ref_obj_class]["ref_path"])
            ref_lwh = ref_data[ref_obj_class]["lwh"]
            ref_img = refile.smart_load_image(ref_path)  # BGR
            # update class
            cur_box["class"] = ref_obj_class
            # 根据ref obj lwn，生成3d坐标
            center_x = random.uniform(RFU_CORE_BOX[0], RFU_CORE_BOX[1])
            # NOTE: 后续循环只能比上次更近
            y_upper_bound = RFU_CORE_BOX[3] if len(coor_3d_list) == 0 else coor_3d_list[-1][1]
            # y_lower_bound = RFU_CORE_BOX[2] if len(coor_3d_list) > 0 else 
            y_lower_bound = (CYCLE_NUMS - i) * 10  # 0:30, 1:20, 2:10
            # 由于防撞柱比较小，限制最远距离
            if ref_obj_class == "collision_bar":
                y_upper_bound = min(35, y_upper_bound)
            # center_y = random.uniform(RFU_CORE_BOX[2], RFU_CORE_BOX[3])
            center_y = random.uniform(y_lower_bound, y_upper_bound)
            center = np.array([center_x, center_y, ref_lwh[2]/2])
            # update influence
            # cur_box["influence"] = "avoid"
            cur_box["influence"] = get_label(center)
            coor_3d_list.append(center)
            # 默认朝向
            yaw = np.zeros(3)  # xyz euler
            yaw[2] = 0
            x1, y1, x2, y2 = get_3d_vertex(center, ref_lwh, yaw, trans_rfu2cam, K)
            # TODO: 根据后续框给前面的框添加遮挡属性
            # update rect
            cur_box["rects"]["xmin"] = x1
            cur_box["rects"]["ymin"] = y1
            cur_box["rects"]["xmax"] = x2
            cur_box["rects"]["ymax"] = y2
            coor_2d_list.append(np.array([x1, y1, x2, y2]))
            #! 需要check坐标的可行性

            # 根据3d坐标生成scene mask
            scene_mask = np.zeros((IMG_H, IMG_W, 3), np.uint8)
            scene_mask[y1:y2, x1:x2, :] = 255

            # reference image + reference mask
            # You could use the demo of SAM to extract RGB-A image with masks
            # https://segment-anything.com/demo
            h, w, c = ref_img.shape
            # mask = (ref_img[:,:,-1] > 128).astype(np.uint8)  # 由于ref_img本身就是黑底的单个物体
            if c == 4:
                mask = (ref_img[:,:,-1] > 128).astype(np.uint8)  # 由于ref_img本身就是黑底的单个物体
                ref_img = ref_img[:,:,:-1]
            else:
                mask = (np.sum(ref_img, axis=2) > 1).astype(np.uint8)
            ref_img = cv2.cvtColor(ref_img.copy(), cv2.COLOR_BGR2RGB)
            ref_image = ref_img 
            ref_mask = mask

            # background image
            # back_image = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
            back_image = scene_img

            # background mask 
            tar_mask = scene_mask[:,:,0] > 128
            tar_mask = tar_mask.astype(np.uint8)
            
            gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
            # cv2.imwrite(f"test/test_{i}_gen.png", gen_image[:, :, ::-1])

            # save img to check 
            h,w = back_image.shape[0], back_image.shape[0]
            ref_image = cv2.resize(ref_image, (w,h))
            back_display = copy.deepcopy(back_image)
            cv2.rectangle(back_display, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
            cv2.putText(back_display, f"({center})", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            vis_image = cv2.hconcat([ref_image, back_display, gen_image])
            cv2.imwrite(f"test/test_{nori_id}_{i}_all.png", vis_image[:,:,::-1])

            # fill label info 
            new_scene_data[nori_id]["labels"]["boxes"].append(cur_box)
                            
        save_path = refile.smart_path_join(PREFIX, TODAY, f"{nori_id}_fake.png")
        with refile.smart_open(save_path, 'wb') as file:
            file.write(jpeg.encode(gen_image[:,:,::-1]))

        
        # 4. fill label info for generated 
        new_scene_data[nori_id]["labels"]['boxes_label_info']["skipped"] = False
        new_scene_data[nori_id]["img_path"] = save_path

    with refile.smart_open(json_save_path, "w") as f:
        json.dump(new_scene_data, f, indent=2)
    

