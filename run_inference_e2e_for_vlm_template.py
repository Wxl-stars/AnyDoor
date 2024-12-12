import copy
import os
import cv2
import einops
from loguru import logger
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
from datetime import date, datetime
from turbojpeg import TurboJPEG


"""
bash scripts/inference_e2e_for_vlm_template.sh 2 s3://sdagent-shard-bj-baiducloud/wuxiaolei/static/bev_range_-2_2_0_50/2024-11-27.json


for e2e
--scene_json s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/e2e_for_vlm_-2_2_0_50.json
# --ref_json s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/generated/ref_obj_info.json
--ref_json /gpfs/shared_files/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/ref_obj_info.json
--save_path s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/vlm_empty_scene_fill.json
"""

"""
--scene_json s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/e2e_for_vlm_-2_2_0_50.json
# --ref_json s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/generated/ref_obj_info.json
--ref_json /gpfs/shared_files/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/ref_obj_info.json
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
# MASK_X = 0.6
CYCLE_NUMS = 3
HALF_LANE = 1.875
BEV_RANGE = [-2, 2, 0, 50]
BEV_RESOLUTION = 0.02
EGO_px = int((0 - BEV_RANGE[0]) / BEV_RESOLUTION)
EGO_py = int((0 - BEV_RANGE[2]) / BEV_RESOLUTION)
BEV_width = int((BEV_RANGE[1] - BEV_RANGE[0]) / BEV_RESOLUTION)
BEV_height = int((BEV_RANGE[3] - BEV_RANGE[2]) / BEV_RESOLUTION)
# Y_THREH_MIN = 30
IMG_H = 1080
IMG_W = 1920
PREFIX = "s3://sdagent-shard-bj-baiducloud/wuxiaolei/vlm/anydoor_generated/template/"
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


#TODO: 修改一下拼个batch
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

def get_label_(coor_3d):
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

def get_label(coor_3d):
    label = None
    if coor_3d[0] > 1:
        label = "keep"
    else:
        label = "avoid"
    return label


def compute_boundary_intersection(origin, point, x_bounds, z_bounds):
    """
    计算从原点到目标点的延长线与地图边界的交点。
    :param origin: 原点坐标 (x0, z0)
    :param point: 目标点坐标 (x1, z1)
    :param x_bounds: X轴边界 [x_min, x_max]
    :param z_bounds: Z轴边界 [z_min, z_max]
    :return: 与地图边界的交点 (x', z')
    """
    x0, z0 = origin
    x1, z1 = point

    xmin, xmax, ymin, ymax = BEV_RANGE
    width = int((xmax - xmin) / BEV_RESOLUTION)
    height = int((ymax - ymin) / BEV_RESOLUTION)
    x_min = 0
    x_max = width
    z_min = 0
    z_max = height
    # x_min, x_max = x_bounds
    # z_min, z_max = z_bounds

    # 计算方向向量
    dx, dz = x1 - x0, z1 - z0
    intersections = []

    # 遍历边界 (x = x_min, x = x_max, z = z_min, z = z_max)
    if dx != 0:  # 避免除以零
        t_x_min = (x_min - x0) / dx
        t_x_max = (x_max - x0) / dx
        intersections.append((x_min, z0 + t_x_min * dz))  # 与左边界
        intersections.append((x_max, z0 + t_x_max * dz))  # 与右边界
    if dz != 0:  # 避免除以零
        t_z_min = (z_min - z0) / dz
        t_z_max = (z_max - z0) / dz
        intersections.append((x0 + t_z_min * dx, z_min))  # 与上边界
        intersections.append((x0 + t_z_max * dx, z_max))  # 与下边界

    # 筛选边界内的交点
    valid_points = [
        (int(x), int(z)) for x, z in intersections
        if x_min <= x <= x_max and z_min <= z <= z_max
    ]

    # 按照与原点的距离进行排序
    valid_points.sort(key=lambda v: v[0]**2 + v[1]**2, reverse=True)

    # 返回最近的交点
    if valid_points:
        return valid_points[0]  # 选择第一个有效点
    else:
        return None
    
def wether_add_bordar(corners):
    xmin, xmax, ymin, ymax = BEV_RANGE
    width = int((xmax - xmin) / BEV_RESOLUTION)
    height = int((ymax - ymin) / BEV_RESOLUTION)
    intersection = corners[:, 0]
    if (0 in intersection) and (width in intersection):
        corners = np.concatenate((corners, np.array([[0, height], [width, height]])))
    elif 0 in intersection:
        corners = np.concatenate((corners, np.array([[0, height]])))
    elif width in intersection:
        corners = np.concatenate((corners, np.array([[width, height]])))
    return corners
    
def anticlockwise_order_points(corners):
    # 计算中心点
    center = np.mean(corners, axis=0)

    # 计算每个点相对于中心的角度
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])

    # 根据角度排序点
    sorted_indices = np.argsort(angles)  # 默认从小到大排序 (逆时针)
    sorted_points = corners[sorted_indices]
    return sorted_points


def generate_bev_mask(bev_mask, bboxes, bev_range=BEV_RANGE, resolution=BEV_RESOLUTION, ratio=2):
    """
    Generate BEV mask from 3D bounding boxes.

    Args:
        bboxes (list): List of bounding boxes [(x, y, z, width, length, height, yaw), ...].
        bev_range (tuple): BEV range (xmin, xmax, ymin, ymax).
        resolution (float): Resolution in meters/pixel.

    Returns:
        np.ndarray: BEV mask as a binary image.
    """
    # Define BEV dimensions
    xmin, xmax, ymin, ymax = bev_range
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)    

    for bbox in bboxes:
        cur_mask = np.ones((height, width), dtype=np.uint8)
        x, y, _, w, l, _, yaw = bbox

        # 扩框
        w *= ratio
        l *= ratio
        
        # Calculate corners in world coordinates
        corners = np.array([
            [w / 2, l / 2],
            [-w / 2, l / 2],
            [-w / 2, -l / 2],
            [w / 2, -l / 2]
        ])
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T)
        translated_corners = rotated_corners + np.array([x, y])
        
        # Map to BEV pixel coordinates
        pixel_corners = ((translated_corners - [xmin, ymin]) / resolution).astype(np.int32)

        # Fill polygon on the mask
        cv2.fillPoly(cur_mask, [pixel_corners], 0)
        if np.all(cur_mask == 1):
            continue
        bev_mask = bev_mask * cur_mask
        # cv2.fillPoly(bev_mask, [pixel_corners], 0)

        # 计算前角交叉点
        pixel_corners = pixel_corners[pixel_corners[:, 1].argsort()]
        # pixel_corners = pixel_corners[np.lexsort((pixel_corners[:, 0], pixel_corners[:, 1]))]
        origin = (EGO_px, EGO_py)
        intersection_A = compute_boundary_intersection(origin, pixel_corners[0], bev_range[:2], bev_range[2:])
        intersection_B = compute_boundary_intersection(origin, pixel_corners[1], bev_range[:2], bev_range[2:])
        if intersection_A is not None and intersection_B is not None:
            intersection_A = np.array(intersection_A)[None, :]
            intersection_B = np.array(intersection_B)[None, :]
            corners = np.concatenate((intersection_A, intersection_B, pixel_corners[:2]), axis=0)
            intersections = corners[:2]
            # corners = corners[corners[:, 0].argsort()]
            corners = anticlockwise_order_points(corners)
            cv2.fillPoly(bev_mask, [corners], 0)
            corners = wether_add_bordar(intersections)
            corners = anticlockwise_order_points(corners)
            cv2.fillPoly(bev_mask, [corners], 0)
            # cv2.imwrite("test_2.png", bev_mask*255)

        # 计算后角交叉点
        intersection_A = compute_boundary_intersection(origin, pixel_corners[2], bev_range[:2], bev_range[2:])
        intersection_B = compute_boundary_intersection(origin, pixel_corners[3], bev_range[:2], bev_range[2:])
        if intersection_A is not None and intersection_B is not None:
            intersection_A = np.array(intersection_A)[None, :]
            intersection_B = np.array(intersection_B)[None, :]
            corners = np.concatenate((intersection_A, intersection_B, pixel_corners[2:]), axis=0)
            intersections = corners[:2]
            # corners = corners[corners[:, 0].argsort()]
            corners = anticlockwise_order_points(corners)
            cv2.fillPoly(bev_mask, [corners], 0)
            corners = wether_add_bordar(intersections)
            corners = anticlockwise_order_points(corners)
            cv2.fillPoly(bev_mask, [corners], 0)
            # cv2.imwrite("test_3.png", bev_mask*255)

    if np.all(bev_mask == 0):
        return bev_mask

    return bev_mask

def check_new_box_in_bev_mask(new_box, bev_mask, bev_range=BEV_RANGE, resolution=BEV_RESOLUTION):
    """
    Check if a new 3D box's corresponding BEV mask region contains any non-zero value.

    Args:
        new_box (tuple): A single bounding box (x, y, z, width, length, height, yaw).
        bev_mask (np.ndarray): Existing BEV mask.
        bev_range (tuple): BEV range (xmin, xmax, ymin, ymax).
        resolution (float): Resolution in meters/pixel.

    Returns:
        bool: True if the new box's BEV mask region is all zeros, False otherwise.
    """
    x, y, _, w, l, _, yaw = new_box
    xmin, xmax, ymin, ymax = bev_range

    # Calculate corners in world coordinates
    corners = np.array([
        [w / 2, l / 2],  # 右前角
        [-w / 2, l / 2],  # 左前角
        [-w / 2, -l / 2],  # 左后角
        [w / 2, -l / 2]   # 右后角
    ])
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    translated_corners = rotated_corners + np.array([x, y])

    # Map to BEV pixel coordinates
    pixel_corners = ((translated_corners - [xmin, ymin]) / resolution).astype(np.int32)

    # Create a mask for the new box
    new_box_mask = np.zeros_like(bev_mask, dtype=np.uint8)
    cv2.fillPoly(new_box_mask, [pixel_corners], 1)
    # 填与ego形成的锥形区域
    corners = np.array([EGO_px, EGO_py])
    corners = np.concatenate((corners[None, :],pixel_corners[:2]), axis=0)
    cv2.fillPoly(new_box_mask, [corners], 1)
    # cv2.imwrite("bev_mask.png", bev_mask*255)
    # cv2.imwrite("new_box_mask.png", new_box_mask*255)
    # total_mask = np.concatenate((bev_mask, new_box_mask), axis=1)
    # total_mask = bev_mask | new_box_mask

    # Check overlap between the new box mask and the BEV mask
    overlap = bev_mask * new_box_mask
    return np.all(overlap == 0), bev_mask, new_box_mask  # True if all values are 0

def add_occ_attr(boxes):
    for i in range(len(boxes)):
        cur_box = boxes[i]
        src_mask = cur_box["mask"]
        ratio = 0
        for j in range(i+1, len(boxes)):
            if i == j:
                continue
            tgt_mask = boxes[j]["mask"]
            ratio = max((src_mask & tgt_mask).sum() / src_mask.sum(), ratio)
        if ratio == 0:
            cur_box['occluded'] = "occluded_none"
        elif ratio < 0.3:
            cur_box['occluded'] = "occluded_mild"
        elif ratio >= 0.65:
            cur_box['occluded'] = "occluded_moderate"
        else:
            cur_box['occluded'] = "occluded_full"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scripts to generate data for vlm using AnyDoor"
    )
    parser.add_argument("--rank_id", type=int, required=False, default=0)
    parser.add_argument("--world_size", type=int, required=False, default=1)
    parser.add_argument("--scene_json", type=str, required=True, default=None)
    parser.add_argument("--ref_json", type=str, required=False, default="/gpfs/shared_files/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/ref_obj_info.json")
    parser.add_argument("--save_path", type=str, required=False, default=None)
    parser.add_argument("--save_local", action="store_true")
    args = parser.parse_args()
    return args

def load_data(data_json):
    json_paths = json.load(refile.smart_open(data_json))["paths"]
    data_list = []
    logger.info("[loading json data...]")
    for path in json_paths:
        json_data = json.load(refile.smart_open(path))
        calibrated_sensors = json_data["calibrated_sensors"]
        json_date = get_date_from_json(path)
        for frame in json_data["frames"]:
            data_info = dict()
            data_info["img_path"] = frame["img_path"]
            data_info["bev_mask"] = frame["bev_mask"]
            data_info["bboxes"] = [box["coor"] for box in frame["bbox"]]
            data_info["calib"] = calibrated_sensors
            data_info["json_date"]  = json_date
            data_info["nori_id"] = frame["nori_id"]
            data_list.append(data_info)
    return data_list

if __name__ == '__main__': 
    # 1. Preparasion 
    args = parse_args()
    if args.save_local:
        time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        local_dir = f"test/{time}_template"
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
            logger.info(f"generated imgs saved to: {local_dir}")

    if args.save_path is not None:
        json_last_name = args.save_path.split("-")[-1]
        json_last_name = json_last_name.replace(".json", "_fill_fake.json")
        json_save_path = refile.smart_path_join(PREFIX, TODAY, json_last_name)
    else:
        json_save_path = args.scene_json.replace(".json", f"_{TODAY}_fill_fake_template{args.rank_id}.json")

    
    img_dir = refile.smart_path_join(PREFIX, TODAY, "imgs")
    json_save_path = refile.smart_path_join(PREFIX, TODAY, f"_{TODAY}_fill_fake_random_{args.rank_id}.json")

    scene_data_list = load_data(args.scene_json)
    logger.info(f"Total {len(scene_data_list)} datas.")
    scene_data_list = scene_data_list[:10]
    ref_data = json.load(refile.smart_open(args.ref_json))

    new_scene_data = dict()
    jpeg = TurboJPEG()

    # 2. prepare(random create) input data
    ref_obj_classes = list(ref_data.keys())

    xmin, xmax, ymin, ymax = BEV_RANGE
    width = int((xmax - xmin) / BEV_RESOLUTION)
    height = int((ymax - ymin) / BEV_RESOLUTION)

    # multi-process
    if args.rank_id is not None and args.world_size is not None:
        step = args.world_size
        begin = args.rank_id
    else:
        step = 1
        begin = 0

    # 控制img resolution
    scale = 3

    for idx in tqdm(range(begin, len(scene_data_list)-1, step)):
        scene_data = scene_data_list[idx]
        nori_id = scene_data["nori_id"]
        gen_image = None
        new_scene_data[nori_id] = dict()
        new_scene_data[nori_id]["labels"] = {"boxes":[]}
        new_scene_data[nori_id]["calib"] = scene_data["calib"]
        # new_scene_data[nori_id]["labels"]['boxes_label_info']["skipped"] = False   #TODO
        json_date = scene_data["json_date"]
        calib_info = scene_data["calib"]
        # prepare scene img
        scene_img_path = scene_data["img_path"]
        scene_img = refile.smart_load_image(scene_img_path)  # BGR
        scene_img = cv2.resize(scene_img, (1920 * scale, 1080 * scale))
        scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
        # prepare transformation
        trans_rfu2cam = get_trans_rfu2cam(calib_info["extrinsic"])
        K = get_camera_intrinsic(calib_info["intrinsic"])
        K[:2] = K[:2] * scale
        # bev_mask = np.array(scene_data["bev_mask"])  # ! load 提前计算好的bev mask
        bboxes = scene_data['bboxes']
        
        # 生成动态障碍物的bev_mask
        bev_mask = np.ones((height, width), dtype=np.uint8)
        # mask掉例自车较近的一块区域
        y_lower_bound = RFU_CORE_BOX[2]
        end_y =  int((y_lower_bound - BEV_RANGE[2]) / BEV_RESOLUTION)
        bev_mask[0:end_y,  :] = 0

        # # mask掉自车挣钱方的一块区域
        # x_lower_bound = -1 * MASK_X
        # x_upper_bound = MASK_X
        # start_x = int((x_lower_bound - BEV_RANGE[0]) / BEV_RESOLUTION)
        # end_x = int((x_upper_bound - BEV_RANGE[0]) / BEV_RESOLUTION)
        # bev_mask[:, start_x:end_x] = 0

        # 根据动态目标的3d框生成mask
        bev_mask= generate_bev_mask(bev_mask, bboxes, BEV_RANGE, BEV_RESOLUTION, ratio=1.1)
        empty_ratio = bev_mask.mean()

        ref_obj_class = random.choice(["cone", "collision_bar", "anti_collision_barrel"])
        ref_path = random.choice(ref_data[ref_obj_class]["ref_path"])
        ref_img = refile.smart_load_image(ref_path)  # BGR
        ref_img = cv2.cvtColor(ref_img.copy(), cv2.COLOR_BGR2RGB)
        ref_lwh = ref_data[ref_obj_class]["lwh"]
        # 竖排：需要col，delta，起始y [由间隔算数量]
        y_lower_bound = int((7 - BEV_RANGE[2]) / BEV_RESOLUTION)
        y_upper_bound = int((10 - BEV_RANGE[2]) / BEV_RESOLUTION)
        bev_mask[:y_lower_bound, :] = 0

        # 贴特殊成排成列的锥桶 TODO: 需要改一下，
        #! 添加横排处理流程
        # col_indices = [j for j in range(bev_mask.shape[1]) if bev_mask[y_lower_bound:, j].mean() > 0.3]
        col_indices = [j for j in range(bev_mask.shape[1]) if np.all(bev_mask[end_y:, j] == 1)]  # 竖排全为1
        # row_indices = [i for i in range(bev_mask.shape[0]) if np.all(bev_mask[i, :] == 1)]  # 横排全为1
        try:
            col = random.choice(col_indices)
        except:
            continue
        start_row = random.randint(y_lower_bound, y_upper_bound)   #! random2
        y_lower_bound = int((25 - BEV_RANGE[2]) / BEV_RESOLUTION)
        y_upper_bound = int((min(ref_data[ref_obj_class]["upper_bound_y"], 35) - BEV_RANGE[2]) / BEV_RESOLUTION)
        end_row = random.randint(y_lower_bound, y_upper_bound)  #! random3
        # 确定间隔
        y = BEV_RANGE[0] + col * BEV_RESOLUTION
        # TODO: 是否需要这个逻辑？这个逻辑 or mask一部分的x？
        if np.abs(y) < 0.3:
            delta = random.uniform(1, 5)  #! random1
        else:
            delta = random.uniform(3, 5)
        delta = delta * ref_data[ref_obj_class]["delta_weight_y"]
        bev_delta = int(delta / BEV_RESOLUTION)
        i = 0
        for row in range(end_row, start_row, -bev_delta):
            i += 1
            if gen_image is not None:
                scene_img = copy.deepcopy(gen_image)

            coors = np.array(
                [BEV_RANGE[0] + col * BEV_RESOLUTION, BEV_RANGE[2] + row * BEV_RESOLUTION]  # 将像素坐标转换为世界坐标
            )
            cur_box = copy.deepcopy(BOX_TEMP)
            cur_box["class"] = ref_obj_class
            center_x, center_y = coors

            # update influence
            center = np.array([center_x, center_y, ref_lwh[2]/2])
            cur_box["influence"] = get_label(center)
            # coor_3d_list.append(center)
            # 默认朝向
            yaw = np.zeros(3)  # xyz euler
            yaw[2] = 0
            x1, y1, x2, y2 = get_3d_vertex(center, ref_lwh, yaw, trans_rfu2cam, K)
            # 如果所占的pixel < 100，认为不合理
            pixel_num = (x2 - x1) * (y2 - y1)
            if pixel_num < 100:
                continue
            cur_box["3d_box"] = center.tolist() + ref_lwh + [yaw[2]]  # xyz lwh yaw
            # # 每贴一个目标就要更新bev mask的遮挡区域 NOTE: template不需要
            # bev_mask = generate_bev_mask(bev_mask, [cur_box["3d_box"]])
            # TODO: 根据后续框给前面的框添加遮挡属性
            # update rect
            cur_box["rects"]["xmin"] = x1
            cur_box["rects"]["ymin"] = y1
            cur_box["rects"]["xmax"] = x2
            cur_box["rects"]["ymax"] = y2
            # coor_2d_list.append(np.array([x1, y1, x2, y2]))
            #! 需要check坐标的可行性

            # 根据3d坐标生成scene mask
            scene_mask = np.zeros((IMG_H * scale, IMG_W * scale, 3), np.uint8)
            scene_mask[y1:y2, x1:x2, :] = 255
            cur_box["mask"] = (scene_mask[:, :, 0] / 255).astype(int)

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
            # ref_img = cv2.cvtColor(ref_img.copy(), cv2.COLOR_BGR2RGB)
            ref_image = ref_img 
            ref_mask = mask

            # background image
            # back_image = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
            back_image = scene_img

            # background mask 
            tar_mask = scene_mask[:,:,0] > 128
            tar_mask = tar_mask.astype(np.uint8)

            # diffusion model
            gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)

            # fill label info 
            new_scene_data[nori_id]["labels"]["boxes"].append(cur_box)
                    
        # 添加遮挡属性
        if len(new_scene_data[nori_id]["labels"]["boxes"]) > 1:
            add_occ_attr(new_scene_data[nori_id]["labels"]["boxes"])
        
        for box in new_scene_data[nori_id]["labels"]["boxes"]:
            box.pop("mask")
                            
        # 结果存入s3 
        save_path = refile.smart_path_join(img_dir, f"{nori_id}_fake.png")
        save_img = cv2.resize(gen_image, (IMG_W * 2, IMG_H * 2))
        with refile.smart_open(save_path, 'wb') as file:
            file.write(jpeg.encode(save_img[:,:,::-1]))

        if args.save_local and random.choice([True, False, False, False, False]):
            display = gen_image.copy()
            for cur_box in new_scene_data[nori_id]["labels"]["boxes"]:
                x1 = cur_box["rects"]["xmin"]
                y1 = cur_box["rects"]["ymin"]
                x2 = cur_box["rects"]["xmax"]
                y2 = cur_box["rects"]["ymax"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                center = cur_box["3d_box"][:2]
                center = [round(x, 2) for x in center]
                label = cur_box["class"]
                influence = cur_box["influence"]
                occ = cur_box["occluded"].split("-")[-1]
                cv2.putText(display, f"({center}-{label}-{influence}-{occ})", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            display = cv2.resize(display, (IMG_W * 2, IMG_H * 2))
            save_path = refile.smart_path_join(local_dir, f"{nori_id}.png")
            logger.info(f"save as {save_path}")
            cv2.imwrite(save_path, display[:, :, ::-1])
            cv2.imwrite(save_path.replace(".png", "_mask.png"), bev_mask * 255)

        
        # 4. fill label info for generated 
        # new_scene_data[nori_id]["labels"]['boxes_label_info']["skipped"] = False
        new_scene_data[nori_id]["img_path"] = save_path

    with refile.smart_open(json_save_path, "w") as f:
        json.dump(new_scene_data, f, indent=2)

    logger.info(f"success saving [json data] to : {json_save_path}")
    logger.info(f"success saving [images] to : {img_dir}")
    

