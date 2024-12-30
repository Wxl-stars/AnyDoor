from .private.private_multimodal import PrivateMultiModalData
from .modules.annotation import E2EAnnotations
import random
import numpy as np
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
import cv2
import torch
import copy
from open3d import geometry
import open3d as o3d
import random
import warnings
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class PromptGenerator():
    def __init__(self,
                 empty_rate=0.5,
                 weather_rate=0.5,
                 time_rate=0.5,
                 place_rate=0.5,
                 category_rate=0.5,
                 ) -> None:
        self.empty_rate = empty_rate
        #暂不启用
        self.weather_rate = weather_rate
        self.time_rate = time_rate
        self.place_rate = place_rate
        self.category_rate = category_rate
        self.weather_map = {
                "晴天": "sunny",
                "阴天": "cloudy",
                "雨天": "rainy",
                "雪天": "snowy",
                "雾霾": "foggy",
                # "逆光": "backlight scene"
            }
        self.time_map = {
                "白天": "in the daytime",
                "夜晚": "at night",
                "黄昏": "at dusk"
            }
        self.place_map = {
                "高速公路或城市快速路": "on the highway or urban expressway",
                "隧道": "in the tunnel",
                "城区道路": "on the urban road",
                "高速收费站": "at the highway toll station",
                "匝道": "on the ramp",
                "路口": "at the intersection",
                # "施工路段": "construction zone"
            }
        
    def gen_category_prompt(self, labels):
        cat_map = {}
        for category in labels:
            if not category in cat_map:
                cat_map[category] = 0
            cat_map[category] += 1
        prompt_parts = []
        category_prompt_type = random.randint(0, 1)
        if category_prompt_type == 0:
            #句式1:数量与类型
            for category in cat_map:
                num = cat_map[category]
                category = category + "s" if num > 1 else category
                prompt_parts.append("{} {}".format(num, category))
        else:
            #句式2:只有类型
            for category in cat_map:
                prompt_parts.append(category + "s")
        prompt = " contains {},".format(",".join(prompt_parts)) if prompt_parts else ""
        return prompt
        
    def gen_weather_prompt(self, scene_tags):
        prompt = ""
        for tag in scene_tags:
            if tag in self.weather_map:
                prompt = " the weather is {},".format(self.weather_map[tag])
                break
        return prompt
    
    def gen_time_prompt(self, scene_tags):
        prompt = ""
        for tag in scene_tags:
            if tag in self.time_map:
                prompt = " the time is {},".format(self.time_map[tag])
                break
        return prompt
    
    def gen_place_prompt(self, scene_tags):
        prompt = ""
        for tag in scene_tags:
            if tag in self.place_map:
                prompt = " we are {},".format(self.place_map[tag])
        return prompt
    
    def gen_rand_prompt(self, labels, scene_tags):
        if random.random() < self.empty_rate:
            return ""

        prompt = "This describes driving scene constructed from six surrounding viewpoint images, "
        if labels is None or len(labels) < 1:
            category_prompt = ""
        
        category_prompt = self.gen_category_prompt(labels)
        if scene_tags is None or len(scene_tags) < 1:
            weather_prompt = ""
            place_prompt = ""
            time_prompt = ""
        else:
            random.shuffle(scene_tags) #可能同时有类似城区道路、路口类似的描述，采用随机的策略
            weather_prompt = self.gen_weather_prompt(scene_tags)
            place_prompt = self.gen_place_prompt(scene_tags)
            time_prompt = self.gen_time_prompt(scene_tags)
        prompt_parts = [category_prompt, weather_prompt, place_prompt, time_prompt]
        # desc_num = random.randint(1, 4)
        # use_parts = random.sample(prompt_parts, desc_num)
        random.shuffle(prompt_parts)
        for prompt_part in prompt_parts:
            prompt += prompt_part
        return prompt.strip(",") + "."
    

def get_rot(h):
    return np.array(
        [
            [np.cos(h), -np.sin(h), 0],
            [np.sin(h), np.cos(h), 0],
            [0, 0, 1],
        ]
    )

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (3840, 2160)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def get_interp_line(line, delta=1e-3):
    if all(line[:, -1] <= 0):
        return
    elif any(line[:, -1] <= 0):
        interpolate = (delta - line[line[:, -1] <= 0][0, -1]) / (
            line[line[:, -1] > 0][0, -1] - line[line[:, -1] <= 0][0, -1]
        )
        line[line[:, -1] <= 0] = (
            interpolate * (line[line[:, -1] > 0] - line[line[:, -1] <= 0]) + line[line[:, -1] <= 0]
        )
        return line


def draw_rect(img, selected_corners, color, linewidth):
    prev = selected_corners[-1]
    for corner in selected_corners:
        line = np.stack([prev, corner])
        line = get_interp_line(line)
        if line is None:
            continue
        cv2.line(img,
                    tuple(map(int, line[0, :2])),
                    tuple(map(int, line[1, :2])),
                    color, linewidth)
        prev = corner

class ControlPrivate(PrivateMultiModalData):
    view_colors = {
            'CAM_FRONT':[0, 130, 180],
            'CAM_FRONT_RIGHT':[220, 20, 60],
            'CAM_BACK_RIGHT':[255, 0, 0],
            'CAM_BACK':[0, 0, 142],
            'CAM_BACK_LEFT':[0, 60, 100],
            'CAM_FRONT_LEFT': [119, 11, 32]
    }
    max_classes = 10 # 可以留空位对齐
    colors = np.array([
                [255, 255, 255],
                [128, 64, 128], # car
                [244, 35, 232], # truck
                [70, 70, 70], # construction_vehicle
                [102, 102, 156], # bus
                [190, 153, 153], # motorcycle
                [0, 165, 255], # bicycle
                [250, 170, 30], # tricycle
                [144, 238, 144], # cyclist
                [42, 42, 165], # pedestrian
            ])

    def __init__(self, resemble_size=(1,1), final_dim=(1080, 1920), *args, **kwargs):
        if "annotation" in kwargs and isinstance(kwargs["annotation"], list):
            warnings.warn(
                "Task list annotations is deprecated for e2e dataset, should use E2EAnnotations instead.",
                DeprecationWarning,
            )
            super().__init__(**kwargs)
        else:
            annotations_e2e = kwargs.pop("annotation")
            kwargs["annotation"] = annotations_e2e["box"]
            super().__init__(**kwargs)
            self.annotation = E2EAnnotations(self.loader_output, self.mode, annotations_e2e)
        if isinstance(self.annotation, list):
            for task in self.annotation:
                task.loader_output["calibrated_sensors"] = self.image.loader_output["calibrated_sensors"]
        else:
            self.annotation.loader_output["calibrated_sensors"] = self.image.loader_output["calibrated_sensors"]

        self.resemble_size = resemble_size
        self.final_dim = final_dim
        r, c = self.resemble_size
        h, w = final_dim
        self.image_size = (r*h, c*w)
        self.prompt_generator = PromptGenerator(empty_rate=0) # 模型自带drop
        self.img_key_list = self.image.camera_names
        self.class_names = self.annotation.tasks["box"].class_names

    def generate_prompts(self, labels, scene_tags):
        labels = [self.class_names[lbl] for lbl in labels]
        prompt = self.prompt_generator.gen_rand_prompt(labels, scene_tags)
        return prompt

    def draw_bboxes(self, target, bboxes, labels, depths, colors, thickness=12):
        
        img = np.zeros((target.shape[0], target.shape[1], self.max_classes)) * 255 
        img = img.copy().astype(np.uint8)
        if labels is None or len(labels) == 0:
            return img

        for i, name in enumerate(self.class_names):
            mask = (labels == i)
            lab = labels[mask]
            dep = depths[mask]
            if bboxes is not None: bbox = bboxes[mask] 
            if bboxes is None or len(bbox) == 0:
                continue
            dep = dep * 3
            for j in range(len(bbox)):
                xmin,ymin,xmax,ymax = bbox[j]
                img[int(ymin) : int(ymax), int(xmin) : int(xmax), i] = np.where(img[int(ymin) : int(ymax), int(xmin) : int(xmax), i] > dep[j], dep[j], img[int(ymin) : int(ymax), int(xmin) : int(xmax), i])

        return img 
    
    def draw_dynamic_boxes3d(self, target, boxes, labels, colors, trans_lidar2cam, trans_cam2pix, linewidth=4):
        # corners [n, 8, 2]
        img = np.zeros((target.shape[0], target.shape[1], 3))
        img = img.copy().astype(np.uint8)
        if boxes is None or len(boxes) == 0:
            return img
        
        delta = 0.001
        for i in range(len(boxes)):
            color = colors[labels[i] + 1]
            color = (int(color[0]), int(color[1]), int(color[2]))
            center = boxes[i][0:3]
            dim = boxes[i][3:6]
            yaw = np.zeros(3)
            yaw[2] = boxes[i][6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
            line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
            line_set.transform(trans_lidar2cam)
            pts = np.asarray(line_set.points)  # [8,3]
            if all(pts[:, 2] <= 0):
                continue
            front_face = pts[[4,6,1,7]]   # open3d front face idx
            front_face, _ = cv2.projectPoints(front_face, np.zeros(3), np.zeros(3), trans_cam2pix, np.zeros(5))
            front_face = front_face.astype(np.int32)
            ori_color = (int(color[0]*0.5 + 255*0.5), int(color[1]*0.5 + 255*0.5), int(color[2]*0.5 + 255*0.5))
            cv2.fillPoly(img, [front_face], ori_color)

            lines_3d = []
            for x, y in np.array(line_set.lines):
                line = np.stack([pts[x], pts[y]])
                lines_3d.append(line)

            real_lines = []
            for line in lines_3d:
                if all(line[:, -1] > 0):
                    line, _ = cv2.projectPoints(line, np.zeros(3), np.zeros(3), trans_cam2pix, np.zeros(5))
                    real_lines.append(line[:, 0])
                elif any(line[:, -1] > 0):
                    interpolate = (delta - line[line[:, -1] <= 0][0, -1]) / (
                        line[line[:, -1] > 0][0, -1] - line[line[:, -1] <= 0][0, -1]
                    )
                    line[line[:, -1] <= 0] = (
                        interpolate * (line[line[:, -1] > 0] - line[line[:, -1] <= 0]) + line[line[:, -1] <= 0]
                    )
                    line, _ = cv2.projectPoints(line, np.zeros(3), np.zeros(3), trans_cam2pix, np.zeros(5))
                    real_lines.append(line[:, 0])
            max_val = 65536
            for line in real_lines:
                if abs(line).max() > max_val:
                    continue
                cv2.line(img, tuple(map(int, line[0])), tuple(map(int, line[1])), color, linewidth)
        return img

    def draw_corners(self, target, corners, labels, depths2d, colors, linewidth=4):
        img = np.zeros((target.shape[0], target.shape[1], 3)) * 255 
        img = img.copy().astype(np.uint8)

        if corners is None or len(corners) == 0:
            return img
        
        # print(corners.shape, labels.shape, depths2d.shape)
        sort_indexes = np.argsort(depths2d)[::-1]
        corners = corners[sort_indexes]
        labels = labels[sort_indexes]
        depths2d = depths2d[sort_indexes]
        delta = 0.001
        for j in range(len(corners)):
            color = colors[labels[j] + 1]
            color = (int(color[0]), int(color[1]), int(color[2]))

            # points = corners[j, [0, 1, 2, 3]]
            points =  np.array([[int(corners[j, 0, 0]), int(corners[j, 0, 1])], [int(corners[j, 1, 0]), int(corners[j, 1, 1])], [int(corners[j, 2, 0]), int(corners[j, 2, 1])], [int(corners[j, 3, 0]), int(corners[j, 3, 1])]])
            # points =  np.array([[int(corners[j, 1, 0]), int(corners[j, 1, 1])], [int(corners[j, 2, 0]), int(corners[j, 2, 1])], [int(corners[j, 6, 0]), int(corners[j, 6, 1])], [int(corners[j, 5, 0]), int(corners[j, 5, 1])]])
            points = points.reshape(-1, 1, 2)
            points[..., 0] = np.clip(points[..., 0], 0, target.shape[1]) 
            points[..., 1] = np.clip(points[..., 1], 0, target.shape[0]) 
            ori_color = (int(color[0]*0.5 + 255*0.5), int(color[1]*0.5 + 255*0.5), int(color[2]*0.5 + 255*0.5))
            cv2.fillPoly(img, [points], ori_color)

            for i in range(4):
                line = np.stack((corners[j][i], corners[j][i + 4]))
                line = get_interp_line(line)
                if line is None:
                    continue
                cv2.line(img, tuple(map(int, line[0, :2])), tuple(map(int, line[1, :2])), color[::-1], 2)
                # try:
                #     cv2.line(img,
                #         (int(corners[j][i][0]), int(corners[j][i][1])),
                #         (int(corners[j][i + 4][0]), int(corners[j][i + 4][1])),
                #         color[::-1], linewidth)
                # except:
                #     print(int(corners[j][i][0]), int(corners[j][i][1]), int(corners[j][i+4][0]), int(corners[j][i+4][1]))
                #     print(corners[j][i][0], corners[j][i][1], corners[j][i+4][0], corners[j][i+4][1])
                #     raise

            draw_rect(img, corners[j][:4], color[::-1], linewidth)
            draw_rect(img, corners[j][4:], color[::-1], linewidth)
        
        return img 

    def render_views(self, shapes, camera_views):
        img_list = []
        for i, view in enumerate(camera_views):
            img = np.zeros((shapes[0], shapes[1], 3))
            img = img.copy().astype(np.uint8)
            color = np.array(self.view_colors[view])
            img = img + color[None, None, :]
            img_list.append(img)
        return img_list

    def render_directions(self, shapes, img2egos):

        eps = 1e-5
        N = len(img2egos)
        H, W, _ = shapes
        coords_h = np.arange(H)
        coords_w = np.arange(W)
        # coords_d = np.array([1.0])
        coords_d = np.array([1.0, 2.0])

        D = coords_d.shape[0]
        coords = np.stack(np.meshgrid(coords_w, coords_h, coords_d)).transpose((1, 2, 3, 0)) # W, H, D, 3
        coords = np.concatenate((coords, np.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * np.maximum(coords[..., 2:3], np.ones_like(coords[..., 2:3])*eps)
        coords = coords.reshape(1, W, H, D, 4, 1)
        img2egos = img2egos.reshape(N, 1, 1, 1, 4, 4)
        # coords3d = np.matmul(img2lidar, coords).squeeze(-1).squeeze(-2)[..., :3]
        # coords3d = coords3d.transpose((0, 2, 1, 3))
        coords3d = np.matmul(img2egos, coords).squeeze(-1)[..., :3]
        coords3d = coords3d.transpose((0, 2, 1, 3, 4))

        directions = coords3d[:, :, :, 1, :] - coords3d[:, :, :, 0, :]
        coords3d = (directions - directions.min()) / (directions.max() - directions.min()) * 255
        coords3d = coords3d.copy().astype(np.uint8)
        coords3d = [cord3d for cord3d in coords3d]

        # directions = coords3d[:, :, :, 1, :] - coords3d[:, :, :, 0, :]
        # print(directions.min(), directions.max())
        # coords3d = sigmoid(directions) * 255
        # print(coords3d.min(), coords3d.max())
        # coords3d = coords3d.copy().astype(np.uint8)
        # coords3d = [cord3d for cord3d in coords3d]

        # coords3d = [direction for direction in directions]

        return coords3d
    
    def get_corners(self, gt_box) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        x, y, z, l, w, h, yaw = gt_box[:7]

        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        rot = get_rot(yaw)
        corners = np.dot(rot, corners)

        # Translate
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners.T
    
    
    def _get_2d_annos(self, shape, source_bbox_3d, source_label_3d, lidar2img):
        gt_bboxes_3d = source_bbox_3d
        gt_label_3d = source_label_3d
        corners_3d = np.stack([self.get_corners(gt_box) for gt_box in gt_bboxes_3d], 0)   # (n, 8, 3)
        # corners_3d = bbox3d_to_8corners(gt_bboxes_3d)
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)

        gt_bbox2d = []
        gt_depth2d = []
        gt_label2d = []
        gt_corners3d = []
        for i in range(len(self.img_key_list)):
            lidar2img_rt = np.array(lidar2img[i])
            pts_2d = pts_4d @ lidar2img_rt.T
            # pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=0.01, a_max=None)
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            
            H, W = shape[0], shape[1]
            imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
            imgfov_pts_depth = pts_2d[..., 2].reshape(num_bbox, 8)
            mask = (imgfov_pts_depth > 0.01).any(1)    # 数值稳定
            # mask = imgfov_pts_depth.mean(1) > 0.01

            if mask.sum() == 0:
                gt_bbox2d.append([])
                gt_depth2d.append([])
                gt_label2d.append([]) 
                gt_corners3d.append([])
                continue

            imgfov_pts_2d = imgfov_pts_2d[mask]
            imgfov_pts_depth = imgfov_pts_depth[mask]
            imgfov_pts_label= gt_label_3d[mask]

            bbox = []
            label = []
            depth = []
            corners3d = []
            imgfov_pts_3d = np.concatenate([imgfov_pts_2d, imgfov_pts_depth[..., None]], -1)
            for j, corner_coord in enumerate(imgfov_pts_2d):
                final_coords = post_process_coords(corner_coord, imsize = (W,H))
                if final_coords is None:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords
                    if ((max_x - min_x) >W-50) and ((max_y - min_y)>H-50):
                        continue
                    bbox.append([min_x, min_y, max_x, max_y])
                    label.append(imgfov_pts_label[j])
                    depth.append(imgfov_pts_depth[j].mean())
                    corners3d.append(copy.deepcopy(imgfov_pts_3d[j]))
            gt_bbox2d.append(np.array(bbox))
            gt_depth2d.append(np.array(depth))
            gt_label2d.append(np.array(label)) 
            gt_corners3d.append(np.array(corners3d)) 
        bbox2d_info = {
            'gt_bbox2d' : gt_bbox2d,
            'gt_depth2d' : gt_depth2d,
            'gt_label2d' : gt_label2d,
            'gt_corners3d': gt_corners3d
        }

        return bbox2d_info
    
    def _prepare_one_frame(self, item):
        source_img = item['imgs'] #torch.Size([6, 3, 256, 704]) 
        source_label_3d = item['gt_labels'] # torch.Size([32])  
        source_bbox_3d = item['gt_boxes'] # torch.Size([32, 9])
        # source_corner = item["corners"].numpy()
        lidar2imgs = item["lidar2imgs"] #torch.Size([6, 4, 4]) 
        ida_mats = item["ida_mats"] #torch.Size([6, 4, 4])
        lidar2cam = item["lidar2cam"] # [6, 4, 4]
        cam2img = item["cam2img"]  # [6, 3, 3]
        cam2pix = ida_mats[:, [0, 1, 3]][..., [0, 1, 3]] @ cam2img

        lidar2imgs = ida_mats @ lidar2imgs
        img2lidars = np.linalg.inv(lidar2imgs)

        # intrin = ida_mats @ intrin_mats
        # extrin = np.linalg.inv(cam2ego) @ lidar2ego[None] @ np.linalg.inv(bda_mat)[None]

        # img = draw_boxes_on_img(source_img[0].transpose(1,2,0), item["gt_bboxes_3d"], extrin[0], intrin[0])
        if len(source_bbox_3d) == 0:
            bbox2d_info = {
                'gt_bbox2d' : [[] for _ in range(len(self.img_key_list))],
                'gt_depth2d' : [[] for _ in range(len(self.img_key_list))],
                'gt_label2d' : [[] for _ in range(len(self.img_key_list))],
                'gt_corners3d': [[] for _ in range(len(self.img_key_list))],
            }
        else:
            bbox2d_info = self._get_2d_annos((source_img.shape[1], source_img.shape[2]), source_bbox_3d, source_label_3d, lidar2imgs)
        
        source_label_2d = bbox2d_info['gt_label2d']
        source_bbox_2d = bbox2d_info['gt_bbox2d']
        source_depth_2d = bbox2d_info['gt_depth2d']
        source_corner_2d = bbox2d_info['gt_corners3d']

        img_list = []
        dynamic_list = []
        map_list = []
        for view_id, view in enumerate(self.img_key_list):
            img = source_img[view_id]
            
            bboxes2d = source_bbox_2d[view_id]
            labels2d = source_label_2d[view_id]
            depths2d = source_depth_2d[view_id]
            corners2d = source_corner_2d[view_id]
            # source = self.draw_bboxes(img, bboxes2d, labels2d, depths2d, self.colors)
            # source_corner = self.draw_corners(img, corners2d, labels2d, depths2d, self.colors, linewidth=2) ###for 512
            dynamic = self.draw_dynamic_boxes3d(img, source_bbox_3d, source_label_3d, self.colors, lidar2cam[view_id], cam2pix[view_id], linewidth=2) ###for 512
            # source_corner = self.draw_corners(img, corners2d, labels2d, self.colors, linewidth=4) ###for 800
            render_map = np.zeros_like(dynamic)
            img_list.append(img)
            dynamic_list.append(dynamic)
            map_list.append(render_map)

        # render_list = self.render_directions(img.shape, img2lidars)

        target = np.stack(img_list, 0)
        dynamic = np.stack(dynamic_list, 0)
        render_map = np.stack(map_list, 0)
        # render_pe = np.stack(render_list, 0)

        # source = np.concatenate([source, render_pe], -1)
        source = np.concatenate([dynamic, render_map], -1)

        # source = np.concatenate([source, np.zeros([source.shape[0], source.shape[1], 13], dtype=np.float32)], -1)
        # filenames = item['img_metas']._data[0]['filename']
        return target, source

    def get_single_data(self, index):
        while True:
            frame_idx = self.loader_output["frame_index"][index]
            data_dict = {
                "frame_id": frame_idx,
            }
            # annotation
            if isinstance(self.annotation, list):
                for task in self.annotation:
                    annos = task[index]
                    if annos is None:
                        return None
            else:
                annos = self.annotation[index]
                if annos is None:
                    return None

            if annos is not None:  # self.is_train:
                data_dict.update(annos)

            # image
            img_info = self.image.get_images(frame_idx, data_dict)
            if img_info is None:
                index = self._rand_index()
                print("Image in this frame is not valid! Pick another one.")
                continue
            else:
                break

        # preocess image
        data_dict = self.pipeline(data_dict)
        data_dict["tags"] = self.loader_output["frame_data_list"][index]["tags"]
        data_dict["json_path"] = self.loader.get_scene_name(index)
        return data_dict

    def __getitem__(self, idx):
        data_dict = self.get_single_data(idx)
        target, source = self._prepare_one_frame(data_dict)
        target = torch.from_numpy(target).to(torch.float32)
        source = torch.from_numpy(source).to(torch.float32)
        target = target / 127.5 - 1.0
        source = source / 127.5 - 1.0
        target = target.permute(0,3,1,2)
        source = source.permute(0,3,1,2)

        prompt = self.generate_prompts(data_dict["gt_labels"], data_dict["tags"])

        ret = dict(
            video=target,
            text=prompt, 
            hint=source,
            height=target.shape[0],
            width=target.shape[1],
                   )
        return ret
    
if __name__ == "__main__":
    from datasets.private_data.data_cfg.HF_image_1v import base_dataset_cfg
    dataset = ControlPrivate(
        **base_dataset_cfg
    )
    data = dataset[256]
    from IPython import embed; embed()