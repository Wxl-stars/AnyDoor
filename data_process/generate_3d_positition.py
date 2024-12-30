import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录
parent_dir = os.path.dirname(current_dir)
# 将上层目录添加到 sys.path
sys.path.insert(0, parent_dir)
import cv2
import numpy as np

from datasets.private_data.utils.file_io import load_json
from pyquaternion import Quaternion
from open3d import geometry

RFU_CORE_BOX = [-2, 2, 150, -150]  # 左右前后

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

lidar2RFU = {
    "transform": {
        "translation": {
            "x": 0.0,
            "y": 0.0,
            "z": -0.33000001311302185
        },
        "rotation": {
            "w": -0.7071067811865474,
            "x": -0.0,
            "y": -0.0,
            "z": 0.7071067811865477
        }
    },
    "euler_degree": {
        "RotX": -0.0,
        "RotY": 0.0,
        "RotZ": -90.0
    },
    "calib_status": 0,
    "information": "ego_tf_rfu",
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

if __name__ == "__main__":
    # prepare ref obj info
    ref_obj_path = "/gpfs/public-shared/fileset-groups/wheeljack/wuxiaolei/projs/AnyDoor/ref_obj/ref_obj_info.json"
    ref_obj_info = load_json(ref_obj_path)
    # prepare transformation
    trans_rfu2cam = get_trans_rfu2cam(cam_front_120_extrinsic)
    K = get_camera_intrinsic(cam_front_120_intrinsic)
    # generate 3d 坐标 NOTE: 默认yaw = 0
    num = 1
    h = 1080
    w = 1920
    # x = np.linspace(-20, 20, num)
    x = [0]
    # y = np.linspace(4, 150, num)
    y = [20]
    print("x: ", x)
    print("y: ", y)
    for x,y in zip(x, y):
        center = np.array([x, y, 0])
        lwh = np.array([0.45, 0.45, 0.75])
        yaw = np.zeros(3)
        yaw[2] = 0
        x1, y1, x2, y2 = get_3d_vertex(center, lwh, yaw, trans_rfu2cam, K)
        print("!!! points_2d: ", x1, y1, x2, y2)
        box = np.concatenate((center, lwh, yaw))
        mask = np.zeros((h, w, 3), np.uint8)
        mask[y1:y2, x1:x2, :] = 255
        # cv2.putText(mask, f"({box[:6]})", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        cv2.imwrite("test_mask.png", mask)
        import IPython; IPython.embed()



    



