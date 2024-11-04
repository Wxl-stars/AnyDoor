import bisect
import cv2
import numpy as np
from copy import deepcopy
from pyquaternion import Quaternion
from .base import AnnotationBase


class AnnotationADMap(AnnotationBase):
    def __init__(
        self,
        loader_output,
        mode,
        label_key="bev",  # 在 json 中对应的字段
    ) -> None:
        super(AnnotationADMap, self).__init__(loader_output, mode)
        self.label_key = label_key

    def __getitem__(self, idx):
        frame_data_list = self.loader_output["frame_data_list"]
        if self.label_key not in frame_data_list[idx]:
            return None

        cumulative_sizes = self.loader_output["frame_data_list"].cumulative_sizes
        scene_id = bisect.bisect_right(cumulative_sizes, idx)

        bev_info = frame_data_list[idx][self.label_key]
        calibrated_sensors = self.loader_output["calibrated_sensors"][scene_id]
        timestamp = frame_data_list[idx].get("timestamp", -1)
        # fpath = frame_data_list[idx]['fpath']

        bev_processd = self.process_frame(
            bev_info,
            calibrated_sensors,
            cutoff_distance=100.0,
            min_num=3,
            x_max=100,
            y_max=15,
            fov_filter_cam="cam_front_120",
        )
        lane_processd = bev_processd["lane"]
        lane_seq, lane_seq_topo = self.precess_lane_topo(lane_processd)
        lane_seq_sample = self.sequence_sample(lane_seq, sample_distance=5.0)  # 间隔 5 m 取点
        sequence_data, valid_mask_seq, sequence_data_noise, fork_idx, end_idx = self.prepare_data_discrete_3D(
            lane_seq_sample,
            MAX_NUM_SEQ=10,
            MAX_SEQ_LEN=180,
            range_quantize=500,
            min_range_y=-15,
            max_range_y=15,
            min_range_x=0,
            max_range_x=100,
            min_range_z=-10,
            max_range_z=10,
        )

        # aux rv seg supervision, 注意这里出来的是原图大小，没有 img aug 对齐的 mask
        img_semantic_seg, img_ins_seg = self.prepare_img_segment_map(
            lane_seq_sample, calibrated_sensors, cam_names=["cam_front_120"]
        )
        # 变小可节约内存
        # img_semantic_seg = [cv2.resize(img_semantic_seg[i], (56, 32), interpolation=cv2.INTER_NEAREST) for i in range(len(img_semantic_seg))]
        sequence_pe = np.zeros(1)
        item = dict(
            sequence_pe=sequence_pe,
            sequence_data=sequence_data,
            sequence_data_noise=sequence_data_noise,
            valid_mask_seq=valid_mask_seq,
            fork_idx=fork_idx,
            end_idx=end_idx,
            lane_seg_rv=img_semantic_seg,
            timestamp=timestamp,
        )
        return item

    def process_frame(
        self,
        bev_info,
        calibrated_sensors,
        cutoff_distance=100.0,
        min_num=3,
        x_max=100,
        y_max=15,
        fov_filter_cam="cam_front_120",
    ):
        bev_info = deepcopy(bev_info)

        # 去重复点
        all_lane_id = list(bev_info["lane"].keys())
        for _lane_id in all_lane_id:
            points = bev_info["lane"][_lane_id]["points"]
            if len(points) < 2:
                continue
            for i in range(len(points) - 1, 0, -1):
                if points[i] == points[i - 1]:
                    del bev_info["lane"][_lane_id]["points"][i]

        # 删除100m外的点
        all_lane_id = list(bev_info["lane"].keys())
        for _lane_id in all_lane_id:
            points = bev_info["lane"][_lane_id]["points"]
            if len(points) < 2:
                continue
            for i in range(len(points)):
                if np.linalg.norm(np.array(points[i])) >= cutoff_distance:
                    bev_info["lane"][_lane_id]["points"] = bev_info["lane"][_lane_id]["points"][:i]
                    break

        # 删除 [x_max,y_max] 外的点
        all_lane_id = list(bev_info["lane"].keys())
        for _lane_id in all_lane_id:
            points = bev_info["lane"][_lane_id]["points"]
            if len(points) < 2:
                continue
            for i in range(len(points)):
                if abs(points[i][0]) >= x_max or abs(points[i][1]) >= y_max:
                    bev_info["lane"][_lane_id]["points"] = bev_info["lane"][_lane_id]["points"][:i]
                    break

        # 删除少于3个点的线
        all_lane_id = list(bev_info["lane"].keys())
        for _lane_id in all_lane_id:
            points = bev_info["lane"][_lane_id]["points"]
            if len(points) < min_num:
                del bev_info["lane"][_lane_id]

        # 将每条线下采样50倍，方便后面计算
        all_lane_id = list(bev_info["lane"].keys())
        for _lane_id in all_lane_id:
            points = bev_info["lane"][_lane_id]["points"]
            new_points = []
            for i in range(len(points) - 1):
                _s = np.arange(0.02, 1, 0.02)
                pts = (1 - _s[:, None]) * np.array(points[i]) + _s[:, None] * np.array(points[i + 1])
                pts = pts.tolist()
                new_points.append(points[i])
                new_points += pts
            new_points.append(points[-1])
            bev_info["lane"][_lane_id]["points"] = new_points

        # 删除FOV外的点
        W, H = calibrated_sensors[fov_filter_cam]["intrinsic"]["resolution"]
        K = calibrated_sensors[fov_filter_cam]["intrinsic"]["K"]
        ext_R = calibrated_sensors[fov_filter_cam]["extrinsic"]["transform"]["rotation"]
        ext_t = calibrated_sensors[fov_filter_cam]["extrinsic"]["transform"]["translation"]
        lidar2cam = self.transform_matrix(
            rotation=Quaternion([ext_R["w"], ext_R["x"], ext_R["y"], ext_R["z"]]),
            translation=[ext_t["x"], ext_t["y"], ext_t["z"]],
            inverse=False,
        )
        lidar_ego_ext_R = calibrated_sensors["lidar_ego"]["extrinsic"]["transform"]["rotation"]
        lidar_ego_ext_t = calibrated_sensors["lidar_ego"]["extrinsic"]["transform"]["translation"]
        ego2lidar = self.transform_matrix(
            rotation=Quaternion(
                [lidar_ego_ext_R["w"], lidar_ego_ext_R["x"], lidar_ego_ext_R["y"], lidar_ego_ext_R["z"]]
            ),
            translation=[lidar_ego_ext_t["x"], lidar_ego_ext_t["y"], lidar_ego_ext_t["z"]],
            inverse=True,
        )
        ego2cam = lidar2cam @ ego2lidar
        all_lane_id = list(bev_info["lane"].keys())

        for _lane_id in all_lane_id:
            points = bev_info["lane"][_lane_id]["points"]
            for i in range(len(points) - 1, -1, -1):
                p0_ego = np.array(points[i])
                p0_cam = ego2cam[:3, :3] @ p0_ego + ego2cam[:3, 3]
                p0_uv = K @ (p0_cam / p0_cam[2])
                if p0_cam[2] > 1e-2 and 0 <= p0_uv[0] <= W and 0 <= p0_uv[1] <= H:
                    continue
                else:
                    del bev_info["lane"][_lane_id]["points"][i]

        # 删除少于40个点的线
        all_lane_id = list(bev_info["lane"].keys())
        for _lane_id in all_lane_id:
            points = bev_info["lane"][_lane_id]["points"]
            if len(points) < 40:
                del bev_info["lane"][_lane_id]

        return bev_info

    @staticmethod
    def transform_matrix(
        translation: np.ndarray = np.array([0, 0, 0]),
        rotation: Quaternion = Quaternion([1, 0, 0, 0]),
        inverse: bool = False,
    ) -> np.ndarray:
        tm = np.eye(4)
        if inverse:
            rot_inv = rotation.rotation_matrix.T
            trans = np.transpose(-np.array(translation))
            tm[:3, :3] = rot_inv
            tm[:3, 3] = rot_inv.dot(trans)
        else:
            tm[:3, :3] = rotation.rotation_matrix
            tm[:3, 3] = np.transpose(np.array(translation))
        return tm

    def precess_lane_topo(self, lane_processd):
        lane_processd = deepcopy(lane_processd)
        all_lane_id = list(lane_processd.keys())
        point_dict = {}
        for _lane_id in all_lane_id:
            points = lane_processd[_lane_id]["points"]
            if len(points) < 2:
                print("ERROR SIZE")
            if points[0][0] >= points[-1][0]:
                print("ERROR order")  # 路口处顺序不一定能保证
                # exit(0)
            for _p in points:
                if tuple(_p) not in point_dict:
                    point_dict[tuple(_p)] = [_lane_id]
                else:
                    if _lane_id in point_dict[tuple(_p)]:  # 有重复点
                        print("ERROR points", _lane_id)
                    else:
                        point_dict[tuple(_p)].append(_lane_id)
        PRE, POST, EQL_begin, EQL_end = {}, {}, {}, {}
        for _lane_id in all_lane_id:
            PRE[_lane_id] = set()
            POST[_lane_id] = set()
            EQL_begin[_lane_id] = set()
            EQL_end[_lane_id] = set()
        for _lane_id in all_lane_id:
            points_start = lane_processd[_lane_id]["points"][0]
            for tmp_lane_id in point_dict[tuple(points_start)]:
                if tmp_lane_id == _lane_id:
                    continue
                if points_start == lane_processd[tmp_lane_id]["points"][-1]:
                    PRE[_lane_id].add(tmp_lane_id)
                    POST[tmp_lane_id].add(_lane_id)
                if points_start == lane_processd[tmp_lane_id]["points"][0]:
                    EQL_begin[_lane_id].add(tmp_lane_id)
                    EQL_begin[tmp_lane_id].add(_lane_id)
            points_end = lane_processd[_lane_id]["points"][-1]
            for tmp_lane_id in point_dict[tuple(points_end)]:
                if tmp_lane_id == _lane_id:
                    continue
                if points_end == lane_processd[tmp_lane_id]["points"][0]:
                    POST[_lane_id].add(tmp_lane_id)
                    PRE[tmp_lane_id].add(_lane_id)
                if points_end == lane_processd[tmp_lane_id]["points"][-1]:
                    EQL_end[_lane_id].add(tmp_lane_id)
                    EQL_end[tmp_lane_id].add(_lane_id)

        # 处理V型线
        new_id = "999"
        for _lane_id in all_lane_id:
            if len(PRE[_lane_id]) == 0 and len(EQL_begin[_lane_id]) != 0:
                # lane_processd[new_id] = {'points':[lane_processd[_lane_id]["points"][0]]}

                p_fake = deepcopy(lane_processd[_lane_id]["points"][0])
                p_fake[0] -= 2
                lane_processd[new_id] = {"points": [p_fake, lane_processd[_lane_id]["points"][0]]}

                PRE[_lane_id].add(new_id)
                PRE[new_id] = set()
                POST[new_id] = set([_lane_id])
                EQL_begin[new_id] = set()
                EQL_end[new_id] = set()
                for _id_ in EQL_begin[_lane_id]:
                    PRE[_id_].add(new_id)
                    POST[new_id].add(_id_)
                new_id = str(int(new_id) + 1)

        # 处理倒V型线
        new_id = "9999"
        for _lane_id in all_lane_id:
            if len(POST[_lane_id]) == 0 and len(EQL_end[_lane_id]) != 0:
                lane_processd[new_id] = {
                    "points": [lane_processd[_lane_id]["points"][-1], lane_processd[_lane_id]["points"][-1]]
                }
                POST[_lane_id].add(new_id)
                POST[new_id] = set()
                PRE[new_id] = set([_lane_id])
                EQL_begin[new_id] = set()
                EQL_end[new_id] = set()
                for _id_ in EQL_end[_lane_id]:
                    POST[_id_].add(new_id)
                    PRE[new_id].add(_id_)
                new_id = str(int(new_id) + 1)

        all_lane_id = list(lane_processd.keys())
        # generate sequence
        SEQUENCE = []
        for _lane_id in all_lane_id:
            if len(PRE[_lane_id]) == 0:
                SEQUENCE.append(self.generate_seq(lane_processd, PRE, POST, EQL_begin, EQL_end, _lane_id))
        FULL_SEQUENCE = []
        for _seq in SEQUENCE:
            point_sequence = []
            for idx in _seq:
                if idx == "END" or idx == "FORK" or idx == "MERGE":
                    point_sequence.append(idx)
                else:
                    ind_, is_forward = idx
                    points = lane_processd[ind_]["points"][::is_forward]
                    if points[0] in point_sequence:
                        point_sequence += points[1:]  # 删除前后重复的点
                    else:
                        point_sequence += points
            FULL_SEQUENCE.append(point_sequence)
        FULL_SEQUENCE_REDUCED = []
        for _seq in FULL_SEQUENCE:
            data_sequence = []
            idx = 0
            while idx < len(_seq):
                if _seq[idx] == "END" or _seq[idx] == "FORK" or _seq[idx] == "MERGE":
                    print("ERROR seq")
                    exit(0)
                if _seq[idx + 1] == "END":
                    data_sequence.append((_seq[idx], "END"))
                    idx += 2
                elif _seq[idx + 1] == "FORK":
                    data_sequence.append((_seq[idx], "FORK"))
                    idx += 2
                elif _seq[idx + 1] == "MERGE":
                    data_sequence.append((_seq[idx], "MERGE"))
                    idx += 2
                else:
                    data_sequence.append((_seq[idx], "CONTINUE"))
                    idx += 1
            FULL_SEQUENCE_REDUCED.append(data_sequence)

        return FULL_SEQUENCE_REDUCED, SEQUENCE

    def generate_seq(self, lane_processd, PRE, POST, EQL_begin, EQL_end, _lane_id):
        result = []
        result.append((_lane_id, 1))
        if len(EQL_end[_lane_id]) != 0 and len(POST[_lane_id]) <= 1:  # 包含了 X 型处理 和 O 型线
            result = result + ["MERGE"]
        if len(POST[_lane_id]) == 0:
            result += ["END"]
            return result
        elif len(POST[_lane_id]) == 1:
            post_id = list(POST[_lane_id])[0]
            result += self.generate_seq(lane_processd, PRE, POST, EQL_begin, EQL_end, post_id)
            return result
        elif len(POST[_lane_id]) > 1:  # 没有考虑1分3的情况
            post_id1 = list(POST[_lane_id])[0]
            post_id2 = list(POST[_lane_id])[1]
            points1_y = lane_processd[post_id1]["points"][1][1]
            points2_y = lane_processd[post_id2]["points"][1][1]
            if points1_y > points2_y:
                left = self.generate_seq(lane_processd, PRE, POST, EQL_begin, EQL_end, post_id1)
                right = self.generate_seq(lane_processd, PRE, POST, EQL_begin, EQL_end, post_id2)
            else:
                left = self.generate_seq(lane_processd, PRE, POST, EQL_begin, EQL_end, post_id2)
                right = self.generate_seq(lane_processd, PRE, POST, EQL_begin, EQL_end, post_id1)
            result = result + ["FORK"] + left + right
            return result

    def sequence_sample(self, sequence, sample_distance=5.0):
        SEQ = []
        for seq in sequence:
            post, pre = self.parse_seq(seq)
            sequence_new = []
            # start_ind = np.random.randint(1,39)
            start_ind = 0
            sequence_new.append(seq[start_ind])
            last_point = sequence_new[-1][0]
            for ind in range(start_ind + 1, len(seq)):
                distance = np.linalg.norm(np.array(last_point) - np.array(seq[ind][0]))
                if seq[ind][1] == "CONTINUE" and distance >= sample_distance:
                    sequence_new.append(seq[ind])
                    last_point = sequence_new[-1][0]
                elif seq[ind][1] == "END":
                    sequence_new.append(seq[ind])
                    if ind + 1 in pre:
                        last_point = seq[pre[ind + 1]][0]
                    else:
                        assert ind + 1 == len(seq)
                        last_point = sequence_new[-1][0]
                elif seq[ind][1] == "FORK" or seq[ind][1] == "MERGE":
                    sequence_new.append(seq[ind])
                    last_point = sequence_new[-1][0]
            SEQ.append(sequence_new)
        return SEQ

    @staticmethod
    def parse_seq(seq):
        post = {}
        _forkpoint_stack = []
        for ind, p in enumerate(seq):
            if p[1] == "CONTINUE" or p[1] == "MERGE":
                post[ind] = [ind + 1]
            if p[1] == "END":
                post[ind] = [-1]
                if len(_forkpoint_stack) == 0:
                    if ind == len(seq) - 1:
                        break
                    else:
                        print("PARSE ERROR")
                        exit()
                while len(_forkpoint_stack) != 0:
                    _forkpoint_stack[-1][1] -= 1
                    if _forkpoint_stack[-1][1] == 0:
                        _forkpoint_stack.pop(-1)
                        continue
                    else:
                        post[_forkpoint_stack[-1][0]].append(ind + 1)
                        break
            if p[1] == "FORK":
                _forkpoint_stack.append([ind, 2])
                post[ind] = [ind + 1]
        if len(_forkpoint_stack) != 0 or seq[-1][1] != "END":
            print("PARSE ERROR")
            exit()

        pre = {}
        for k, v in post.items():
            for next_i in v:
                if next_i == -1:
                    continue
                pre[next_i] = k
        return post, pre

    def prepare_img_segment_map(self, sequence, calibrated_sensors, cam_names=["cam_front_120"]):
        input_shape = [calibrated_sensors[cam]["intrinsic"]["resolution"][::-1] for cam in cam_names]  # [[2160, 3840]]
        img_semantic_map, img_ins_map = [], []
        for cam_id in range(len(input_shape)):
            # get footprint2img
            cam = cam_names[cam_id]
            K = calibrated_sensors[cam]["intrinsic"]["K"]
            ext_R = calibrated_sensors[cam]["extrinsic"]["transform"]["rotation"]
            ext_t = calibrated_sensors[cam]["extrinsic"]["transform"]["translation"]
            lidar2cam = self.transform_matrix(
                rotation=Quaternion([ext_R["w"], ext_R["x"], ext_R["y"], ext_R["z"]]),
                translation=[ext_t["x"], ext_t["y"], ext_t["z"]],
                inverse=False,
            )
            lidar_ego_ext_R = calibrated_sensors["lidar_ego"]["extrinsic"]["transform"]["rotation"]
            lidar_ego_ext_t = calibrated_sensors["lidar_ego"]["extrinsic"]["transform"]["translation"]
            ego2lidar = self.transform_matrix(
                rotation=Quaternion(
                    [lidar_ego_ext_R["w"], lidar_ego_ext_R["x"], lidar_ego_ext_R["y"], lidar_ego_ext_R["z"]]
                ),
                translation=[lidar_ego_ext_t["x"], lidar_ego_ext_t["y"], lidar_ego_ext_t["z"]],
                inverse=True,
            )
            ego2cam = lidar2cam @ ego2lidar
            K_44 = np.eye(4).astype(np.float32)
            K_44[:3, :3] = K
            ego2img = K_44 @ ego2cam

            img_h, img_w = input_shape[cam_id]  # [2160, 3840]
            ins_num = len(sequence)
            seg = np.zeros((ins_num, img_h, img_w), dtype=np.uint8)
            for ins_id, seq in enumerate(sequence):
                pt_seq = self.seq_to_pt_seq(seq)
                pt_seq = np.array(pt_seq)  # list 转 array, [N, 3]  xyz
                pt_seq = np.concatenate([pt_seq, np.ones([pt_seq.shape[0], 1])], axis=1)  # [N, 4]
                uv_pts = ego2img @ pt_seq.T  # [4, N]
                uv_pts = uv_pts.T
                z_mask = uv_pts[:, 2] > 0

                uv_pts = uv_pts[z_mask]  # 有一定风险出错  [N, 4]
                uv_pts = uv_pts[:, :3] / uv_pts[:, 2:3]
                uv_pts = uv_pts[:, :2].astype(int)  # 往下取
                seg[ins_id] = cv2.polylines(
                    seg[ins_id], [uv_pts], isClosed=False, thickness=48, color=1
                )  # 在 [512, 896] 上是 线宽 48, 在 32*56 上应当是 4

            semantic_seg = (np.sum(seg, axis=0) > 0).astype(np.uint8)
            img_semantic_map.append(semantic_seg)
            img_ins_map.append(seg)
        return img_semantic_map, img_ins_map

    @staticmethod
    def seq_to_pt_seq(seq):
        pts_seq = []
        fork = None  # 只考虑 二分叉
        for pt in seq:
            pts_seq.append(pt[0])
            if fork is None:  # 前面没有 fork
                if pt[1] == "FORK":  # 当前点是 FORK
                    fork = pt[0]
            else:  # 前面有 FORK
                if pt[1] == "END":  # 当前点是 END, 就复制一个前面的 FORK
                    pts_seq.append(fork)
        return pts_seq

    @staticmethod
    def prepare_data_discrete_3D(
        sequence,
        MAX_NUM_SEQ=10,
        MAX_SEQ_LEN=180,
        range_quantize=500,
        min_range_y=-15,
        max_range_y=15,
        min_range_x=0,
        max_range_x=100,
        min_range_z=-10,
        max_range_z=10,
    ):
        # MAX_NUM_SEQ = 10
        # MAX_SEQ_LEN = 180
        if len(sequence) > MAX_NUM_SEQ:
            sequence = np.random.permutation(np.array(sequence, dtype=object)).tolist()[:MAX_NUM_SEQ]

        data_raw = np.zeros((MAX_NUM_SEQ, MAX_SEQ_LEN)).astype(np.float32)
        data_noise = np.zeros((MAX_NUM_SEQ, MAX_SEQ_LEN)).astype(np.float32)
        # valid_mask = np.zeros((MAX_NUM_SEQ,MAX_SEQ_LEN)).astype(np.bool)
        valid_mask = np.zeros((MAX_NUM_SEQ, MAX_SEQ_LEN)).astype(bool)

        # -----------return forkseq------------------------------
        # is_forkseq = np.zeros((MAX_NUM_SEQ,1)).astype(np.bool)
        fork_idx = []
        end_idx = []
        # -------------------------------------------------------

        noise_std_x = 0  # 0.1
        noise_std_y = 0  # 0.06
        noise_std_z = 0  # 0.02

        for i, seq in enumerate(sequence):
            _ind = 0
            for token in seq:
                if token[1] == "FORK":
                    data_noise[i][_ind] = range_quantize
                    data_raw[i][_ind] = range_quantize
                    valid_mask[i][_ind] = True
                    fork_idx.append([i, _ind])
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

                    data_noise[i][_ind] = range_quantize
                    data_raw[i][_ind] = range_quantize
                    valid_mask[i][_ind] = True
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

                    data_noise[i][_ind] = range_quantize
                    data_raw[i][_ind] = range_quantize
                    valid_mask[i][_ind] = True
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

                if token[1] == "MERGE":
                    data_noise[i][_ind] = range_quantize + 2
                    data_raw[i][_ind] = range_quantize + 2
                    valid_mask[i][_ind] = True
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

                    data_noise[i][_ind] = range_quantize + 2
                    data_raw[i][_ind] = range_quantize + 2
                    valid_mask[i][_ind] = True
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

                    data_noise[i][_ind] = range_quantize + 2
                    data_raw[i][_ind] = range_quantize + 2
                    valid_mask[i][_ind] = True
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

                x, y, z = token[0][0], token[0][1], token[0][2] + 0.33
                x_noise = x + np.random.normal(0, noise_std_x)
                y_noise = y + np.random.normal(0, noise_std_y)
                z_noise = z + np.random.normal(0, noise_std_z)

                data_raw[i][_ind] = round((y - min_range_y) * range_quantize / (max_range_y - min_range_y))
                data_raw[i][_ind] = np.clip(data_raw[i][_ind], 0, range_quantize - 1)
                data_noise[i][_ind] = round((y_noise - min_range_y) * range_quantize / (max_range_y - min_range_y))
                data_noise[i][_ind] = np.clip(data_noise[i][_ind], 0, range_quantize - 1)
                valid_mask[i][_ind] = True
                _ind += 1
                if _ind >= MAX_SEQ_LEN:
                    break

                data_raw[i][_ind] = round((x - min_range_x) * range_quantize / (max_range_x - min_range_x))
                data_raw[i][_ind] = np.clip(data_raw[i][_ind], 0, range_quantize - 1)
                data_noise[i][_ind] = round((x_noise - min_range_x) * range_quantize / (max_range_x - min_range_x))
                data_noise[i][_ind] = np.clip(data_noise[i][_ind], 0, range_quantize - 1)
                valid_mask[i][_ind] = True
                _ind += 1
                if _ind >= MAX_SEQ_LEN:
                    break

                data_raw[i][_ind] = round((z - min_range_z) * range_quantize / (max_range_z - min_range_z))
                data_raw[i][_ind] = np.clip(data_raw[i][_ind], 0, range_quantize - 1)
                data_noise[i][_ind] = round((z_noise - min_range_z) * range_quantize / (max_range_z - min_range_z))
                data_noise[i][_ind] = np.clip(data_noise[i][_ind], 0, range_quantize - 1)
                valid_mask[i][_ind] = True
                _ind += 1
                if _ind >= MAX_SEQ_LEN:
                    break

                if token[1] == "END":
                    data_raw[i][_ind] = range_quantize + 1
                    data_noise[i][_ind] = range_quantize + 1
                    valid_mask[i][_ind] = True
                    end_idx.append([i, _ind])
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

                    data_raw[i][_ind] = range_quantize + 1
                    data_noise[i][_ind] = range_quantize + 1
                    valid_mask[i][_ind] = True
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

                    data_raw[i][_ind] = range_quantize + 1
                    data_noise[i][_ind] = range_quantize + 1
                    valid_mask[i][_ind] = True
                    _ind += 1
                    if _ind >= MAX_SEQ_LEN:
                        break

        data_raw = data_raw.astype(np.int64).copy()
        data_noise = data_noise.astype(np.int64).copy()
        return data_raw, valid_mask, data_noise, fork_idx, end_idx
