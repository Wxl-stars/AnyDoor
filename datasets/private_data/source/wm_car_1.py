# 详细见 https://tf-discourse.megvii-inc.com/t/topic/1426

# TODO: 数据情况需要更新
# TODO: `wangningzi-data`需要改为公共路径

# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {
    "HUMAN_LABEL_DET": {
        "s3://wangningzi-data/dataset/private_occlution_human_label": [
            "20220211_det_checked",
            "20220218_det_checked",
            "20220225_det_checked",
            "20220309_det_checked",
            "20220316_det_yueying_checked",
        ]
    },
    "HUMAN_LABEL_TRACK": {
        "s3://wangningzi-data/dataset/private_occlution_human_label": [
            "20220204_tracking_checked",
            "20220211_tracking_checked",
            "20220218_tracking_checked",
            "20220225_tracking_checked",
            "20220309_tracking_checked",
            "20220316_tracking_checked",
        ]
    },
    "LIDAR_LABEL_DET": {
        "s3://wangningzi-data/dataset/private_occlution_det_sweep": [
            "20220211_det",
            "20220218_det",
            "20220225_det",
            "20220309_det",
            "20220316_det",
            "20220323_det",
            "20220330_det",
            "20220406_det",
            "20220413_det",
            "20220420_det",
        ]
    },
    "LIDAR_LABEL_TRACK": {
        "s3://wangningzi-data/dataset/private_occlution_motor_fix_2": [
            "20220204_tracking",
            "20220211_tracking",
            "20220218_tracking",
            "20220225_tracking",
            "20220309_tracking",
            "20220316_tracking",
            "20220323_tracking",
            "20220330_tracking",
            "20220406_tracking",
            "20220413_tracking",
            "20220420_tracking",
        ]
    },
    "SYNTHETIC": {},
}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {"train_6w_checked": "s3://camera-perceptron/sub_dataset/car_1/train_6w_checked.json"}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = ["s3://camera-perceptron/benchmark/car_1/20220215_BMK_checked_Footprint_RFU/"]


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {"WM_CAR1_BMK": "s3://camera-perceptron/sub_dataset/car_1/bmk_6k.json"}
