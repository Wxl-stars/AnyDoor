# TODO: 数据情况需要更新


# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {
    "HUMAN_LABEL_DET": {
        "s3://tf-22q3-shared-data/labeled_data/car_3/": [
            "20220922_dp-det_yueying_checked",
            "20220928_dp-det_yueying_checked",
            "20220929_dp-det_yueying_checked",
        ]
    },
    "HUMAN_LABEL_TRACK": {
        "s3://tf-22q3-shared-data/labeled_data/car_3/": [
            "20220922_dp-tracking_yueying_checked",
            "20220923_dp-tracking_yueying_checked",
            "20220924_dp-tracking_yueying_checked",
            "20220925_dp-tracking_yueying_checked",
            "20220926_dp-tracking_yueying_checked",
            "20220927_dp-tracking_yueying_checked",
            "20220928_dp-tracking_yueying_checked",
            "20220929_dp-tracking_yueying_checked",
        ]
    },
    "LIDAR_LABEL_DET": {},
    "LIDAR_LABEL_TRACK": {},
    "SYNTHETIC": {},
}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {
    "wm3_0903_0930_det_731": "s3://camera-perceptron/sub_dataset/car_3/wm3_train_0903_0930_731.json",  # NOTE: special crop aug
    "wm3_0930_1106_det_471": "s3://camera-perceptron/sub_dataset/car_3/wm3_train_0930_1106_471.json",  # normal sensor location
    "wm34_4w_training": "s3://e2emodel-data/data-collect/wm-car34-4w-training.json",
    "wm34-mini": "s3://e2emodel-data/data-collect/car34-mini.json",
    "wm3_det": "s3://e2emodel-data/data-collect/car3-det.json",
    "wm3_tracking": "s3://e2emodel-data/data-collect/car3-tracking.json",
}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = ["s3://camera-perceptron/benchmark/car_3/20220921_3dbmk-tracking_yueying_checked/"]


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {
    "WM_CAR3_BMK": "s3://camera-perceptron/sub_dataset/car_3/bmk_5k.json",
    "WM_CAR34_BMK": "s3://wposs/car34-bmk/car34bmk.json",
    "WM_CAR3_BMK_CHENGQU": "s3://e2emodel-data/data-collect/car3-bmk-chengqu.json",
    "WM_CAR3_BMK_GAOSU": "s3://e2emodel-data/data-collect/car3-bmk-gaosu.json",
    "WM_CAR3_BMK_OTHER": "s3://e2emodel-data/data-collect/car3-bmk-other.json",
}
