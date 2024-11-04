# TODO: 数据情况需要更新


# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {
    "HUMAN_LABEL_DET": {
        "s3://tf-labeled-res/": [
            "20220824_det_yueying_checked",
            "20220827_det_yueying_checked",
            "20220828_det_yueying_checked",
            "20220829_det_yueying_checked",
            "20220830_det_yueying_checked",
            "20220831_det_yueying_checked",
        ],
        "s3://tf-22q3-shared-data/labeled_data/car_101/": [
            "20220901_dp-det_yueying_checked",
            "20220902_dp-det_yueying_checked",
            "20220903_dp-det_yueying_checked",
            "20220904_dp-det_yueying_checked",
            "20220905_dp-det_yueying_checked",
            "20220906_dp-det_yueying_checked",
            "20220907_dp-det_yueying_checked",
            "20220908_dp-det_yueying_checked",
            "20220909_dp-det_yueying_checked",
            "20220910_dp-det_yueying_checked",
            "20220911_dp-det_yueying_checked",
            "20220912_dp-det_yueying_checked",
            "20220913_dp-det_yueying_checked",
            "20220914_dp-det_yueying_checked",
            "20220919_dp-det_yueying_checked",
            "20220920_dp-det_yueying_checked",
            "20220922_dp-det_yueying_checked",
            "20220923_dp-det_yueying_checked",
            "20220924_dp-det_yueying_checked",
            "20220925_dp-det_yueying_checked",
            "20220926_dp-det_yueying_checked",
            "20220928_dp-det_yueying_checked",
            "20220929_dp-det_yueying_checked",
            "20221001_dp-det_yueying_checked",
            "20221005_dp-det_yueying_checked",
        ],
    },
    "HUMAN_LABEL_TRACK": {
        "s3://tf-labeled-res/": [
            "20220824_tracking_yueying_checked",
            "20220825_tracking_yueying_checked",
            "20220826_tracking_yueying_checked",
            "20220827_tracking_yueying_checked",
        ],
        "s3://tf-22q3-shared-data/labeled_data/car_101/": [
            "20220901_dp-tracking_yueying_checked/",
            "20220902_dp-tracking_yueying_checked/",
        ],
    },
    "LIDAR_LABEL_DET": {},
    "LIDAR_LABEL_TRACK": {},
    "SYNTHETIC": {},
}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = [
    # 北京
    "s3://camera-perceptron/benchmark/car_101/20220825_3dbmk_tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_101/20220826_3dbmk_tracking_yueying_checked/",
    # 宁波
    "s3://camera-perceptron/benchmark/car_101/20220930_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_101/20221006_3dbmk-tracking_yueying_checked/",
    # 1114是自己整理的Detection类型的BMK
    "s3://camera-perceptron/benchmark/car_101/20221114_3dbmk-det_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_101/20221114_3dbmk-tracking_yueying_checked/",
]


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {
    "NINGBO_GAOSU_BMK": "s3://camera-perceptron/sub_dataset/car_101/ningbo_gaosu_bmk.json",
    "NINGBO_GAOSU_track_BMK": "s3://camera-perceptron/sub_dataset/car_101/ningbo_gaosu_velocity_bmk.json",
    "BEIJING_GAOSU_BMK": "s3://camera-perceptron/sub_dataset/car_101/beijing_gaosu_bmk.json",
    "RAINY_BMK": "s3://camera-perceptron/sub_dataset/car_101/rainy_bmk.json",
    "RAINT_GAOSU_BMK": "s3://camera-perceptron/sub_dataset/car_101/rainy_gaosu_bmk.json",
    "SUIDAO_BMK": "s3://camera-perceptron/sub_dataset/car_101/suidao_bmk.json",
    "OTHER": "s3://camera-perceptron/sub_dataset/car_101/other_bmk.json",
}
