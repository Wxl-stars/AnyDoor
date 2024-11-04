# TODO: 数据情况需要更新
# TODO: 还有4个BMK在训练数据里


# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {
    "HUMAN_LABEL_DET": {},
    "HUMAN_LABEL_TRACK": {},
    "LIDAR_LABEL_DET": {},
    "LIDAR_LABEL_TRACK": {},
    "SYNTHETIC": {},
}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {
    "wm4_1003_1203_det_2576": "s3://camera-perceptron/sub_dataset/car_4/wm4_train_1003_1203_det_2576.json",
    "wm_car4_train_20230306": "s3://camera-perceptron/sub_dataset/car_4/wm_car4_train_20230306.json",
    "debug": [  # NOTE: list形式也已经支持，适用于临时使用，正常训练和测试请dump成制定形式的json
        "s3://malihua-data/private_dataset/car_4/20221003_dp-det_yueying_checked/ppl_bag_20221003_060413_det/v0_221018_180235/0007.json",
        "s3://malihua-data/private_dataset/car_4/20221003_dp-det_yueying_checked/ppl_bag_20221003_132227_det/v0_221019_172008/0002.json",
        "s3://malihua-data/private_dataset/car_4/20221005_dp-det_yueying_checked/ppl_bag_20221005_145249_det/v0_221020_194848/0000.json",
        "s3://malihua-data/private_dataset/car_4/20221211_dp-det_yueying_checked/ppl_bag_20221211_082022_det/v0_230120_165212/0001.json",
        "s3://malihua-data/private_dataset/car_4/20230101_dp-det_yueying_checked/ppl_bag_20230101_092037_det/v0_230128_015220/0011.json",
    ],
    "wm4_det": "s3://e2emodel-data/data-collect/car4-det.json",
    "wm4_tracking": "s3://e2emodel-data/data-collect/car4-tracking.json",
    # 0601
    "wm4_train_0601": "s3://camera-perceptron/sub_dataset/car_4/wm4_train_det_20230601.json",
}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = {
    "s3://camera-perceptron/benchmark/car_4/20221007_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_4/20221010_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_4/20221015_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_4/20221021_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_4/20221022_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_4/20221029_3dbmk-tracking_yueying_checked/",
}

# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {
    "CHENGQU_BMK": "s3://camera-perceptron/sub_dataset/car_4/chengqu_bmk.json",
    "GAOSU_BMK": "s3://camera-perceptron/sub_dataset/car_4/gaosu_bmk.json",
    "OTHER": "s3://camera-perceptron/sub_dataset/car_4/other_bmk.json",
    "wm_car4_chengqu_test_20230306": "s3://camera-perceptron/sub_dataset/car_4/wm_car4_chengqu_test_20230306.json",
    "WM_CAR4_BMK_CHENGQU": "s3://e2emodel-data/data-collect/car4-bmk-chengqu.json",
    "WM_CAR4_BMK_GAOSU": "s3://e2emodel-data/data-collect/car4-bmk-gaosu.json",
    "WM_CAR4_BMK_OTHER": "s3://e2emodel-data/data-collect/car4-bmk-other.json",
    "CHENGQU_freespace_BMK": "s3://malihua-data/tmp/car4_chengqu_bmk.json",
    # CHENGQU_BMK的子集, 仅包含一个json，方便trt上单卡评测对点
    "CHENGQU_BMK_trt": "s3://far-range-shared/data/bmk/wm_car_4/CHENGQU_BMK_for_trt.json",
}
