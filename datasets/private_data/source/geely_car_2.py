# TODO: 数据情况需要补充

TRAINSET = {
    "HUMAN_LABEL_DET": {},
    "HUMAN_LABEL_TRACK": {},
    "LIDAR_LABEL_DET": {},
    "LIDAR_LABEL_TRACK": {},
    "SYNTHETIC": {},
}

# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {
    "TRAIN_22Q4_DATALIST": "s3://far-range-shared/data/common_data_list/train_data/car_102/car102_train_22Q4Det.json",
    "TRAIN_23Q1_DET_DATALIST": "s3://far-range-shared/data/common_data_list/train_data/car_102/car102_train_23Q1Det.json",
    "TRAIN_23Q1_TRACKING_DATALIST": "s3://far-range-shared/data/common_data_list/train_data/car_102/car102_train_23Q1Track.json",
}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = {"s3://camera-perceptron/benchmark/car_102/20221210_3dbmk_dp-det_yueying_checked/"}


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {
    "JL_CAR2_BMK": "s3://camera-perceptron/sub_dataset/car_102/bmk_3k.json",
    # "Geely2_GAOSU_BMK": "s3://httdet3d/bmks/geely2_gs_bmk.json",
    "Geely2_GAOSU_BMK": "s3://httdet3d/bmks/geely2_gs_bmk_repaired.json",
}
