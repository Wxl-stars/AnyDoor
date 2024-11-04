# TODO: 数据情况需要更新


# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {
    "HUMAN_LABEL_DET": {
        "s3://tf-labeled-res/": [
            "20220601_det_yueying_checked_footprintRFU",
            "20220608_det_yueying_checked",
            "20220610_det_yueying_checked",
            "20220611_det_yueying_checked",
            "20220612_det_yueying_checked",
            "20220615_det_yueying_checked",
            "20220616_det_yueying_checked",
            "20220617_det_yueying_checked",
            "20220618_det_yueying_checked",
            "20220619_det_yueying_checked",
            "20220621_det_yueying_checked",
            "20220622_det_yueying_checked",
            "20220623_det_yueying_checked",
            "20220624_det_yueying_checked",
            "20220625_det_yueying_checked",
            "20220702_det_yueying_checked",
            "20220703_det_yueying_checked",
            "20220706_det_yueying_checked",
            "20220707_det_yueying_checked",
            "20220708_det_yueying_checked",
            "20220709_det_yueying_checked",
            "20220710_det_yueying_checked",
            "20220711_det_yueying_checked",
            "20220712_det_yueying_checked",
            "20220713_det_yueying_checked",
            "20220714_det_yueying_checked",
            "20220715_det_yueying_checked",
            "20220716_det_yueying_checked",
            "20220717_det_yueying_checked",
            "20220718_det_yueying_checked",
            "20220719_det_yueying_checked",
            "20220720_det_yueying_checked",
            "20220722_det_yueying_checked",
            "20220723_det_yueying_checked",
            "20220724_det_yueying_checked",
            "20220725_det_yueying_checked",
            "20220726_det_yueying_checked",
            "20220727_det_yueying_checked",
            "20220728_det_yueying_checked",
            "20220729_det_yueying_checked",
            "20220730_det_yueying_checked",
            "20220731_det_yueying_checked",
            "20220810_det_yueying_checked",
            "20220811_det_yueying_checked",
            "20220816_det_yueying_checked",
        ],
        "s3://camera-perceptron/history/trainset/car2/": [
            "20220601-0603_0606-0607_det_fixed",
        ],
    },
    "HUMAN_LABEL_TRACK": {
        "s3://tf-labeled-res/": [
            "20220601_tracking_checked_footprintRFU",
            "20220608_tracking_checked",
            "20220610_tracking_checked",
            "20220615_tracking_checked",
            "20220616_tracking_yueying_checked",
            "20220617_tracking_yueying_checked",
            "20220618_tracking_checked",
            "20220619_tracking_checked",
            "20220621_tracking_checked",
            "20220622_tracking_checked",
            "20220623_tracking_checked",
            "20220624_tracking_checked",
            "20220625_tracking_checked",
            "20220702_tracking_yueying_checked",
            "20220703_tracking_yueying_checked",
            "20220706_tracking_checked",
            "20220707_tracking_checked",
            "20220708_tracking_yueying_checked",
            "20220709_tracking_checked",
            "20220710_tracking_checked",
            "20220711_tracking_checked",
            "20220712_tracking_checked",
            "20220713_tracking_checked",
            "20220714_tracking_yueying_checked",
            "20220716_tracking_checked",
            "20220717_tracking_checked",
            "20220718_tracking_checked",
            "20220719_tracking_checked",
            "20220720_tracking_checked",
            "20220722_tracking_yueying_checked",
            "20220723_tracking_yueying_checked",
            "20220724_tracking_yueying_checked",
            "20220725_tracking_yueying_checked",
            "20220726_tracking_yueying_checked",
            "20220810_tracking_yueying_checked",
            "20220811_tracking_yueying_checked",
            "20220815_tracking_yueying_checked",
            "20220816_tracking_yueying_checked",
            "20220817_tracking_yueying_checked",
        ],
    },
    "LIDAR_LABEL_DET": {
        "s3://tf-labeled-res/": [
            "20220815_det",
            "20220817_det",
            "20220818_det",
            "20220819_det",
        ],
    },
    "LIDAR_LABEL_TRACK": {
        "s3://tf-labeled-res/": [
            "20220818_tracking",
            "20220819_tracking",
        ],
    },
    "SYNTHETIC": {},
}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {
    "det_tracking_18w": "s3://camera-perceptron/sub_dataset/car_2/det+tracking_bus-18w.json",
    "det_train_9.6w": "s3://e2emodel-data/wm_car2_train9.6w.json",
}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = [
    "s3://camera-perceptron/benchmark/car_2/20220524_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_2/20220526_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_2/20220528_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_2/20220530_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_2/20220531_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_2/20220602_3dbmk-tracking_yueying_checked/",
    "s3://camera-perceptron/benchmark/car_2/20220616_3dbmk-tracking_yueying_checked/",
]


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {
    "CHENGQU_BMK": "s3://camera-perceptron/sub_dataset/car_2/chengqu_bmk.json",
    "GAOSU_BMK": "s3://camera-perceptron/sub_dataset/car_2/gaosu_bmk.json",
    "OTHER": "s3://camera-perceptron/sub_dataset/car_2/other_bmk.json",
    "CHENGQU_BMK_REPAIRED": "s3://e2emodel-data/wm_car2_bmk_chengqu_repaired.json",
}
