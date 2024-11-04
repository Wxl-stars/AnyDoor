# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {
    "HF_e2e_labeled_861": "s3://wangningzi-data/e2e_data/data_lists/e2e/hf/HF_e2e_labeled_861.json",
    "HF_e2e_labeled_bad_radar": "s3://wangningzi-data/e2e_data/data_lists/e2e/hf/HF_e2e_labeled_bad_radar.json",
    "map_share_prelabels_for_e2e": "s3://gongjiahao/wangningzi-data/e2e_data/data_lists/e2e/hf/HF_map_share_prelabel.json",
    "0720_0831": "s3://wangningzi-data/e2e_data/data_lists/det/hf/hf_car9_train_0720_0831.json",
    "0901_0926_filter_bmk": "s3://wangningzi-data/e2e_data/data_lists/det/hf/hf_car9_train_0901_0926_filted_bmk.json",
    "bmk_new_withOCC": "s3://wangningzi-data/e2e_data/data_lists/e2e/hf/HF_e2e_bmk_107.json",
    "debug_100": "s3://gengyu/e2e_data/hf/HF_debug_wmap.json",
    "debug_100_city": "s3://gengyu/e2e_data/hf/HF_debug_city.json",
    "track_good_radar_97json": "s3://wangningzi-data/e2e_data/data_to_transfer0723_merge/hf_label_1410_jsons_goodradar.json", #0.03M 是下面标labels的子集
    "track_bad_radar_608json": "s3://wangningzi-data/e2e_data/data_to_transfer0723_merge/hf_label_1410_jsons_badradar.json",  #0.18M 是下面标labels的子集
    "track_good_2738json": "s3://wangningzi-data/e2e_data/data_to_transfer0723_merge/hf_prelabel_0717_track_4441_goodradar.json", #0.82M pre_labels
    "track_bad_1703json": "s3://wangningzi-data/e2e_data/data_to_transfer0723_merge/hf_prelabel_0717_track_4441_badradar.json.json", #0.51M pre_labels
}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = {}


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {
    "bmk_new_withOCC": "s3://wangningzi-data/e2e_data/data_lists/e2e/hf/HF_e2e_bmk_107.json",
    "HF_e2e_labeled_861": "s3://wangningzi-data/e2e_data/data_lists/e2e/hf/bmk100_extrack_from_HF_e2e_labeled_861.json",
    "map20w_val": "s3://wangningzi-data/e2e_data/data_lists/map_bmk/base_align_prelabel_BMK_6k.json",
    "map20w_val_mf": "s3://wangningzi-data/e2e_data/data_lists/map_bmk/6k_bmk_mf_list.json",
}
