# TODO: 数据情况需要更新


# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {
    "0929": "s3://peppa-meg-detect/tmpdata/datasets/car9_train_0929.json",
    "1023": "s3://peppa-meg-detect/tmpdata/datasets/car9_train_1023_more.json",
    "prelabels_radar": "s3://gongjiahao-share/e2e/test-file/HF9_RADAR_prelabel_path_e2e.json",
    "prelabels_radar_all": "/home/gongjiahao/HF9_RADAR_prelabelALL_path_e2e.json",
    "HF_e2e_labeled": "s3://gongjiahao-share/e2e/test-file/HF_e2e_labeled.json",
    "HF_e2e_labeled_861": "s3://gongjiahao-share/e2e/test-file/HF_e2e_labeled_861.json",
    "HF_e2e_labeled_861_with_occ": "s3://gongjiahao-share/e2e/test-file/HF_e2e_labeled_with_occ.json",
    "HF_e2e_labeled_bad_radar": "s3://gongjiahao-share/e2e/test-file/HF_e2e_labeled_bad_radar.json",
    "HF_e2e_labeled_bad_radar_with_occ": "s3://gongjiahao-share/e2e/test-file/HF_e2e_labeled_bad_radar_with_occ.json",
    "0720_0831": "s3://end2end/data/paths_list/det/hf/hf_car9_train_0720_0831.json",
    "0901_0926_filter_bmk": "s3://end2end/data/paths_list/det/hf/hf_car9_train_0901_0926_filted_bmk.json",
    "0720_0831_add_map_prelabels": "s3://end2end/data/paths_list/det/hf/hf_car9_train_0720_0831_add_map_prelabels.json",
    "0901_0926_filter_bmk_add_map_prelabels": "s3://end2end/data/paths_list/det/hf/hf_car9_train_0901_0926_filted_bmk_add_map_prelabels.json",
    "debug_sample_map": "s3://wangningzi-data/dataset/e2e_map_dataset/reorg_sample/sample_trainset_det_map.json",
    "debug_sample_track": "s3://wangningzi-data/dataset/e2e_map_dataset/reorg_sample/sample_trainset_track.json",
    "quick": "/home/gongjiahao/bmk_quick.json",
    "map_share_prelabels_bad_radar": "s3://gongjiahao-share/e2e/test-file/HF_map_share_prelabel_bad_radar.json",  # 带了map
    "map_share_prelabels": "s3://gongjiahao-share/e2e/test-file/HF_map_share_prelabel.json",  # 带了map
    "mapbmk_align": "s3://dwj-share/tmp/admap_bmk/base_align_prelabel_list.json",  # prelabel
    "CAR9_BMK_OCC_DAY": "s3://peppa-meg-detect/tmpdata/datasets/car9_bmk_occ_day.json",
    "bmk_new_withOCC": "s3://end2end/data/paths_list/track/HF_e2e_bmk_107_with_occ.json",
    "map_share_prelabels_for_e2e": "s3://gongjiahao-share/e2e/test-file/HF_map_share_prelabel_for_e2e.json",
}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = []


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {
    "CAR12_BMK_LIST_LABEL_OCC_GS": "s3://hangzhou-tf/tf_labels/动态感知2dbmk/export_labels/100874/car_12/20230920_dp-det_yueying_checked/ppl_bag_20230920_163417_det/v0_230926_152943/",
    # "CAR9_BMK_OCC": '/data/tmpdata/car9_bmk_radar.json',
    "CAR9_BMK_OCC_DAY": "s3://peppa-meg-detect/tmpdata/datasets/car9_bmk_occ_day.json",
    "CAR9_BMK_OCC_DAY_e2e": "s3://end2end/data/paths_list/track/hf9_track_bmk119.json",
    "CAR9_BMK_OCC_DAY_e2e_filter": "/home/gongjiahao/hf9_track_bmk119.json",
    "bmk_new": "s3://end2end/data/paths_list/track/HF_e2e_bmk_107.json",
    "bmk_new_withOCC": "s3://end2end/data/paths_list/track/HF_e2e_bmk_107_with_occ.json",
    "trt_eval_sample_300f": "s3://wangningzi-data/dataset/e2e_map_dataset/data_lists_v5/trt_eval_300f_sample_hf.json",
    "trt_eval_sample_14scene": "s3://end2end/data/paths_list/track/hf9_track_bmk15_trt_eval.json",
    "quick": "/home/gongjiahao/bmk_quick.json",
    "mapbmk_align": "s3://dwj-share/tmp/admap_bmk/base_align_prelabel_list.json",  # prelabel
    "map_share_prelabels": "s3://gongjiahao-share/e2e/test-file/HF_map_share_prelabel.json",  # 带了map
    "map20w_val": "s3://dwj-share/tmp/admap_bmk/base_align_prelabel_BMK_6k.json",
    "map20w_val_down100": "s3://dwj-share/tmp/admap_bmk/base_align_prelabel_BMK_6k_downsample100.json",
    "map20w_val_mf": "s3://mj-share/admap_e2e/data_e2eformat_bmk_mf/6k_bmk_mf_list.json",  # bmk_6k_mf
    "map-test": "/home/gongjiahao/6k_bmk_mf_list.json",
}
