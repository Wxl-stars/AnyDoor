# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {
    "HF_E2E_city_label": "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/json_collection/E2E_HF_city_label.json", # keyframe 25375, frame 1031600
}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = {}


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {

}
