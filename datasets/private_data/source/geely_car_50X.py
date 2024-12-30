# ----- 训练数据全集，定期增量回流 -----
TRAINSET = {}


# ----- 训练数据子集，人工设计规则 -----
TRAINSET_PARTIAL = {
    "Static": "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/json_collection/E171_Static.json", # frame 698396
    "debug": [
        's3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/static_obj/hardcase/zhichengzhu/labels_20240926/labeled/0004.json',
        's3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/static_obj/hardcase/zhichengzhu/labels_20240926/labeled/0005.json',
        's3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/static_obj/hardcase/zhichengzhu/labels_20240926/labeled/0006.json',
        # 's3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/static_obj/hardcase/zhichengzhu/labels_20240926/labeled/0007.json',
        's3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/static_obj/hardcase/zhichengzhu/labels_20240926/labeled/0008.json',
        ],
}


# ----- 测试数据全集，定期增量回流 -----
BENCHMARK = {}


# ----- 测试数据子集，人工设计规则 -----
BENCHMARK_PARTIAL = {

}
