#!/bin/bash

# 检查是否传入参数
if [ -z "$1" ]; then
  echo "Usage: $0 <integer>"
  exit 1
fi

# 获取输入整数
N=$1
SCENE_JSON=$2
echo "total gpus "$N""



# 检查输入是否为正整数
if ! [[ "$N" =~ ^[0-9]+$ ]]; then
  echo "Error: Argument must be a positive integer."
  exit 1
fi

# 获取当前日期
current_date=$(date +"%Y-%m-%d")
echo $current_date

# 遍历小于 N 的所有整数，并行运行 Python 脚本
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i python run_inference_e2e_for_vlm.py --rank_id "$i" --world_size "$N" --scene_json "$SCENE_JSON" --save_local &  # 将当前整数传递给 Python 脚本
done

wait

python data_process/check_all_label.py --key "random" --date "$current_date"