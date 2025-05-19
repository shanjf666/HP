#!/usr/bin/env bash
# =========================================
# 批量启动 LLaMA-Factory 训练脚本
# =========================================

set -e  # 任何命令出错即退出

# 需要训练的目录编号（可按需增删）
for dir in 3812 8280 9864; do
  echo "⏳ 正在训练目录 ${dir} ..."
  # 调用训练脚本，指定对应的训练目录
  python /root/autodl-tmp/HP/run.py --train-dir "/root/autodl-tmp/HP/train/${dir}"
done

echo "✅ 所有训练任务已完成"