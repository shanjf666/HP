"""
python /root/autodl-tmp/HP/yamlgenerate.py \
  --experiment_path /root/autodl-tmp/data/output/helpsteer2/experiments.txt \
  --output_path /root/autodl-tmp/HP/train
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import random           
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

SAMPLE_SIZE = 20

TEMPLATE = """\
### model
model_name_or_path: /root/autodl-tmp/model/tulu-2-dpo-7b
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_beta: 0.1
pref_loss: simpo  # choices: [sigmoid (dpo), orpo, simpo]

### dataset
dataset: {dataset_name}
template: tulu2
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /root/autodl-tmp/HP/data/{dataset_name}
logging_steps: 50
save_steps: 1000
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# eval_dataset: dpo_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
"""

def get_args():
    p = argparse.ArgumentParser(
        description="Generate YAML configs in bulk",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--experiment_path", type=Path, required=True,
                   help="文本文件，每行一个 experiment_name")
    p.add_argument("--output_path", type=Path, default=Path("yaml_configs"),
                   help="*.yaml 的保存目录")
    p.add_argument("--sort_by_swaps", action="store_true",
                   help="若文件名包含 SWAPS_123，则按数字降序排序")
    return p.parse_args()


def main() -> None:
    args = get_args()

    # 1. 读取 &（可选）排序
    names = [line.strip() for line in args.experiment_path.read_text().splitlines() if line.strip()]
    if args.sort_by_swaps:
        names.sort(key=lambda n: int(n.split("SWAPS_")[1].split("::")[0]), reverse=True)

    # 2. 随机抽 SAMPLE_SIZE 条
    if len(names) > SAMPLE_SIZE:
        random.seed()                 # 如需可复现改成固定种子
        names = random.sample(names, SAMPLE_SIZE)
        logging.info("已随机抽取 %d/%d 个实验名。", SAMPLE_SIZE, len(names))

    # 3. 创建输出目录
    args.output_path.mkdir(parents=True, exist_ok=True)

    picked = []                        # 用于记录抽中的数据集名

    # 4. 生成 YAML
    for name in names:
        exp_name = name.split("::")[0]     # 去掉可能的 ::comment
        picked.append(exp_name)

        yaml_text = TEMPLATE.format(dataset_name=exp_name)
        yaml_file = args.output_path / f"{exp_name}.yaml"
        yaml_file.write_text(yaml_text, encoding="utf-8")
        logging.info("✅ 生成 %s", yaml_file)

    # 5. 把抽样结果写到 experiment.txt
    picked_file = args.output_path / "wait_experiments.txt"
    picked_file.write_text("\n".join(picked) + "\n", encoding="utf-8")
    logging.info("📄 已保存选定实验名至 %s", picked_file)

    logging.info("全部完成，共生成 %d 个 YAML。", len(picked))

if __name__ == "__main__":
    main()
