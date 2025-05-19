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
NUM_SEEDS = 3  # 为每个实验生成3个不同的随机种子

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
seed: {seed}
output_dir: /root/autodl-tmp/HP/data/{seed}/{dataset_name}
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
    p.add_argument("--seed", type=int, default=42, help="随机实验的随机种子")
    return p.parse_args()


def main() -> None:
    args = get_args()
    random.seed(args.seed)  # 设置全局随机种子

    # 1. 读取 &（可选）排序
    names = [line.strip() for line in args.experiment_path.read_text().splitlines() if line.strip()]
    if args.sort_by_swaps:
        names.sort(key=lambda n: int(n.split("SWAPS_")[1].split("::")[0]), reverse=True)

    # 2. 随机抽 SAMPLE_SIZE 条
    if len(names) > SAMPLE_SIZE:
        names = random.sample(names, SAMPLE_SIZE)
        logging.info("已随机抽取 %d/%d 个实验名。", SAMPLE_SIZE, len(names))

    # 3. 创建输出目录
    args.output_path.mkdir(parents=True, exist_ok=True)

    global_picked = []  # 用于全局记录所有实验
    seed_experiments = {}  # 记录每个种子对应的实验

    # 为每个实验生成NUM_SEEDS个不同的随机种子
    seeds = random.sample(range(1, 9999), NUM_SEEDS)  # 生成不重复的随机种子
    
    # 4. 生成 YAML
    for name in names:
        exp_name = name.split("::")[0]  # 去掉可能的 ::comment
        
        for seed in seeds:
            # 初始化该种子的实验列表（如果尚未存在）
            if seed not in seed_experiments:
                seed_experiments[seed] = []
            
            seed_experiments[seed].append(exp_name)
            global_picked.append(f"{seed}/{exp_name}")
            
            # 创建种子目录
            seed_dir = args.output_path / str(seed)
            seed_dir.mkdir(parents=True, exist_ok=True)
            
            yaml_text = TEMPLATE.format(dataset_name=exp_name, seed=seed)
            yaml_file = seed_dir / f"{exp_name}.yaml"
            yaml_file.write_text(yaml_text, encoding="utf-8")
            logging.info("✅ 生成 %s", yaml_file)

    # 5. 为每个种子创建wait_experiments.txt
    for seed, experiments in seed_experiments.items():
        seed_dir = args.output_path / str(seed)
        picked_file = seed_dir / "wait_experiments.txt"
        picked_file.write_text("\n".join(experiments) + "\n", encoding="utf-8")
        logging.info("📄 为种子 %d 创建实验列表: %s", seed, picked_file)

    # 6. 创建全局的wait_experiments.txt
    global_picked_file = args.output_path / "wait_experiments.txt"
    global_picked_file.write_text("\n".join(global_picked) + "\n", encoding="utf-8")
    logging.info("📄 创建全局实验列表: %s", global_picked_file)
    
    logging.info("全部完成，共生成 %d 个 YAML。", len(global_picked))

if __name__ == "__main__":
    main()