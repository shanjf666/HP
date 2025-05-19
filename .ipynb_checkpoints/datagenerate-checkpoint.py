"""
python datagenerate.py \
  --wait-file /root/autodl-tmp/HP/train/3812/wait_experiments.txt \
  --dataset-json /root/autodl-tmp/HP/LLaMA-Factory/data/dataset_info.json \
  --data-root /root/autodl-tmp/data/output/helpsteer2/transwaps
因为只是微调种子不同，数据集是相同的，所以只需要添加一次就好了
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------- CLI & 环境变量 -----------------------------
DEFAULT_WAIT_FILE = "/root/autodl-tmp/HP/train/wait_experiments.txt"
DEFAULT_DATASET_JSON = "/root/autodl-tmp/HP/LLaMA-Factory/data/dataset_info.json"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/data/output/helpsteer2/transwaps"

parser = argparse.ArgumentParser(
    description="将待训练实验列表写入 dataset_info.json",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--wait-file",
    type=Path,
    default=os.getenv("WAIT_FILE", DEFAULT_WAIT_FILE),
    help="wait_experiments.txt 的路径 (可用环境变量 WAIT_FILE 覆盖)",
)
parser.add_argument(
    "--dataset-json",
    type=Path,
    default=os.getenv("DATASET_JSON", DEFAULT_DATASET_JSON),
    help="dataset_info.json 的路径 (可用环境变量 DATASET_JSON 覆盖)",
)
parser.add_argument(
    "--data-root",
    type=Path,
    default=os.getenv("DATA_ROOT", DEFAULT_DATA_ROOT),
    help="数据集根目录 (可用环境变量 DATA_ROOT 覆盖)",
)
args = parser.parse_args()

WAIT_FILE: Path = args.wait_file.expanduser().resolve()
DATASET_JSON: Path = args.dataset_json.expanduser().resolve()
DATA_ROOT: Path = args.data_root.expanduser().resolve()

# ----------------------------- Logging 设置 ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ------------------------------ 工具函数 --------------------------------

def load_wait_list(path: Path) -> List[str]:
    """读取 wait_experiments.txt，返回实验名列表。"""
    if not path.is_file():
        logging.error("找不到 wait_experiments: %s", path)
        sys.exit(1)
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_dataset_info(path: Path) -> Dict[str, dict]:
    """读取或初始化 dataset_info.json。"""
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    logging.info("dataset_info.json 不存在，将创建新文件：%s", path)
    return {}

# ---------------------------- 主流程 ------------------------------------

def main() -> None:
    experiments = load_wait_list(WAIT_FILE)
    info = load_dataset_info(DATASET_JSON)

    added, skipped = 0, 0
    for exp in experiments:
        if exp in info:
            logging.warning("已存在条目，跳过: %s", exp)
            skipped += 1
            continue

        info[exp] = {
            "file_name": str(DATA_ROOT / f"{exp}.json"),
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
        added += 1

    # 确保输出目录存在
    DATASET_JSON.parent.mkdir(parents=True, exist_ok=True)

    # 写回 JSON（漂亮缩进）
    with DATASET_JSON.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logging.info("✅ 已写入 %s；新增 %d 条，跳过 %d 条。", DATASET_JSON, added, skipped)


if __name__ == "__main__":
    main()
