"""
# 指定根目录
python /root/autodl-tmp/HP/run.py --train-dir /root/autodl-tmp/HP/train/3812
python /root/autodl-tmp/HP/run.py --train-dir /root/autodl-tmp/HP/train/8280
python /root/autodl-tmp/HP/run.py --train-dir /root/autodl-tmp/HP/train/9864
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

# -------------------- CLI / ENV 处理 --------------------
DEFAULT_TRAIN_DIR = "/root/autodl-tmp/HP/train"

def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in {"1", "true", "yes", "y"}

parser = argparse.ArgumentParser(
    description="按队列顺序训练 LLaMA-Factory 实验",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--train-dir",
    type=Path,
    default=os.getenv("TRAIN_DIR", DEFAULT_TRAIN_DIR),
    help="训练相关文件所在根目录 (可用环境变量 TRAIN_DIR 覆盖)",
)
parser.add_argument(
    "--strict",
    type=str2bool,
    default=str2bool(os.getenv("STRICT", "1")),
    help="严格模式：缺失 YAML 时立即退出 (0/1, true/false)",
)
args = parser.parse_args()

TRAIN_DIR: Path = args.train_dir.expanduser().resolve()
WAIT_FILE: Path = TRAIN_DIR / "wait_experiments.txt"
READY_FILE: Path = TRAIN_DIR / "ready_experiments.txt"
STRICT: bool = args.strict

# --------------------- 工具函数 ---------------------

def load_wait_list(path: Path) -> List[str]:
    if not path.is_file():
        logging.error("找不到 wait_experiments: %s", path)
        sys.exit(1)
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        logging.error("wait_experiments 为空：%s", path)
        sys.exit(1)
    return lines


def save_wait_list(path: Path, remaining: List[str]) -> None:
    txt = "\n".join(remaining) + ("\n" if remaining else "")
    path.write_text(txt, encoding="utf-8")


def append_ready(path: Path, exp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(f"{exp}\n")


def run_yaml(cfg_path: Path, idx: int, total: int) -> bool:
    logging.info("(%d/%d) ➜ %s", idx, total, cfg_path.name)
    cmd = ["llamafactory-cli", "train", str(cfg_path)]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        logging.info("✓ 完成 %s", cfg_path.name)
        return True
    logging.error("✗ %s 失败（退出码 %d）", cfg_path.name, result.returncode)
    return False

# --------------------- 主流程 ---------------------

def main() -> None:
    logging.info("使用 TRAIN_DIR: %s", TRAIN_DIR)
    experiments = load_wait_list(WAIT_FILE)
    total = len(experiments)
    logging.info("将按顺序训练 %d 个实验。", total)

    for idx, exp in enumerate(experiments, start=1):
        yaml_path = TRAIN_DIR / f"{exp}.yaml"
        if not yaml_path.is_file():
            msg = f"缺少 YAML 文件：{yaml_path}"
            if STRICT:
                logging.error(msg)
                sys.exit(1)
            logging.warning(msg + "，跳过。")
            continue

        if not run_yaml(yaml_path, idx, total):
            logging.error("中断执行。可修复问题后重跑剩余任务。")
            sys.exit(1)

        # -------- 成功后更新队列 --------
        append_ready(READY_FILE, exp)
        remaining = experiments[idx:]  # idx 已经是下一个
        save_wait_list(WAIT_FILE, remaining)
        # --------------------------------

    logging.info("全部完成 🎉")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()



