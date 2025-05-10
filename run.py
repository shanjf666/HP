#!/usr/bin/env python3
"""
Sequential trainer driven by wait_experiments

步骤
----
1. 读取 /root/autodl-tmp/HP/test/wait_experiments
   - 每一行 {experiment} 代表要训练的 YAML: /root/autodl-tmp/HP/train/{experiment}.yaml
   - 忽略空行、前后空白
2. 按读取顺序依次运行 llamafactory-cli train <yaml>
3. 若某个任务失败 (非零退出码) 就停止执行

如需改动：
- WAIT_FILE、YAML_DIR 改成自己的路径
- 想跳过不存在的文件而不是报错，可把 `strict=True` 改成 `False`
"""

import subprocess
import logging
from pathlib import Path
import sys

WAIT_FILE = Path("/root/autodl-tmp/HP/train/wait_experiments.txt") 
YAML_DIR  = Path("/root/autodl-tmp/HP/train")
STRICT    = True   # True: 缺文件直接报错退出；False: 打警告然后跳过

def load_wait_list(path: Path) -> list[str]:
    if not path.is_file():
        logging.error("找不到 wait_experiments: %s", path)
        sys.exit(1)

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        logging.error("wait_experiments 为空：%s", path)
        sys.exit(1)
    return lines


def run_yaml(cfg_path: Path, idx: int, total: int) -> bool:
    """调用 llama‑factory‑cli 训练单个 YAML；成功返回 True"""
    logging.info("(%d/%d) ➜ %s", idx, total, cfg_path.name)
    cmd = ["llamafactory-cli", "train", str(cfg_path)]
    print(cmd)
    result = subprocess.run(cmd)
    if result.returncode == 0:
        logging.info("✓ 完成 %s", cfg_path.name)
        return True
    else:
        logging.error("✗ %s 失败（退出码 %d）", cfg_path.name, result.returncode)
        return False


def main() -> None:
    experiments = load_wait_list(WAIT_FILE)
    total = len(experiments)
    logging.info("将按顺序训练 %d 个实验。", total)

    for idx, exp in enumerate(experiments, start=1):
        yaml_path = YAML_DIR / f"{exp}.yaml"
        if not yaml_path.is_file():
            msg = f"缺少 YAML 文件：{yaml_path}"
            if STRICT:
                logging.error(msg)
                sys.exit(1)
            else:
                logging.warning(msg + "，跳过。")
                continue

        ok = run_yaml(yaml_path, idx, total)
        if not ok:
            logging.error("中断执行。你可以修复问题后再重跑剩余任务。")
            sys.exit(1)

    logging.info("全部完成 🎉")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

