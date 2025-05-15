import subprocess
import logging
from pathlib import Path
import sys

BASE_DIR   = Path("/root/autodl-tmp/HP/data")
READY_FILE = Path("/root/autodl-tmp/HP/train/ready_experiments.txt")
TERM_DIR   = Path("/root/autodl-tmp/data1")

def load_wait_list(path: Path) -> list[str]:
    if not path.is_file():
        logging.error("找不到 wait_experiments: %s", path)
        sys.exit(1)
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        logging.error("wait_experiments 为空：%s", path)
        sys.exit(1)
    return lines
    
def main() -> None:
    experiments = load_wait_list(READY_FILE)
    # 只处理前10个文件夹
    experiments = experiments[0:10]
    total = len(experiments)
    logging.info("将按顺序移动训练 %d 个实验。", total)

    for idx, exp in enumerate(experiments, start=1):
        data_path = BASE_DIR / f"{exp}"
        term = TERM_DIR / f"{exp}"
        cmd = ["mv", term, data_path]
        subprocess.run(cmd, check=True)

    logging.info("全部完成 🎉")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
