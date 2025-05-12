import json
from pathlib import Path
import logging
import sys

WAIT_FILE     = Path("/root/autodl-tmp/HP/train/wait_experiments.txt")
DATASET_JSON  = Path("/root/autodl-tmp/HP/LLaMA-Factory/data/dataset_info.json")   
DATA_ROOT     = Path("/root/autodl-tmp/data/output/helpsteer2/transwaps")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_wait_list(path: Path) -> list[str]:
    if not path.is_file():
        logging.error("找不到 wait_experiments: %s", path)
        sys.exit(1)
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_dataset_info(path: Path) -> dict:
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    logging.info("dataset_info.json 不存在，将创建新文件。")
    return {}


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
                "chosen":   "chosen",
                "rejected": "rejected"
            }
        }
        added += 1

    # 写回 JSON（漂亮缩进）
    with DATASET_JSON.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logging.info("✅ 已写入 %s；新增 %d 条，跳过 %d 条。", DATASET_JSON, added, skipped)


if __name__ == "__main__":
    main()
