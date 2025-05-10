import subprocess
import logging
from pathlib import Path
import sys

# ----------- 路径配置 ----------
BASE_DIR   = Path("/root/autodl-tmp/HP/train")     # 统一根目录
WAIT_FILE  = BASE_DIR / "wait_experiments.txt"
READY_FILE = BASE_DIR / "ready_experiments.txt"
STRICT     = True   # True: 缺文件即退出；False: 跳过
# --------------------------------

def load_wait_list(path: Path) -> list[str]:
    if not path.is_file():
        logging.error("找不到 wait_experiments: %s", path)
        sys.exit(1)
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        logging.error("wait_experiments 为空：%s", path)
        sys.exit(1)
    return lines


def save_wait_list(path: Path, remaining: list[str]) -> None:
    txt = "\n".join(remaining) + ("\n" if remaining else "")
    path.write_text(txt, encoding="utf-8")


def append_ready(path: Path, exp: str) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(f"{exp}\n")


def run_yaml(cfg_path: Path, idx: int, total: int) -> bool:
    logging.info("(%d/%d) ➜ %s", idx, total, cfg_path.name)
    cmd = ["llamafactory-cli", "train", str(cfg_path)]
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
        yaml_path = BASE_DIR / f"{exp}.yaml"
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

        # ----------------- 成功后更新清单 -----------------
        append_ready(READY_FILE, exp)                 # 加到 ready
        remaining = experiments[idx:]                 # 剩余待跑
        save_wait_list(WAIT_FILE, remaining)          # 覆盖写回
        # ------------------------------------------------

    logging.info("全部完成 🎉")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()


