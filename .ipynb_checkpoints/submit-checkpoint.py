import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# 配置日志设置
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  # 日志格式
    datefmt="%Y-%m-%d %H:%M:%S",  # 日期格式
    handlers=[logging.StreamHandler(sys.stdout)],  # 输出到控制台
    level=logging.INFO,  # 日志级别为INFO
)
TO_DPO_TEMPLATE = (
    "python /root/autodl-tmp/HP/todpo.py "
    "{input_path}/{experiment_name}.jsonl "
    "{output_path}/{experiment_name}.json "
)
# 解析命令行参数
def get_args():
    # fmt: off
    description = """Utility CLI for easily submitting jobs to a local GPU"""

    # 创建 ArgumentParser 对象，用于解析命令行参数增
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

    # 添加命令行参数
    parser.add_argument("--experiment_path", type=Path, required=True,help="Path to a TXT file containing the experiments.")
    parser.add_argument("--input_path", type=str, default="data/",help="Path to the local directory containing the datasets.")
    parser.add_argument("--output_path", type=str, default="output_path/",help="Path to the local directory to save the models.")
    parser.add_argument("--sort_by_swaps", action="store_true", default=False,help="If set, will prioritize running experiments with high swaps.")
    # fmt: on
    return parser.parse_args()

# 主程序执行逻辑
def main():
    # 获取命令行参数
    args = get_args()

    # 读取实验文件，获取实验名称列表
    experiment_path: Path = args.experiment_path
    with experiment_path.open("r") as f:
        experiment_names = f.read().splitlines()

    # 如果设置了根据交换量排序实验，则按交换量排序
    if args.sort_by_swaps:
        experiment_names = sorted(
            experiment_names,
            key=lambda x: int(x.split("SWAPS_")[1].split("::")[0]),  # 获取交换量值进行排序
            reverse=True,  # 按降序排序
        )

    # 为每个实验创建训练命令
    commands_for_experiments = []
    for idx, experiment_str in enumerate(experiment_names):
        if "::" in experiment_str:
            experiment_name, _ = experiment_str.split("::")
        else:
            experiment_name = experiment_str
        cmd = TO_DPO_TEMPLATE.format(
            input_path = args.input_path,
            experiment_name = experiment_name,
            output_path = args.output_path
        )
        logging.info(cmd)
        subprocess.run(cmd, shell=True)
    logging.info("All jobs finished. You can track the logs above.")

# 执行主程序
if __name__ == "__main__":
    main()