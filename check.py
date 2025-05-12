from pathlib import Path

# 目标文件路径
file_path = Path("/root/autodl-tmp/data/output/helpsteer2/transwaps/human_datamodel_counts_7000_ID__055e91677d7d41aa990440ebc5e13f75__SWAPS_5709.json")

# 检查文件是否存在
if file_path.exists():
    print(f"文件 {file_path} 存在。")
else:
    print(f"文件 {file_path} 不存在。")

# 检查是否是文件而不是目录
if file_path.is_file():
    print(f"{file_path} 是一个普通文件。")
else:
    print(f"{file_path} 不是一个普通文件（可能是目录）。")
