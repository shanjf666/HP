from huggingface_hub import HfApi

api = HfApi()

# 上传大文件夹到 Hugging Face
api.upload_large_folder(
    repo_id="shanjf/hpdata",  # 仓库名称
    folder_path="/root/autodl-tmp/HP/data",  # 文件夹路径
    repo_type="dataset"  # 设置仓库类型为 "dataset"，如果上传的是模型请改为 "model"
)
