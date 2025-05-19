# LLaMA‑Factory 训练流程

> 本 README 记录了从数据预处理到回归模型训练的完整工作流，便于在本地或服务器上快速复现。

---

## 1. 克隆 LLaMA‑Factory 项目

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

## 2. 批量转换数据集结构（`jsonl` → `json`）

```bash
python submit.py \
  --experiment_path /root/autodl-tmp/data/output/helpsteer2/experiments.txt \
  --input_path      /root/autodl-tmp/data/output/helpsteer2/swaps \
  --output_path     /root/autodl-tmp/data/output/helpsteer2/transwaps
```

> 说明：`submit.py` 会遍历 `--input_path` 下的 `*.jsonl` 文件，并将其转换为同名 `json` 文件，输出至 `--output_path`。

## 3. 生成训练用 YAML 与 `wait_experiments.txt`

```bash
python yamlgenerate.py \
  --experiment_path /root/autodl-tmp/data/output/helpsteer2/experiments.txt \
  --output_path     /root/autodl-tmp/HP/train
```

## 4. 批量在 `dataset_info.json` 中注册数据集

```bash
python datagenerate.py \
  --wait-file /root/autodl-tmp/HP/train/3812/wait_experiments.txt \
  --dataset-json /root/autodl-tmp/HP/LLaMA-Factory/data/dataset_info.json \
  --data-root /root/autodl-tmp/data/output/helpsteer2/transwaps
```


## 5. 启动模型训练

```bash
python /root/autodl-tmp/HP/run.py --train-dir /root/autodl-tmp/HP/train/3812
python /root/autodl-tmp/HP/run.py --train-dir /root/autodl-tmp/HP/train/8280
python /root/autodl-tmp/HP/run.py --train-dir /root/autodl-tmp/HP/train/9864
```
> **或者** 先赋予脚本执行权限后，直接运行项目脚本：
>
> ```bash
> chmod +x /root/autodl-tmp/HP/run.sh
> /root/autodl-tmp/HP/run.sh
> ```
>

## 6. 评估

训练完成后可按项目内评估脚本进行效果评估，结果默认保存在 `HP/eval_results/` 目录。

## 7. 生成回归训练集 (`overall_scores.csv`)

```bash
python fetch.py \
  --results_dir       /root/autodl-tmp/HP/eval_results \
  --output_path       /root/autodl-tmp/HP/train/overall_scores.csv \
  --feature_counts_dir /root/autodl-tmp/data/output/helpsteer2/counts \
  --experiments_file  /root/autodl-tmp/data/output/helpsteer2/experiments.txt
```

## 8. 训练 PPM 回归模型（使用 AllenAI `hybrid-preferences`）

首先克隆项目并进入目录：

```bash
git clone https://github.com/allenai/hybrid-preferences.git
cd hybrid-preferences
```

然后执行训练脚本：

```bash
python scripts/train_regressor.py \
  --input_path /root/autodl-tmp/HP/train/overall_scores.csv \
  --output_dir /root/autodl-tmp/HP/regressor/quadratic \
  --model      quadratic
```

---

### 附注

* 所有脚本中的路径均已硬编码；若目录结构不同，请相应修改脚本。
* 运行环境建议：Python ≥ 3.10，CUDA ≥ 11.8，确保已安装所需依赖（可参考各仓库 `requirements.txt`）。
* 如需分布式训练或自定义超参数，请参考官方文档并在 `run.py` 中进行调整。

