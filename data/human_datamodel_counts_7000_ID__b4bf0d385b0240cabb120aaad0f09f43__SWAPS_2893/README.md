---
library_name: peft
license: other
base_model: /root/autodl-tmp/model/tulu-2-dpo-7b
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: human_datamodel_counts_7000_ID__b4bf0d385b0240cabb120aaad0f09f43__SWAPS_2893
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# human_datamodel_counts_7000_ID__b4bf0d385b0240cabb120aaad0f09f43__SWAPS_2893

This model is a fine-tuned version of [/root/autodl-tmp/model/tulu-2-dpo-7b](https://huggingface.co//root/autodl-tmp/model/tulu-2-dpo-7b) on the human_datamodel_counts_7000_ID__b4bf0d385b0240cabb120aaad0f09f43__SWAPS_2893 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.15.1
- Transformers 4.51.3
- Pytorch 2.1.2+cu121
- Datasets 3.5.0
- Tokenizers 0.21.1