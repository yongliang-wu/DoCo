<h2 align="center">Unlearning Concepts in Diffusion Model via Concept Domain Correction and Concept Preserving Gradient</h2>

## News

* :fire: [2024.12.10] Our paper is accepted by AAAI-2025 !

## Introduction
This repository contains the PyTorch implementation for the paper [Unlearning Concepts in Diffusion Model via Concept Domain Correction and Concept Preserving Gradient](https://arxiv.org/abs/2405.15304).

In this paper, we propose DoCo (Domain Correction), a novel concept domain correction framework for machine unlearning in text-to-image diffusion models. Our method addresses two major challenges in existing machine unlearning approaches: limited generalization and utility degradation. DoCo achieves comprehensive concept unlearning through:

1. A domain correction mechanism that aligns the output domains of sensitive and anchor concepts via adversarial training, ensuring effective unlearning across both seen and unseen prompts.

2. A concept-preserving gradient surgery technique that mitigates conflicting gradient components, maintaining the model's overall utility while selectively removing targeted concepts.

Our experiments demonstrate superior performance in unlearning various types of concepts (instances, styles, and offensive content) while preserving model functionality, even for out-of-distribution prompts.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/3b3fd95c-7b2d-4843-bf5b-4d8fe6498718">

## Getting Started
```
git clone git@github.com:yongliang-wu/DoCo.git
cd DoCo
conda create -n DoCo python=3.10
conda activate DoCo
pip install -r requirements.txt
```

### Training
Before training, please replace the `site-packages/diffusers/schedulers/scheduling_ddpm.py` file with the `DoCo/scheduling_ddpm.py` file provided in this repository. Our version includes an additional `step_batch` function and modifications to the `_get_variance` function to enable batch processing.

**Note:** To achieve the best results, set different hyperparameters such as anchor concept, max_train_steps, warm_up, and learning_rate. These hyperparameters may vary depending on distinct concepts.

**Unlearning Style**

Setup accelerate config and pretrained model and then launch training. 

```
accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="logs_ablation/vangogh"

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=$OUTPUT_DIR \
          --class_data_dir=./data/samples_painting/ \
          --class_prompt="painting"  \
          --caption_target "van gogh" \
          --concept_type style \
          --resolution=512  \
          --train_batch_size=8 \
          --learning_rate=6e-6  \
          --max_train_steps=2000 \
          --scale_lr --hflip --noaug \
          --parameter_group cross-attn \
          --allow_tf32 \
          --enable_xformers_memory_efficient_attention \
          --warm_up=1000 \
          --with_prior_preservation \
          --lambda_ 1 \
          --gradient_clip \
          --dlr 0.0005
```


**Unlearning Instance**
```
accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OPENAI_API_KEY="provide-your-openai-api-key"
export OUTPUT_DIR="logs_ablation/parachute"

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=$OUTPUT_DIR \
          --class_data_dir=./data/samples_airplane/ \
          --class_prompt="airplane" \
          --caption_target "airplane+parachute" \
          --concept_type object \
          --resolution=512  \
          --train_batch_size=8  \
          --learning_rate=6e-6  \
          --max_train_steps=2000 \
          --scale_lr --hflip \
          --parameter_group cross-attn \
          --enable_xformers_memory_efficient_attention \
          --warm_up=1000 \
          --with_prior_preservation \
          --dlr 0.0001 \
          --lambda_ 1 \
          --gradient_clip
```


**Unlearning Nudity**
```
accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OPENAI_API_KEY="provide-your-openai-api-key"
export OUTPUT_DIR="logs_ablation/nudity"

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=$OUTPUT_DIR \
          --class_data_dir=./data/samples_clothed/ \
          --class_prompt="clothed" \
          --caption_target "clothed+nudity" \
          --concept_type object \
          --resolution=512  \
          --train_batch_size=8  \
          --learning_rate=6e-6  \
          --max_train_steps=2000 \
          --scale_lr --hflip \
          --parameter_group cross-attn \
          --enable_xformers_memory_efficient_attention \
          --warm_up=1000 \
          --with_prior_preservation \
          --dlr 0.0001 \
          --lambda_ 1 \
          --gradient_clip
```

### Inference
You can download our checkpoint from this [Google Drive link](https://drive.google.com/drive/folders/1xPe4BDUa2Rn8jQ90-Onr4kq6Mr4Oip7f?usp=sharing).

```python
from DoCo.model_pipeline import CustomDiffusionPipeline
import torch

pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipe.load_model('logs_ablation/vangogh/delta.bin')
image = pipe("painting of a house in the style of van gogh", num_inference_steps=50, guidance_scale=6., eta=1.).images[0]

image.save("vangogh.png")
```

### Evaluation
Please refer to the GitHub repository of [SPM](https://github.com/Con6924/SPM).

### Acknowledgements
We extend our gratitude to the following repositories for their contributions and resources:

- [SPM](https://github.com/Con6924/SPM)
- [Concept Ablation](https://github.com/nupurkmr9/concept-ablation)
- [Erasing](https://github.com/rohitgandikota/erasing)

Their works have significantly contributed to the development of our work.

