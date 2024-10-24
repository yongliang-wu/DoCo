# [Preprint] Unlearning Concepts in Diffusion Model via Concept Domain Correction and Concept Preserving Gradient
This repository contains the PyTorch implementation for the preprint paper [Unlearning Concepts in Diffusion Model via Concept Domain Correction and Concept Preserving Gradient](https://arxiv.org/abs/2405.15304).

<img width="700" alt="image" src="https://github.com/user-attachments/assets/3b3fd95c-7b2d-4843-bf5b-4d8fe6498718">

<img width="700" alt="image" src="https://github.com/user-attachments/assets/2b448503-5dd9-42c5-a731-509cbfef659f">

## Getting Started
```
conda create -n unlearn python=3.10
conda activate unlearn
pip install -r requirements.txt
```
Please refer to the repository [Concept Ablation](https://github.com/nupurkmr9/concept-ablation) for more details.

### Training

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
          --learning_rate=2e-6  \
          --max_train_steps=1000 \
          --scale_lr --hflip --noaug \
          --parameter_group cross-attn \
          --allow_tf32 \
          --enable_xformers_memory_efficient_attention \
          --warm_up=200 \
          --prior_loss_weight=0 \
          --with_prior_preservation \
          --lambda_ 1 \
          --gradient_clip \
          --dlr 0.0005
```


**Unlearning Instance**
```
accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OPENAI_API_KEY="provide-your-api-key"
export OUTPUT_DIR="logs_ablation/r2d2"

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=./logs_ablation/r2d2 \
          --class_data_dir=./data/samples_robot/ \
          --class_prompt="robot" \
          --caption_target "robot+r2d2" \
          --concept_type object \
          --resolution=512  \
          --train_batch_size=8  \
          --learning_rate=2e-6  \
          --max_train_steps=1000 \
          --scale_lr --hflip \
          --parameter_group cross-attn \
          --enable_xformers_memory_efficient_attention \
          --warm_up=200 \
          --prior_loss_weight=0 \
          --with_prior_preservation \
          --dlr 0.0001 \
          --lambda_ 0.5 \
          --gradient_clip
```


**Unlearning Nudity**
```
accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OPENAI_API_KEY="provide-your-api-key"
export OUTPUT_DIR="logs_ablation/nudity"

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=./logs_ablation/nudity \
          --class_data_dir=./data/samples_clothed/ \
          --class_prompt="clothed" \
          --caption_target "clothed+nudity" \
          --concept_type object \
          --resolution=512  \
          --train_batch_size=8  \
          --learning_rate=2e-6  \
          --max_train_steps=1000 \
          --scale_lr --hflip \
          --parameter_group cross-attn \
          --enable_xformers_memory_efficient_attention \
          --warm_up=200 \
          --prior_loss_weight=0 \
          --with_prior_preservation \
          --dlr 0.0001 \
          --lambda_ 0.5 \
          --gradient_clip
```

### Inference

```python
from model_pipeline import CustomDiffusionPipeline
import torch

pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipe.load_model('logs_ablation/vangogh/delta.bin')
image = pipe("painting of a house in the style of van gogh", num_inference_steps=50, guidance_scale=6., eta=1.).images[0]

image.save("vangogh.png")
```
### Checkpoint


### Evaluation
For further details, please refer to the GitHub repository of [SPM](https://github.com/Con6924/SPM).

### Acknowledgements
We extend our gratitude to the following repositories for their contributions and resources:

- [SPM](https://github.com/Con6924/SPM)
- [Concept Ablation](https://github.com/nupurkmr9/concept-ablation)
- [Erasing](https://github.com/rohitgandikota/erasing)

Their work has significantly contributed to the development of our work.

