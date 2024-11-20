# accelerate config
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