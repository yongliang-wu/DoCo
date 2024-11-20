# accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OPENAI_API_KEY="your-openai-api-key"
export OUTPUT_DIR="logs_ablation/church"

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=./logs_ablation/church \
          --class_data_dir=./data/samples_house/ \
          --class_prompt="house" \
          --caption_target "house+church" \
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