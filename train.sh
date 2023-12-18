## launch training script (2 GPUs recommended, increase --max_train_steps to 500 if 1 GPU)
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="/data/home/chensh/data/huggingface_model/stable-diffusion-xl-base-1.0"

export INSTANCE_DIR="./data/cat"
export INSTANCE_PROMPT="photo of a <new1> cat"
export CLASS_DIR="./sample_reg/samples_cat/"
export CLASS_PROMPT="cat"
export OUTPUT_DIR="./logs/cat"
export modifier_token="<new1>"

#export INSTANCE_DIR="./data/wooden_pot"
#export INSTANCE_PROMPT="photo of a <new2> wooden pot"
#export CLASS_DIR="./data/prior_woodenpot/"
#export CLASS_PROMPT="wooden pot"
#export OUTPUT_DIR="./logs/wooden_pot"
#export modifier_token="<new2>"

accelerate launch src/diffusers_training_sdxl.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DIR \
          --output_dir=$OUTPUT_DIR  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --instance_prompt="${INSTANCE_PROMPT}"  \
          --class_prompt="${CLASS_PROMPT}" \
          --resolution=1024  \
          --train_batch_size=1  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=1000 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token="${modifier_token}"

### sample
#python src/diffusers_sample.py --delta_ckpt logs/cat/delta.bin --ckpt "CompVis/stable-diffusion-v1-4" --prompt "<new1> cat playing with a ball"