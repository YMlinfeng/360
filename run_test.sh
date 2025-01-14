export MODEL_DIR="/home/jovyan/liushanyuan-sh-ceph/model/cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/303875135fff3f05b6aa893d544f28833a237d58"
export OUTPUT_DIR="output"
export CUDA_VISIBLE_DEVICES=1
export NCCL_TIMEOUT=3600
# accelerate launch --config_file ./config_mg/gpu_64_deepspeed_config.yaml --machine_rank=$RANK --main_process_ip=$MASTER_ADDR train_dit_all_control.py \
# accelerate launch --config_file ./config_mg/gpu_8_deepspeed_config.yaml train_dit_all_control.py \
accelerate launch --config_file ./config_mg/gpu_1_deepspeed_config.yaml train_dit_all_control.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="bf16" \
    --weighting_scheme="logit_normal" \
    --resolution=256 \
    --learning_rate=1e-4 \
    --max_train_steps=100000000 \
    --checkpointing_steps=50 \
    --dataloader_num_workers=16 \
    --gradient_accumulation_steps=4 \
    --proportion_empty_prompts=0.1 \
    --seed=43 \
    --train_batch_size=1 \
    


