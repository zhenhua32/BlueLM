LR=1e-5

# 获取脚本所在目录
SCRIPT_DIR=$(cd $(dirname $0); pwd)
# 根目录是脚本目录的上一级
ROOT_DIR=$(dirname $SCRIPT_DIR)
echo "root_dir is $ROOT_DIR"

OUTPUT_PATH=$ROOT_DIR/output/bluelm-7b-sft-lora
MODEL_PATH="/home/pretrain_model_dir/_modelscope/vivo-ai/BlueLM-7B-Chat"

# OUTPUT
MODEL_OUTPUT_PATH=$OUTPUT_PATH/model
LOG_OUTPUT_PATH=$OUTPUT_PATH/logs
TENSORBOARD_PATH=$OUTPUT_PATH/tensorboard

mkdir -p $MODEL_OUTPUT_PATH
mkdir -p $LOG_OUTPUT_PATH
mkdir -p $TENSORBOARD_PATH

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

echo "master port is $MASTER_PORT"

    # --deepspeed \
    # --train_file "./data/bella_train_demo.json" \

deepspeed --num_gpus=1 --master_port $MASTER_PORT main.py \
    --deepspeed \
    --train_file "./data/bella_dev_demo.json" \
    --prompt_column inputs \
    --response_column targets \
    --model_name_or_path $MODEL_PATH \
    --output_dir $MODEL_OUTPUT_PATH \
    --tensorboard_dir $TENSORBOARD_PATH \
    --seq_len 2048 \
    --batch_size_per_device 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --max_steps 9000 \
    --save_steps 1000 \
    --learning_rate $LR \
    --finetune \
    --lora_rank 8 \
    | tee $LOG_OUTPUT_PATH/training.log