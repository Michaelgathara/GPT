#!/bin/bash
# run_pipeline.sh - Complete pipeline for Qwen2-0.5B fine-tuning on LLaMa Nemotron
# Optimized for H100 GPU with 100GB+ RAM and 800GB storage

set -e 

MODEL_NAME="Qwen/Qwen2-0.5B-Instruct"
DATA_DIR="./preprocessed_data"
OUTPUT_DIR="./output"
EVAL_DIR="./evaluation_results"
MAX_SAMPLES=1000000  # Set to empty string to use all data

mkdir -p $DATA_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $EVAL_DIR
mkdir -p logs

START_TIME=$(date +%s)
echo "===== Starting Qwen2-0.5B Fine-tuning Pipeline ====="
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "Using H100 GPU with optimized settings"

echo "===== Setting up environment ====="
uv pip install transformers datasets accelerate peft deepspeed wandb sentencepiece
uv pip install tensorboard pandas matplotlib seaborn

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "===== Starting data preprocessing ====="
PREPROCESS_START=$(date +%s)

python data_preprocessing.py \
    --model_name $MODEL_NAME \
    --output_dir $DATA_DIR \
    --max_seq_length 2048 \
    ${MAX_SAMPLES:+--max_samples_per_split $MAX_SAMPLES}

PREPROCESS_END=$(date +%s)
PREPROCESS_DURATION=$((PREPROCESS_END - PREPROCESS_START))
echo "Preprocessing completed in $(($PREPROCESS_DURATION / 60)) minutes and $(($PREPROCESS_DURATION % 60)) seconds"

echo "===== Starting model training ====="
TRAIN_START=$(date +%s)

python training_script.py \
    --model_name $MODEL_NAME \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --epochs 2 \
    --batch_size 32 \
    --grad_accum_steps 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_seq_length 2048 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --precision bf16 \
    --gradient_checkpointing \
    --use_deepspeed \
    --deepspeed_stage 3 \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --use_wandb

TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
echo "Training completed in $(($TRAIN_DURATION / 60)) minutes and $(($TRAIN_DURATION % 60)) seconds"

echo "===== Starting model evaluation ====="
EVAL_START=$(date +%s)

python evaluation_script.py \
    --base_model_name $MODEL_NAME \
    --fine_tuned_model_path "$OUTPUT_DIR/final_model" \
    --output_dir $EVAL_DIR \
    --precision bf16

EVAL_END=$(date +%s)
EVAL_DURATION=$((EVAL_END - EVAL_START))
echo "Evaluation completed in $(($EVAL_DURATION / 60)) minutes and $(($EVAL_DURATION % 60)) seconds"

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
echo "===== Pipeline completed ====="
echo "Total duration: $(($TOTAL_DURATION / 3600)) hours, $((($TOTAL_DURATION % 3600) / 60)) minutes, and $(($TOTAL_DURATION % 60)) seconds"
echo "Results saved to $OUTPUT_DIR and $EVAL_DIR"

echo "===== Generating summary report ====="
cat > summary_report.md << EOF
# Qwen2-0.5B Fine-tuning Summary Report
- **Date:** $(date)
- **Model:** $MODEL_NAME
- **Hardware:** H100 GPU
- **Dataset:** LLaMa Nemotron

## Pipeline Duration
- **Preprocessing:** $(($PREPROCESS_DURATION / 60)) minutes
- **Training:** $(($TRAIN_DURATION / 60)) minutes
- **Evaluation:** $(($EVAL_DURATION / 60)) minutes
- **Total:** $(($TOTAL_DURATION / 3600)) hours, $((($TOTAL_DURATION % 3600) / 60)) minutes

## Training Configuration
- LoRA fine-tuning (r=$8, alpha=$16)
- BF16 precision
- DeepSpeed ZeRO Stage 3
- Batch size: 32

## Results
See detailed evaluation results in \`$EVAL_DIR/evaluation_summary.json\`
EOF

echo "Summary report generated: summary_report.md"
echo "===== Pipeline complete ====="