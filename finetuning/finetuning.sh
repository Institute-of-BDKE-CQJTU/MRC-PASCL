export MODEL="./pretrained_model/splinter"
export OUTPUT_DIR="output0"
export CUDA_VISIBLE_DEVICES=0
python /home/nlp/NLP-Group/XQ/splinter++/splinter/finetuning/run_mrqa_bioasq.py \
    --model_type=bert \
    --model_name_or_path=$MODEL \
    --qass_head=True \
    --tokenizer_name=$MODEL \
    --output_dir=$OUTPUT_DIR \
    --train_file="./data/mrqa-few-shot/bioasq/bioasq-train-seed-42-num-examples-16_qass.jsonl" \
    --predict_file="./data/mrqa-few-shot/bioasq/dev_qass.jsonl" \
    --do_train \
    --do_eval \
    --max_seq_length=384 \
    --doc_stride=128 \
    --threads=4 \
    --cl_span_loss=True \
    --save_steps=50000 \
    --per_gpu_train_batch_size=12 \
    --per_gpu_eval_batch_size=16 \
    --learning_rate=3e-5 \
    --max_answer_length=10 \
    --warmup_ratio=0.1 \
    --min_steps=200 \
    --num_train_epochs=10 \
    --seed=42 \
    --use_cache=False \
    --load_pytorch_pretrained_model="PiSCo model's path" \
    --evaluate_every_epoch=False \
    --overwrite_output_dir 