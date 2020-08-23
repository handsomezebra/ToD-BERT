
CUDA_VISIBLE_DEVICES=0 python my_tod_pretraining.py \
    --task=usdl \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --output_dir=save/pretrain/bert-joint \
    --data_path=/data/_working/dialog_datasets \
    --do_train \
    --do_eval \
    --mlm \
    --do_lower_case \
    --evaluate_during_training \
    --save_steps=10000 --logging_steps=1000 \
    --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8 \
    --add_rs_loss \
    --patience=3
