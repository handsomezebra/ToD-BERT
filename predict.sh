

# ./evaluation_pipeline.sh 0 bert bert-base-uncased save/BERT

CUDA_VISIBLE_DEVICES=0 python main.py \
    --my_model=multi_label_classifier \
    --dataset='["ehealth"]' \
    --task=dm --task_name=sysact --example_type=turn \
    --model_type=bert \
    --model_name_or_path=save/pretrain/bert-joint/checkpoint-160000 \
    --load_path=save/DA/ehealth_pretrained/run0/pytorch_model.bin \
    --data_path=/data/_working/dialog_datasets \
    --output_dir=save/DA/ehealth_pretrained \
    --batch_size=8 \
    --eval_batch_size=4 \
    --learning_rate=5e-5 \
    --eval_by_step=1000 \
    --usr_token=[USR] --sys_token=[SYS] \
    --earlystop=f1_weighted \
    --patience=3

