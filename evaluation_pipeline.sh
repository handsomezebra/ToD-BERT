


CUDA_VISIBLE_DEVICES=0 python main.py \
    --my_model=dual_encoder_ranking \
    --do_train --dataset='["ehealth"]' \
    --task=nlg --task_name=rs --example_type=turn \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --data_path=/data/_working/ehealth_dialog/data \
    --output_dir=save/RS/ehealth_base/ \
    --batch_size=8 --eval_batch_size=8 \
    --usr_token=[USR] --sys_token=[SYS] \
    --fix_rand_seed \
    --eval_by_step=1000

exit 0

CUDA_VISIBLE_DEVICES=0 python main.py \
    --my_model=multi_label_classifier \
    --do_train --dataset='["ehealth"]' \
    --task=dm --task_name=sysact --example_type=turn \
    --model_type=bert \
    --model_name_or_path=save/pretrain/bert-joint/checkpoint-160000 \
    --data_path=/data/_working/dialog_datasets \
    --output_dir=save/DA/ehealth_pretrained \
    --batch_size=8 \
    --eval_batch_size=4 \
    --learning_rate=5e-5 \
    --eval_by_step=1000 \
    --usr_token=[USR] --sys_token=[SYS] \
    --earlystop=f1_weighted \
    --patience=3

