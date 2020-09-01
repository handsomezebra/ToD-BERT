


CUDA_VISIBLE_DEVICES=0 python main.py \
    --my_model=dual_encoder_ranking \
    --do_train --dataset='["ehealth"]' \
    --task=nlg --task_name=rs --example_type=turn \
    --model_type=bert \
    --model_name_or_path=save/pretrain/ehealth-bert-joint-3/checkpoint-70000 \
    --data_path=/data/_working/ehealth_dialog/data \
    --output_dir=save/RS/ehealth_pretrained_1/ \
    --batch_size=8 --eval_batch_size=8 \
    --nb_neg_sample_rs=4 \
    --usr_token=[USR] --sys_token=[SYS] \
    --fix_rand_seed \
    --eval_by_step=1000
