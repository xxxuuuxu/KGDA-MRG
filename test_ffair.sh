#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --batch_size 1 \
    --image_size 512 \
    --vocab_size 995 \
    --theta 0.4 \
    --gamma 0.4 \
    --beta 1.0 \
    --delta 0.02 \
    --align_w 0.01 \
    --dataset_name ffa_ir \
    --anno_path /root/autodl-tmp/KGDA-MRG/data/ffa-ir/1.1.0/report.json \
    --data_dir /root/autodl-tmp/KGDA-MRG/data/ffa-ir/1.1.0/FFAIR_1 \
    --mode test \
    --knowledge_prompt_path /root/autodl-tmp/KGDA-MRG/knowledge_path/knowledge_prompt_ffair.pkl \
    --test_path /root/autodl-tmp/KGDA-MRG/ffair_best_model.pt
