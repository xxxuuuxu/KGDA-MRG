export CUDA_VISIBLE_DEVICES=0

python main.py \
    --dataset_name ffa_ir \
    --anno_path /root/autodl-tmp/KGDA-MRG/data/ffa-ir/1.1.0/report.json \
    --data_dir /root/autodl-tmp/KGDA-MRG/data/ffa-ir/1.1.0/FFAIR_1 \
    --epochs 50 \
    --lr_backbone 1e-5 \
    --lr 1e-4 \
    --batch_size 2 \
    --image_size 512 \
    --vocab_size 995 \
    --theta 0.4 \
    --gamma 0.4 \
    --beta 1.0 \
    --delta 0.02 \
    --align_w 0.01 \
    --t_model_weight_path "" \
    --mode train