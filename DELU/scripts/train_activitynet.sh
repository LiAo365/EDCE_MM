#!/usr/bin/env bash
cd ..

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name DELU_ACT \
    --seed 0 \
    --alpha_edl 1.2 \
    --alpha_uct_guide 0.2 \
    --amplitude 0.2 \
    --alpha2 0.8 \
    --rat_atn 5 \
    --k 5 \
    --interval 200 \
    --dataset_name ActivityNet1.2 \
    --path_dataset Here is your local dataset path \
    --num_class 100 \
    --use_model DELU_ACT \
    --dataset AntSampleDataset \
    --lr 3e-5 \
    --max_seqlen 60 \
    --max_iter 22000 \
    --use_causal_intervention 1 \
    --abs_atn_threshold 0.7 \
    --use_consistence_loss 1 \
    --consistence_threshold 0.9 \
    --consistence_proposal_threshold 0.7 \
    --feat_level_loss_weight 1.0 \
    --attn_level_loss_weight 1.0
