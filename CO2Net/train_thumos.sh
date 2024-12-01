CUDA_VISIBLE_DEVICES=1 python main.py \
    --max-seqlen 500 \
    --lr 0.00005 \
    --k 7 \
    --dataset-name Thumos14reduced \
    --path-dataset Here is your local dataset path \
    --num-class 20 \
    --use-model CO2 \
    --max-iter 7500 \
    --dataset SampleDataset \
    --weight_decay 0.001 \
    --model-name CO2_3552 \
    --seed 3552 \
    --AWM BWA_fusion_dropout_feat_v2 \
    --use_causal_intervention 1 \
    --abs_atn_threshold 0.7 \
    --lambd 1.0 \
    --use_consistence_loss 1 \
    --consistence_threshold 0.8 \
    --consistence_proposal_threshold 0.55 \
    --feat_level_loss_weight 0.5 \
    --attn_level_loss_weight 0.5
