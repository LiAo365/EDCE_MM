cd ..

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name DELU \
    --seed 0 \
    --alpha_edl 1.3 \
    --alpha_uct_guide 0.4 \
    --amplitude 0.7 \
    --alpha2 0.4 \
    --interval 50 \
    --max_seqlen 320 \
    --lr 0.00005 \
    --k 7 \
    --dataset_name Thumos14reduced \
    --path_dataset Here is your local dataset path \
    --num_class 20 \
    --use_model DELU \
    --max_iter 7500 \
    --dataset SampleDataset \
    --weight_decay 0.001 \
    --AWM BWA_fusion_dropout_feat_v2 \
    --use_causal_intervention 1 \
    --abs_atn_threshold 0.85 \
    --use_consistence_loss 0 \
    --consistence_threshold 0.8 \
    --consistence_proposal_threshold 0.6 \
    --feat_level_loss_weight 0.3 \
    --attn_level_loss_weight 0.3
