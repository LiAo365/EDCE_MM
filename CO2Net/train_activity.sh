CUDA_VISIBLE_DEVICES=1 python main.py \
    --k 5 \
    --dataset-name ActivityNet1.2 \
    --path-dataset Here is your local dataset path \
    --num-class 100 \
    --use-model ANT_CO2 \
    --dataset AntSampleDataset \
    --lr 3e-5 \
    --max-seqlen 60 \
    --model-name ANT_CO2_3552 \
    --seed 3552 \
    --max-iter 22000 \
    --use_causal_intervention 1 \
    --abs_atn_threshold 0.35 \
    --use_consistence_loss 1 \
    --consistence_threshold 0.7 \
    --consistence_proposal_threshold 0.4 \
    --feat_level_loss_weight 1.5 \
    --attn_level_loss_weight 1.5
