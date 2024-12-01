'''
Author: liyuanshuo
Email: liyuanshuo123@126.com
Date: 2024-03-28 17:17:11
Environment: VSCode with encoding UTF-8
LastEditTime: 2024-03-29 09:46:28
LastEditors: liyuanshuo
Description: Please follow the rules to write the description
'''
import torch
import numpy as np
import wandb

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def train(itr, dataset, args, model, optimizer, device):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    outputs = model(features, seq_len=seq_len, is_training=True, itr=itr, opt=args)

    # here we try use the relative threshold
    if args.use_causal_intervention == 1:
        v_atn = outputs["v_atn"].cpu().data.numpy()
        f_atn = outputs["f_atn"].cpu().data.numpy()
        video_mask = np.abs(v_atn - f_atn) / (v_atn + f_atn + 1e-5) > args.abs_atn_threshold
        video_mask = 1 - video_mask
        video_mask = torch.from_numpy(video_mask).float().to(device)
        features = features * video_mask

        outputs = model(features, seq_len=seq_len, is_training=True, opt=args)

    total_loss, loss_dict = model.criterion(outputs, labels, seq_len=seq_len, device=device, opt=args, itr=itr, pairs_id=pairs_id, inputs=features)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if not args.without_wandb:
        if itr % 20 == 0 and itr != 0:
            wandb.log(loss_dict)

    return total_loss.data.cpu().numpy()
