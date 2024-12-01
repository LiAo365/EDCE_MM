import torch
import numpy as np
import wandb

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def set_grad(modulelist, value):
    for attention in modulelist:
        for module in attention:
            for param in module.parameters():
                param.requires_grad = value


def train(itr, dataset, args, model, optimizer, device):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    outputs = model(features, seq_len=seq_len, is_training=True, opt=args)

    if args.use_causal_intervention == 1:
        # atn mask
        v_atn = outputs["v_atn"].cpu().data.numpy()
        f_atn = outputs["f_atn"].cpu().data.numpy()
        # v_mask = (v_atn > 0.01).astype(np.int32)
        # f_mask = (f_atn > 0.01).astype(np.int32)
        # video_mask = v_mask == f_mask

        # abs diff mask
        v_mask = np.abs(v_atn - f_atn) / (v_atn + f_atn + 1e-8) > args.abs_atn_threshold
        video_mask = 1.0 - v_mask

        video_mask = torch.from_numpy(video_mask).float().to(device)
        features = features * video_mask
        outputs = model(features, seq_len=seq_len, is_training=True, opt=args)

        # outputs["attn"] = mask_outputs["attn"]
        # outputs["v_atn"] = mask_outputs["v_atn"]
        # outputs["f_atn"] = mask_outputs["f_atn"]

    total_loss, loss_dict = model.criterion(outputs, labels, seq_len=seq_len, device=device, opt=args, itr=itr, pairs_id=pairs_id, inputs=features)

    optimizer.zero_grad()
    # try:
    # set_grad(model.Attn.attentions, False)
    # extra_loss = loss_dict['seperate_loss']
    # extra_loss.backward(retain_graph=True)
    # set_grad(model.Attn.attentions, True)
    # except:
    #     set_grad(model.Attn.attentions, True)
    total_loss.backward()
    optimizer.step()

    if not args.without_wandb:
        if itr % 20 == 0 and itr != 0:
            wandb.log(loss_dict)

    return total_loss.data.cpu().numpy()
