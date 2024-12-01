import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import time

from tensorboard_logger import Logger

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def train(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    outputs = model(features, seq_len=seq_len, is_training=True, itr=itr, opt=args)

    if args.use_causal_intervention == 1:
        # add causal intervention
        v_atn = outputs["v_atn"].cpu().data.numpy()
        f_atn = outputs["f_atn"].cpu().data.numpy()
        # v_mask = (v_atn > 0.01).astype(np.int32)
        # f_mask = (f_atn > 0.01).astype(np.int32)
        # video_mask = v_mask == f_mask

        # abs diff mask
        # v_mask = np.abs(v_atn - f_atn) / np.abs(v_atn + f_atn + 1e-8) > args.abs_atn_threshold
        v_mask = np.abs(v_atn - f_atn) / np.abs(v_atn + f_atn + 1e-8) > args.abs_atn_threshold
        v_mask = v_mask * args.lambd
        video_mask = 1.0 - v_mask

        video_mask = torch.from_numpy(video_mask).float().to(device)
        features = features * video_mask
        outputs = model(features, seq_len=seq_len, is_training=True, opt=args)

    total_loss = model.criterion(
        outputs, labels, seq_len=seq_len, device=device, logger=logger, opt=args, itr=itr, pairs_id=pairs_id, inputs=features
    )
    # print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.data.cpu().numpy()
