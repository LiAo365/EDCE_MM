import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConsistenceLoss(nn.Module):
    def __init__(self, consistence_threshold, consistence_proposal_threshold, feat_level_loss_weight, attn_level_loss_weight) -> None:
        super().__init__()
        self.consistence_threshold = consistence_threshold
        self.consistence_proposal_threshold = consistence_proposal_threshold
        self.feat_level_loss_weight = feat_level_loss_weight
        self.attn_level_loss_weight = attn_level_loss_weight

    def _select_rep_feat_(self, outputs):
        attn = outputs["attn"]  # (B, N, 1)
        feat = outputs["feat"]  # (B, N, C)

        # select representative feature
        rep_idxs = attn > self.consistence_threshold  # (B, N, 1)
        rep_idxs = rep_idxs.squeeze(-1)  # (B, N)
        rep_idxs_list = [rep_idxs[i].nonzero().squeeze().tolist() for i in range(rep_idxs.size(0))]
        rep_feat_list = [feat[i][rep_idxs_list[i]] for i in range(feat.size(0))]

        rep_idx_feat = {}  # key: idx in current batch, value: (idex, feat)
        for idx in range(len(rep_idxs_list)):
            rep_idx_feat[idx] = (rep_idxs_list[idx], rep_feat_list[idx])

        return rep_idx_feat

    def _gene_proposal_by_attn_(self, outputs):
        attn = outputs["attn"]  # (B, N, 1)
        attn = attn.squeeze(-1)  # (B, N)
        proposals = {}  # key: idx in current batch, value: list of proposals (start, end)
        for idx in range(attn.size(0)):
            attn_i = attn[idx]
            attn_i = attn_i.detach().cpu().numpy()
            preds_bin = (attn_i > self.consistence_proposal_threshold).astype("int")
            vid_pred_diff = np.append(preds_bin, 0) - np.append(0, preds_bin)
            start_ids = np.where(vid_pred_diff == 1)[0]
            end_ids = np.where(vid_pred_diff == -1)[0]
            if len(start_ids) == 0 or len(end_ids) == 0:
                proposals[idx] = []
                continue
            for start, end in zip(start_ids, end_ids):
                if end < start:
                    continue
                proposals[idx] = proposals.get(idx, []) + [(start, end)]
        return proposals

    def _feat_level_consistence_loss_(self, outputs):
        """
        Consistence loss for feature-level
            average the feature of proposals and calculate the distance between the average feature and the representative features
        """
        rep_idx_feat = self._select_rep_feat_(outputs)
        proposals = self._gene_proposal_by_attn_(outputs)
        feats = outputs["feat"]  # (B, T, D)
        avg_proposal_feat = []
        avg_rep_feat = []
        for idx in range(feats.size(0)):
            proposal_list = proposals[idx]
            if len(proposal_list) == 0:
                continue
            for start, end in proposal_list:
                has_rep_feat = False
                cur_rep_feat = []
                # find all the representative features between start and end
                rep_idxs, rep_feats = rep_idx_feat[idx]
                if isinstance(rep_idxs, int):
                    rep_idxs = [rep_idxs]
                    rep_feats = [rep_feats]
                for _idx_, rep_idx in enumerate(rep_idxs):
                    if start <= rep_idx <= end:
                        cur_rep_feat.append(rep_feats[_idx_])
                        has_rep_feat = True
                if has_rep_feat:
                    avg_proposal_feat.append(feats[idx, start:end].mean(dim=0))
                    avg_rep_feat.append(torch.stack(cur_rep_feat).mean(dim=0))

        # calculate the distance between the average feature and the representative features
        loss = 0
        for i in range(len(avg_rep_feat)):
            # feature contrastive loss between proposal and representative feature
            loss += F.mse_loss(avg_proposal_feat[i], avg_rep_feat[i])

        if len(avg_proposal_feat) > 0:
            loss = loss / len(avg_proposal_feat)
        return loss

    def _attn_level_consistence_loss_(self, outputs):
        """
        Consistence loss for cas-level
            The cas of proposals should be consistent with the proposal attn average value
        """
        proposals = self._gene_proposal_by_attn_(outputs)
        attn = outputs["attn"]  # (B, T, 1)
        attn = attn.squeeze(-1)  # (B, T)
        loss = 0.0
        for idx in range(attn.size(0)):
            proposal_list = proposals[idx]
            video_loss = 0.0
            if len(proposal_list) == 0:
                continue
            for start, end in proposal_list:
                proposal_atn = attn[idx, start:end]
                proposal_avg = proposal_atn.mean()
                video_loss += F.mse_loss(proposal_atn, proposal_avg * torch.ones_like(proposal_atn, device=attn.device))
            if len(proposal_list) > 0:
                video_loss = video_loss / len(proposal_list)
            loss += video_loss
        if len(proposals) > 0:
            loss = loss / len(proposals)
        return loss

    def forward(self, outputs):

        feat_level_loss = self._feat_level_consistence_loss_(outputs)
        attn_level_loss = self._attn_level_consistence_loss_(outputs)

        total_loss = self.feat_level_loss_weight * feat_level_loss + self.attn_level_loss_weight * attn_level_loss
        return total_loss
