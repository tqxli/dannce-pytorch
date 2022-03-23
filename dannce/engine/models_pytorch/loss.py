"""
Partially adapted from KeypointNet implementation
https://github.com/tensorflow/models/tree/archive/research/keypointnet
"""

import torch 
import torch.nn.functional as F
from dannce.engine.models_pytorch.body_limb import SYMMETRY

def separation_loss(keypoints_3d, delta):
    """
    Penalize a pair of 3D keypoints if they are closer than a preset threshold delta.
    Inputs:
        keypoints_3d: Tensor [B, N, 3]
        delta: scalar
    """
    bs, num_kpts = keypoints_3d.shape[0], keypoints_3d.shape[1]

    t1 = keypoints_3d.repeat(1, num_kpts, 1)
    t2 = keypoints_3d.repeat(1, 1, num_kpts).reshape(t1.shape)

    diff = (t1 - t2)**2

    dist = torch.sum(diff, dim=2) # [B, N**2]

    return torch.sum(torch.maximum(delta-dist, 0)) / (num_kpts * bs *2)

def sillhouette_loss(mask, heatmaps):
    """
    Loss for the keypoints to fall inside the 2D mask
    Inputs:
        mask: [B, H, W] binary
        heatmaps: [B, n_joints, H, W]
    """
    mask_dist = F.softmax(mask.flatten()).reshape(mask.shape)
    mask_dist = mask_dist.unsqueeze(1)

    sill = torch.sum(mask_dist * heatmaps, dim=[2, 3])
    sill = torch.sum(-torch.log(sill + 1e-12))

    return sill

def mask_nan(kpts_gt, kpts_pred):
    # import pdb; pdb.set_trace()
    nan_gt = torch.isnan(kpts_gt)
    not_nan = (~nan_gt).sum()
    kpts_gt[nan_gt] = 0 
    kpts_pred[nan_gt] = 0
    return kpts_gt, kpts_pred, not_nan

def compute_mask_nan_loss(loss_fcn, kpts_gt, kpts_pred):
    kpts_gt, kpts_pred, notnan = mask_nan(kpts_gt, kpts_pred)
    return loss_fcn(kpts_gt, kpts_pred) / notnan

def l1_loss(kpts_gt, kpts_pred):
    return F.l1_loss(kpts_gt, kpts_pred, reduction="sum")

def temporal_loss(kpts_pred):
    """
    Unsupervised temporal consistency loss.
    kpts_pred: reshaped tensor [n_chunks, temporal_chunk_size, n_joints, 3]
    """
    diff = torch.diff(kpts_pred, dim=1) # [temporal_chunk_size-1, n_joints, 3]
    loss_temp = torch.abs(diff).mean()
    return loss_temp

def body_symmetry_loss(animal):
    limbL = torch.as_tensor([limbs[0] for limbs in SYMMETRY[animal]])
    limbR = torch.as_tensor([limbs[1] for limbs in SYMMETRY[animal]])
    def _body_symmetry_loss(kpts_pred):
        """
        Unsupervised body symmetry loss.
        kpts_pred: tensor [batch_size, n_joints, 3]
        """
        len_L = torch.diff(kpts_pred[:, limbL, :] ** 2, dim=2).mean(-1).mean(-1).sqrt()
        len_R = torch.diff(kpts_pred[:, limbR, :] ** 2, dim=2).mean(-1).mean(-1).sqrt()
        loss = (len_L - len_R).abs().mean()
        return loss
    return _body_symmetry_loss