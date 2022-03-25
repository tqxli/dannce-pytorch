"""
Partially adapted from KeypointNet implementation
https://github.com/tensorflow/models/tree/archive/research/keypointnet
"""

import torch 
import torch.nn.functional as F
from dannce.engine.models_pytorch.body_limb import SYMMETRY

def mask_nan(kpts_gt, kpts_pred):
    nan_gt = torch.isnan(kpts_gt)
    not_nan = (~nan_gt).sum()
    kpts_gt[nan_gt] = 0 
    kpts_pred[nan_gt] = 0
    return kpts_gt, kpts_pred, not_nan

def compute_mask_nan_loss(loss_fcn, kpts_gt, kpts_pred):
    kpts_gt, kpts_pred, notnan = mask_nan(kpts_gt, kpts_pred)
    # when ground truth is all NaN for certain reasons, do not compute loss since it results in NaN
    if notnan == 0:
        print("Found all NaN ground truth")
        return kpts_pred.new_zeros(())
    return loss_fcn(kpts_gt, kpts_pred) / notnan

##################################################################################################
# SINGLE ANIMAL
##################################################################################################

def l1_loss(kpts_gt, kpts_pred):
    kpts_gt, kpts_pred, notnan = mask_nan(kpts_gt, kpts_pred)
    # when ground truth is all NaN for certain reasons, do not compute loss since it results in NaN
    if notnan == 0:
        print("Found all NaN ground truth")
        return kpts_pred.new_zeros(())
    return F.l1_loss(kpts_gt, kpts_pred, reduction="sum") / notnan

def temporal_loss(kpts_gt, kpts_pred):
    """
    Unsupervised temporal consistency loss.
    kpts_pred: reshaped tensor [n_chunks, temporal_chunk_size, 3, n_joints]
    """
    diff = torch.diff(kpts_pred, dim=1) # [n_chunks, temporal_chunk_size-1, n_joints, 3]
    loss_temp = torch.abs(diff).mean()

    return loss_temp

# def body_symmetry_loss(animal="mouse22"):
#     limbL = torch.as_tensor([limbs[0] for limbs in SYMMETRY[animal]])
#     limbR = torch.as_tensor([limbs[1] for limbs in SYMMETRY[animal]])
#     def _body_symmetry_loss(kpts_gt, kpts_pred):
#         """
#         Unsupervised body symmetry loss.
#         kpts_pred: tensor [batch_size, n_joints, 3]
#         """
#         len_L = torch.diff(kpts_pred[:, limbL, :] ** 2, dim=2).mean(-1).mean(-1).sqrt()
#         len_R = torch.diff(kpts_pred[:, limbR, :] ** 2, dim=2).mean(-1).mean(-1).sqrt()
#         loss = (len_L - len_R).abs().mean()
#         return loss
#     return _body_symmetry_loss
animal = "mouse22"
limbL = torch.as_tensor([limbs[0] for limbs in SYMMETRY[animal]])
limbR = torch.as_tensor([limbs[1] for limbs in SYMMETRY[animal]])

def body_symmetry_loss(kpts_gt, kpts_pred):
    """
    Unsupervised body symmetry loss.
    kpts_pred: tensor [batch_size, 3, n_joints]
    """
    len_L = (torch.diff(kpts_pred[:, :, limbL], dim=-1)**2).sum(1).sqrt().mean()
    len_R = (torch.diff(kpts_pred[:, :, limbR], dim=-1)**2).sum(1).sqrt().mean()
    loss = (len_L - len_R).abs()
    return loss

def silhouette_loss(kpts_gt, kpts_pred):
    # y_true and y_pred will both have shape
    # (n_batch, width, height, n_keypts)
    reduce_axes = (1, 2, 3)
    sil = torch.sum(kpts_pred * kpts_gt, dim=reduce_axes)
    sil = -(sil+1e-12).log()
    
    return sil.mean()

def separation_loss(delta=10):
    def _separation_loss(kpts_gt, kpts_pred):
        """
        Loss which penalizes 3D keypoint predictions being too close.
        """
        num_kpts = kpts_pred.shape[-1]

        t1 = kpts_pred.repeat(1, 1, num_kpts)
        t2 = kpts_pred.repeat(1, num_kpts, 1).reshape(t1.shape)

        lensqr = ((t1 - t2)**2).sum(1)
        sep = torch.maximum(delta - lensqr, 0.0).sum(1) / num_kpts**2

        return sep.mean()
    return _separation_loss

##################################################################################################
# SOCIAL ANIMALS
##################################################################################################

def pair_repulsion_loss(y_pred_s1, y_pred_s2):
    """Unsupervised pairwise loss with respect to two subjects. 
    The predictions should be as far as possible, i.e. repelling each other.
    Input:
        y_pred_s1, y_pred_s2: (B, N, 3)"""

    return 1 / ((y_pred_s1 - y_pred_s2)**2).sum()