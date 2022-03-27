from abc import abstractmethod
from turtle import forward
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dannce.engine.models.body_limb import SYMMETRY

##################################################################################################
# UTIL_FUNCTIONS
##################################################################################################

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
# LOSSES
##################################################################################################

class BaseLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    
    @abstractmethod
    def forward(kpts_gt, kpts_pred):
        return NotImplementedError

class L1Loss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_gt, kpts_pred):
        kpts_gt, kpts_pred, notnan = mask_nan(kpts_gt, kpts_pred)
        if notnan == 0:
            print("Found all NaN ground truth")
            return kpts_pred.new_zeros(())
        loss = F.l1_loss(kpts_gt, kpts_pred, reduction="sum") / notnan
        return self.loss_weight * loss

class TemporalLoss(BaseLoss):
    def __init__(self, temporal_chunk_size, method="l1", **kwargs):
        super().__init__(**kwargs)

        self.temporal_chunk_size = temporal_chunk_size
        assert method in ["l1", "l2"]
        self.method = method
    
    def forward(self, kpts_gt, kpts_pred):
        # [n_chunks, temporal_chunk_size-1, n_joints, 3]
        kpts_pred = kpts_pred.reshape(-1, self.temporal_chunk_size, *kpts_pred.shape[1:])
        diff = torch.diff(kpts_pred, dim=1)
        if self.method == 'l1':
            loss_temp = torch.abs(diff).mean()
        else:
            loss_temp = (diff**2).sum(-1).sqrt().mean()
        return self.loss_weight * loss_temp

class BodySymmetryLoss(BaseLoss):
    def __init__(self, animal="mouse22", **kwargs):
        super().__init__(**kwargs)

        self.animal = animal
        self.limbL = torch.as_tensor([limbs[0] for limbs in SYMMETRY[animal]])
        self.limbR = torch.as_tensor([limbs[1] for limbs in SYMMETRY[animal]])
    
    def forward(self, kpts_gt, kpts_pred):
        len_L = (torch.diff(kpts_pred[:, :, self.limbL], dim=-1)**2).sum(1).sqrt().mean()
        len_R = (torch.diff(kpts_pred[:, :, self.limbR], dim=-1)**2).sum(1).sqrt().mean()
        loss = (len_L - len_R).abs()
        return self.loss_weight * loss

class SeparationLoss(BaseLoss):
    def __init__(self, delta=5, **kwargs):
        super().__init__(**kwargs)

        self.delta = delta
    
    def forward(self, kpts_gt, kpts_pred):
        num_kpts = kpts_pred.shape[-1]

        t1 = kpts_pred.repeat(1, 1, num_kpts)
        t2 = kpts_pred.repeat(1, num_kpts, 1).reshape(t1.shape)

        lensqr = ((t1 - t2)**2).sum(1)
        sep = torch.maximum(self.delta - lensqr, 0.0).sum(1) / num_kpts**2

        return self.loss_weight * sep.mean()

class PairRepulsionLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, kpts_gt, kpts_pred):
        n_joints = kpts_pred.shape[-1]
        kpts_pred = kpts_pred.reshape(-1, 2, *kpts_pred.shape[1:]).permute(1, 0, 2, 3)

        # compute the distance between each joint of animal 1 and all joints of animal 2
        a1 = kpts_pred[0].repeat(1, 1, n_joints)    
        a2 = kpts_pred[1].repeat(1, n_joints, 1).reshape(a1.shape)

        diffsqr = (a1 - a2)**2
        dist = diffsqr.sum(1).sqrt() # [bs, n_joints^2]
        
        return self.loss_weight * dist.mean()

# wait to be implemented
# def silhouette_loss(kpts_gt, kpts_pred):
#     # y_true and y_pred will both have shape
#     # (n_batch, width, height, n_keypts)
#     reduce_axes = (1, 2, 3)
#     sil = torch.sum(kpts_pred * kpts_gt, dim=reduce_axes)
#     sil = -(sil+1e-12).log()
    
#     return sil.mean()

# def separation_loss(delta=10):
#     def _separation_loss(kpts_gt, kpts_pred):
#         """
#         Loss which penalizes 3D keypoint predictions being too close.
#         """
#         num_kpts = kpts_pred.shape[-1]

#         t1 = kpts_pred.repeat(1, 1, num_kpts)
#         t2 = kpts_pred.repeat(1, num_kpts, 1).reshape(t1.shape)

#         lensqr = ((t1 - t2)**2).sum(1)
#         sep = torch.maximum(delta - lensqr, 0.0).sum(1) / num_kpts**2

#         return sep.mean()
#     return _separation_loss