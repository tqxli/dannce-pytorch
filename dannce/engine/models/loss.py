from abc import abstractmethod
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dannce.engine.models.body_limb import SYMMETRY
from dannce.engine.models.vis import draw_voxels

##################################################################################################
# UTIL_FUNCTIONS
##################################################################################################

# def mask_nan(kpts_gt, kpts_pred):
#     nan_gt = torch.isnan(kpts_gt)
#     not_nan = (~nan_gt).sum()
#     kpts_gt[nan_gt] = 0 
#     kpts_pred[nan_gt] = 0
#     return kpts_gt, kpts_pred, not_nan

def compute_mask_nan_loss(loss_fcn, kpts_gt, kpts_pred):
    # kpts_gt, kpts_pred, notnan = mask_nan(kpts_gt, kpts_pred)
    notnan_gt = ~torch.isnan(kpts_gt)
    notnan = notnan_gt.sum()
    # when ground truth is all NaN for certain reasons, do not compute loss since it results in NaN
    if notnan == 0:
        # print("Found all NaN ground truth")
        return kpts_pred.new_zeros((), requires_grad=False)
    
    return loss_fcn(kpts_gt[notnan_gt], kpts_pred[notnan_gt]) / notnan

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

class L2Loss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_gt, kpts_pred):
        loss = compute_mask_nan_loss(nn.MSELoss(reduction="sum"), kpts_gt, kpts_pred)
        return self.loss_weight * loss

class L1Loss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_gt, kpts_pred):
        loss = compute_mask_nan_loss(nn.L1Loss(reduction="sum"), kpts_gt, kpts_pred)
        return self.loss_weight * loss

class TemporalLoss(BaseLoss):
    def __init__(self, temporal_chunk_size, method="l1", **kwargs):
        super().__init__(**kwargs)

        self.temporal_chunk_size = temporal_chunk_size
        assert method in ["l1", "l2"]
        self.method = method
    
    def forward(self, kpts_gt, kpts_pred):
        # reshape w.r.t temporal chunk size
        kpts_pred = kpts_pred.reshape(-1, self.temporal_chunk_size, *kpts_pred.shape[1:])
        diff = torch.diff(kpts_pred, dim=1)
        if self.method == 'l1':
            loss_temp = torch.abs(diff).mean()
        else:
            loss_temp = (diff**2).sum(1).sqrt().mean()
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

class GaussianRegLoss(BaseLoss):
    def __init__(self, method="mse", sigma=5, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.sigma = sigma
    
    def visualize(self, target):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        draw_voxels(target.clone().detach().cpu(), ax)
        plt.show(block=True)
        input("Press Enter to continue...")

    def _generate_gaussian_target(self, centers, grids):
        """
        centers: [batch_size, 3, n_joints]
        grid: [batch_size, 3, h, w, d]
        """
        y_3d = grids.new_zeros(centers.shape[0], centers.shape[2], *grids.shape[2:]) #[bs, n_joints, n_vox, n_vox, n_vox]
        for i in range(y_3d.shape[0]):
            for j in range(y_3d.shape[1]):
                y_3d[i, j] = torch.exp(-((grids[i, 1] - centers[i, 1, j])**2 
                            + (grids[i, 0] - centers[i, 0, j])**2
                            + (grids[i, 2] - centers[i, 2, j])**2))
                y_3d[i, j] /= (2 * self.sigma**2)
                
        return y_3d

    def forward(self, kpts_gt, kpts_pred, heatmaps, grids):
        """
        kpts_pred: [bs, 3, n_joints]
        heatmaps: [bs, n_joints, h, w, d]
        grid_centers: [bs, h*w*d, 3]
        """
        # reshape grids
        grids = grids.permute(0, 2, 1).reshape(grids.shape[0], grids.shape[2], *heatmaps.shape[2:]) # [bs, 3, n_vox, n_vox, n_vox]

        # generate gaussian shaped targets based on current predictions
        gaussian_gt = self._generate_gaussian_target(kpts_pred.clone().detach(), grids) #[bs, n_joints, n_vox**3]

        # apply sigmoid to the exposed heatmap
        heatmaps = torch.sigmoid(heatmaps)

        # compute loss
        if self.method == "mse":
            loss = F.mse_loss(gaussian_gt, heatmaps)
        elif self.method == "cross_entropy":
            loss = - (gaussian_gt * heatmaps.log()).mean()

        return self.loss_weight * loss

class PairRepulsionLoss(BaseLoss):
    def __init__(self, delta=5, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
    
    def forward(self, kpts_gt, kpts_pred):
        """
        [bs, 3, n_joints]
        """
        bs, n_joints = kpts_pred.shape[0], kpts_pred.shape[-1]
        
        kpts_pred = kpts_pred.reshape(2, -1, *kpts_pred.shape[1:]) # [n_pairs, 2, 3, n_joints]

        # compute the distance between each joint of animal 1 and all joints of animal 2
        a1 = kpts_pred[0].repeat(1, 1, n_joints) # [n, 3, n_joints^2]
        a2 = kpts_pred[1].repeat(1, n_joints, 1).reshape(a1.shape)

        diffsqr = (a1 - a2)**2
        dist = diffsqr.sum(1).sqrt() # [bs, n_joints^2] 
        dist = torch.maximum(self.delta - dist, torch.zeros_like(dist))
        
        return self.loss_weight * (dist.sum() / bs)


class SilhouetteLoss(BaseLoss):
    def __init__(self, delta=5, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
    
    def forward(self, y_true, y_pred):
        """
        [bs, H, W, D, n_joints]
        """
        reduce_axes = [1, 2, 3]
        sil = torch.sum(y_pred * y_true, axis=reduce_axes)
        sil = torch.mean(-(sil + 1e-12).log())
        if torch.isnan(sil):
            sil = sil.new_zeros(())
        
        return self.loss_weight * sil


