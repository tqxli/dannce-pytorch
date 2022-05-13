import dannce.engine.models.loss as custom_losses
import dannce.engine.models.metrics as custom_metrics
import numpy as np
# import pandas as pd

def prepare_batch(batch, device):
    volumes = batch[0].float().to(device)
    grids = batch[1].float().to(device) if batch[1] is not None else None
    targets = batch[2].float().to(device)
    auxs = batch[3].to(device) if batch[3] is not None else None
    
    return volumes, grids, targets, auxs

class LossHelper:
    def __init__(self, params):
        self.loss_params = params
        self._get_losses()

    def _get_losses(self):
        self.loss_fcns = {}
        for name, args in self.loss_params["loss"].items():
            self.loss_fcns[name] = getattr(custom_losses, name)(**args)
        
    def compute_loss(self, kpts_gt, kpts_pred, heatmaps, grid_centers, aux):
        """
        Compute each loss and return their weighted sum for backprop.
        """
        loss_dict = {}
        total_loss = []
        for k, lossfcn in self.loss_fcns.items():
            if k == "GaussianRegLoss":
                loss_val = lossfcn(kpts_gt, kpts_pred, heatmaps, grid_centers)
            elif k == 'SilhouetteLoss' or k == 'ReconstructionLoss':
                loss_val = lossfcn(aux, heatmaps)
            elif k == 'VarianceLoss':
                loss_val = lossfcn(kpts_pred, heatmaps, grid_centers)
            else:
                loss_val = lossfcn(kpts_gt, kpts_pred)
            total_loss.append(loss_val)
            loss_dict[k] = loss_val.detach().clone().cpu().item()

        return sum(total_loss), loss_dict

    @property
    def names(self):
        return list(self.loss_fcns.keys())

class MetricHelper:
    def __init__(self, params):
        self.metric_names = params["metric"]
        self._get_metrics()

    def _get_metrics(self):
        self.metrics = {}
        for met in self.metric_names:
            self.metrics[met] = getattr(custom_metrics, met)
        
    def evaluate(self, kpts_gt, kpts_pred):
        # perform NaN masking ONCE before metric computation
        kpts_pred, kpts_gt = self.mask_nan(kpts_pred, kpts_gt)
        metric_dict = {}
        for met in self.metric_names:
            metric_dict[met] = self.metrics[met](kpts_pred, kpts_gt)

        return metric_dict

    @property
    def names(self):
        return self.metric_names
    
    @classmethod
    def mask_nan(self, pred, gt):
        """
        pred, gt: [bs, 3, n_joints]
        """
        pred = np.transpose(pred.copy(), (1, 0, 2))
        gt = np.transpose(gt.copy(), (1, 0, 2)) #[3, bs, n_joints]
        pred = np.reshape(pred, (3, -1))
        gt = np.reshape(gt, (3, -1)) #[3, bs*n_joints]

        gi = np.where(~np.isnan(np.sum(gt, axis=0)))[0] #[bs*n_joints]

        pred = pred[:, gi]
        gt = gt[:, gi] #[3, bs*n_joints]

        return pred, gt


# class MetricTracker:
#     def __init__(self, *keys, writer=None, train=True):
#         self.writer = writer
#         self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
#         self.train = "/train" if train else "/valid"
#         self.reset()

#     def reset(self):
#         for col in self._data.columns:
#             self._data[col].values[:] = 0

#     def update(self, key, value, n=1):
#         if self.writer is not None:
#             self.writer.add_scalar(key + self.train, value)
#         self._data.total[key] += value * n
#         self._data.counts[key] += n
#         self._data.average[key] = self._data.total[key] / self._data.counts[key]

#     def avg(self, key):
#         return self._data.average[key]

#     def result(self):
#         return dict(self._data.average)