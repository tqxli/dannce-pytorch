import dannce.engine.models_pytorch.loss as custom_losses
import dannce.engine.models_pytorch.metrics as custom_metrics
import pandas as pd

def prepare_batch(batch, device):
    volumes = batch[0].float().to(device)
    grids = batch[1].float().to(device) if batch[1] is not None else None
    targets = batch[2].float().to(device)
    auxs = batch[3].to(device) if batch[3] is not None else None
    
    return volumes, grids, targets, auxs

class LossHelper:
    def __init__(self, params):
        self.loss_params = params["loss"]
        self._get_losses()

    def _get_losses(self):
        self.loss_fcns = {}
        for name, loss_weight in self.loss_params.items():
            self.loss_fcns[name] = [getattr(custom_losses, name), loss_weight]
        
    def compute_loss(self, kpts_gt, kpts_pred):
        loss_dict = {}
        total_loss = 0
        for k, (lossfcn, loss_weight) in self.loss_fcns.items():
            loss_val = lossfcn(kpts_gt.clone(), kpts_pred.clone())
            total_loss += loss_weight * loss_val
            loss_dict[k] = loss_val.detach().clone().cpu().item()

        return total_loss, loss_dict

    @property
    def names(self):
        return list(self.loss_params.keys())

class MetricHelper:
    def __init__(self, params):
        self.metric_names = params["metric"]
        self._get_metrics()

    def _get_metrics(self):
        self.metrics = {}
        for met in self.metric_names:
            self.metrics[met] = getattr(custom_metrics, met)
        
    def evaluate(self, kpts_gt, kpts_pred):
        metric_dict = {}
        for met in self.metric_names:
            metric_dict[met] = self.metrics[met](kpts_gt, kpts_pred)

        return metric_dict

    @property
    def names(self):
        return self.metric_names


class MetricTracker:
    def __init__(self, *keys, writer=None, train=True):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.train = "/train" if train else "/valid"
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key + self.train, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)