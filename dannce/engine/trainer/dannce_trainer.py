from dannce.engine.trainer.base_trainer import BaseTrainer
from dannce.engine.trainer.train_utils import prepare_batch, LossHelper, MetricHelper, MetricTracker
import torch

class DannceTrainer(BaseTrainer):
    def __init__(self, device, train_dataloader, valid_dataloader, lr_scheduler=None, **kwargs):
        super().__init__(**kwargs)
        self.loss = LossHelper(self.params)
        self.metrics = MetricHelper(self.params)
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler

        self.train_loss_tracker = MetricTracker(*self.loss.names, writer=self.writer, train=True)
        self.valid_loss_tracker = MetricTracker(*self.loss.names, writer=self.writer, train=False)
        self.valid_metric_tracker = MetricTracker(*self.metrics.names, writer=self.writer, train=False)

    def _train_epoch(self, epoch):
        self.model.train()
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss_dict = {}
            for batch in self.train_dataloader: 
                volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)

                self.optimizer.zero_grad()
                keypoints_3d_pred, _ = self.model(volumes, grid_centers)
                total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred)
                result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val} " for loss, val in loss_dict.items())
                print(result, end='\r')

                total_loss.backward()
                self.optimizer.step()

                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            for k, v in epoch_loss_dict.items():
                self.train_loss_tracker.update(k, sum(v)/len(v))
            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {sum(val)/len(val)} " for loss, val in epoch_loss_dict.items())
            print(result)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return epoch_loss_dict
    
    def _update_step(self, epoch_dict, step_dict):
        if len(epoch_dict) == 0:
            for k, v in step_dict.items():
                epoch_dict[k] = [v]
        else:
            for k, v in step_dict.items():
                epoch_dict[k].append(v)
        return epoch_dict

    def _valid_epoch(self, epoch):
        self.model.eval()

        epoch_loss_dict = {}
        epoch_metric_dict = {}
        with torch.no_grad():
            for batch in self.valid_dataloader:
                volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)
                keypoints_3d_pred, _ = self.model(volumes, grid_centers)

                total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred)
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.cpu().numpy())
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        for k, v in epoch_loss_dict.items():
            self.valid_loss_tracker.update(k, sum(v)/len(v))
        for k, v in epoch_metric_dict.items():
            self.valid_metric_tracker.update(k, sum(v)/len(v))
        result = f"Epoch[{epoch}/{self.epochs}] " \
            + "".join(f"val_{loss}: {sum(val)/len(val)} " for loss, val in epoch_loss_dict.items()) \
            + "".join(f"val_{met}: {sum(val)/len(val)} " for met, val in epoch_metric_dict.items())
        print(result)
        
        return epoch_loss_dict, epoch_metric_dict
        


