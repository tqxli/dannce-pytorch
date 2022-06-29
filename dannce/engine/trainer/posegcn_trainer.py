import torch
import os, csv
from tqdm import tqdm

from dannce.engine.trainer.dannce_trainer import DannceTrainer
from dannce.engine.trainer.train_utils import prepare_batch
import dannce.engine.models.loss as custom_losses

class GCNTrainer(DannceTrainer):
    def __init__(self, predict_diff=True, **kwargs):
        super().__init__(**kwargs)

        self.predict_diff = predict_diff
        if self.predict_diff:
            self.loss_sup = custom_losses.L1Loss()
            self.loss.loss_fcns.pop("L1Loss")

            self._del_loss_attr(["L1Loss"])
            self._add_loss_attr(["L1DiffLoss"])
    
    def _rewrite_csv(self):
        stats_file = open(os.path.join(self.params["dannce_train_dir"], "training.csv"), 'w', newline='')
        stats_writer = csv.writer(stats_file)
        stats_writer.writerow(["Epoch", *self.train_stats_keys, *self.valid_stats_keys])
        stats_file.close()
    
    def _add_loss_attr(self, names):
        self.stats_keys = names + self.stats_keys
        self.train_stats_keys = [f"train_{k}" for k in names] + self.train_stats_keys
        self.valid_stats_keys = [f"val_{k}" for k in names] + self.valid_stats_keys

        self._rewrite_csv()
    
    def _del_loss_attr(self, names):
        for name in names:
            self.stats_keys.remove(name)
            self.train_stats_keys.remove(f"train_{name}")
            self.valid_stats_keys.remove(f"val_{name}")
        
        self._rewrite_csv()

    def _train_epoch(self, epoch):
        self.model.train()

        epoch_loss_dict, epoch_metric_dict = {}, {}
        pbar = tqdm(self.train_dataloader)
        for batch in pbar: 
            volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)

            if self.visualize_batch:
                self.visualize(epoch, volumes)
                return

            self.optimizer.zero_grad()
            init_poses, keypoints_3d_pred, heatmaps = self.model(volumes, grid_centers)

            keypoints_3d_gt, keypoints_3d_pred, heatmaps = self._split_data(keypoints_3d_gt, keypoints_3d_pred, heatmaps)
            
            if self.predict_diff:
                # predictions are offsets from the initial predictions
                diff_gt = keypoints_3d_gt - init_poses
                loss_sup = self.loss_sup(diff_gt, keypoints_3d_pred)
                total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, init_poses+keypoints_3d_pred, heatmaps, grid_centers, aux)
                total_loss += loss_sup
                loss_dict["L1DiffLoss"] = loss_sup.clone().detach().cpu().item()
            else:
                total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux)
            
            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items())
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)
            
            if len(self.metrics.names) != 0: 
                if self.predict_diff:
                    metric_dict = self.metrics.evaluate(
                        (init_poses+keypoints_3d_pred).detach().cpu().numpy(), 
                        keypoints_3d_gt.clone().cpu().numpy()
                    )
                else:
                    metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())
                
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
            
            del total_loss, loss_dict, metric_dict, keypoints_3d_pred, init_poses
            if self.predict_diff:
                del loss_sup, diff_gt

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}

    def _valid_epoch(self, epoch):
        self.model.eval()

        epoch_loss_dict = {}
        epoch_metric_dict = {}

        pbar = tqdm(self.valid_dataloader)
        with torch.no_grad():
            for batch in pbar:
                volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)
                init_poses, keypoints_3d_pred, heatmaps = self.model(volumes, grid_centers)

                keypoints_3d_gt, keypoints_3d_pred, heatmaps = self._split_data(keypoints_3d_gt, keypoints_3d_pred, heatmaps)

                if self.predict_diff:
                    # predictions are offsets from the initial predictions
                    diff_gt = keypoints_3d_gt - init_poses
                    loss_sup = self.loss_sup(diff_gt, keypoints_3d_pred)
                    _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, init_poses+keypoints_3d_pred, heatmaps, grid_centers, aux)

                    loss_dict["L1DiffLoss"] = loss_sup.detach().clone().cpu().item()
                else:
                    _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux)
                
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                if len(self.metrics.names) != 0: 
                    if self.predict_diff:
                        metric_dict = self.metrics.evaluate(
                            (init_poses+keypoints_3d_pred).detach().cpu().numpy(), 
                            keypoints_3d_gt.clone().cpu().numpy()
                        )
                    else:
                        metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())

                    epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}
