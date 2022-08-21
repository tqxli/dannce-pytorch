from matplotlib import scale
import torch
import os, csv
from tqdm import tqdm

from dannce.engine.trainer.dannce_trainer import DannceTrainer
from dannce.engine.trainer.train_utils import prepare_batch
import dannce.engine.models.loss as custom_losses

class TransformerTrainer(DannceTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.loss_sup = custom_losses.L1Loss()
        self.loss.loss_fcns.pop("L1Loss")
    
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
            keypoints_3d_pred, heatmaps = self.model(volumes, grid_centers)
            
            if not isinstance(keypoints_3d_pred, list):
                keypoints_3d_gt, keypoints_3d_pred, heatmaps = self._split_data(keypoints_3d_gt, keypoints_3d_pred, heatmaps)
            
            # normalize absolute 3D poses to voxel coordinates
            com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1) #[N, 3, 1]
            nvox = round(grid_centers.shape[1]**(1/3))
            vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
            
            keypoints_3d_gt = (keypoints_3d_gt - com3d) / vsize #[N, 3, J]

            keypoints_3d_gt = keypoints_3d_gt.repeat(keypoints_3d_pred.shape[0], 1, 1, 1) #[6, N, 3, J]
            keypoints_3d_pred = keypoints_3d_pred.transpose(3, 2) #[6, N, 3, J]

            keypoints_3d_pred = keypoints_3d_pred.reshape(-1, *keypoints_3d_pred.shape[2:]) #[6*N, 3, J]
            keypoints_3d_gt = keypoints_3d_gt.reshape(-1, *keypoints_3d_gt.shape[2:])

            pose_loss = self.loss_sup(keypoints_3d_gt, keypoints_3d_pred)

            # scale back for skeleton loss
            keypoints_3d_pred, keypoints_3d_gt = keypoints_3d_pred * vsize, keypoints_3d_gt * vsize

            total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux)
            total_loss += pose_loss
            loss_dict["L1Loss"] = pose_loss.clone().detach().cpu().item()
            
            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items())
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)
            
            if len(self.metrics.names) != 0: 
                metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())
                
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
            
            del total_loss, loss_dict, metric_dict, keypoints_3d_pred

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
                keypoints_3d_pred, heatmaps = self.model(volumes, grid_centers)

                keypoints_3d_gt, keypoints_3d_pred, heatmaps = self._split_data(keypoints_3d_gt, keypoints_3d_pred, heatmaps)

                # normalize absolute 3D poses to voxel coordinates
                com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1) #[N, 3, 1]
                nvox = round(grid_centers.shape[1]**(1/3))
                vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
                
                keypoints_3d_gt = (keypoints_3d_gt - com3d) / vsize #[N, 3, J]
                keypoints_3d_pred = keypoints_3d_pred[-1].transpose(2, 1)

                pose_loss = self.loss_sup(keypoints_3d_gt, keypoints_3d_pred)

                # scale back for skeleton loss
                keypoints_3d_pred, keypoints_3d_gt = keypoints_3d_pred * vsize, keypoints_3d_gt * vsize                
                _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux)
                loss_dict["L1Loss"] = pose_loss.clone().detach().cpu().item()
                
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)
                
                if len(self.metrics.names) != 0: 
                    metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())
                    
                    epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}
