import torch
from dannce.engine.trainer.dannce_trainer import DannceTrainer
from tqdm import tqdm
from copy import deepcopy


class VoxelPoseTrainer(DannceTrainer):
    def __init__(self, **kwargs):
        super(VoxelPoseTrainer, self).__init__(**kwargs)
    
    def _train_epoch(self, epoch):
        self.model.train()

        epoch_loss_dict, epoch_metric_dict = {}, {}

        pbar = tqdm(self.train_dataloader)
        for batch in pbar:
            images, y2d_gaussian, grids, cameras, keypoints_3d_gt = batch
            images, grids, keypoints_3d_gt = images.to(self.device), grids.to(self.device), keypoints_3d_gt.to(self.device)
            y2d_gaussian = y2d_gaussian.to(self.device)

            self.optimizer.zero_grad()

            heatmaps, keypoints_3d_pred = self.model(images, grids, cameras)
            total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grids, None, heatmaps_gt=y2d_gaussian)

            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items())
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())
            epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}
    
    def _valid_epoch(self, epoch):
        self.model.eval()

        epoch_loss_dict = {}
        epoch_metric_dict = {}
        with torch.no_grad():
            pbar = tqdm(self.valid_dataloader)

            for batch in pbar:
                images, y2d_gaussian, grids, cameras, keypoints_3d_gt = batch
                images, grids, keypoints_3d_gt = images.to(self.device), grids.to(self.device), keypoints_3d_gt.to(self.device)
                y2d_gaussian = y2d_gaussian.to(self.device)
                
                heatmaps, keypoints_3d_pred = self.model(images, grids, cameras)

                _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grids, None, heatmaps_gt=y2d_gaussian)
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}