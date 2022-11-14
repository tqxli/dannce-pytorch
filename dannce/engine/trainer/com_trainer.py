import torch
import csv, os
from tqdm import tqdm
import numpy as np

from dannce.engine.trainer.dannce_trainer import DannceTrainer
from dannce.engine.data.ops import expected_value_2d, spatial_softmax
from dannce.engine.data.processing import get_peak_inds

class COMTrainer(DannceTrainer):
    def __init__(self, return_heatmaps=True, **kwargs):
        super().__init__(dannce=False, **kwargs)

        stats_file = open(os.path.join(self.params["com_train_dir"], "training.csv"), 'w', newline='')
        stats_writer = csv.writer(stats_file)
        self.stats_keys = [*self.loss.names, *self.metrics.names]
        self.train_stats_keys = ["train_"+k for k in self.stats_keys]
        self.valid_stats_keys = ["val_"+k for k in self.stats_keys]
        stats_writer.writerow(["Epoch", *self.train_stats_keys, *self.valid_stats_keys])
        stats_file.close()

        self.return_heatmaps = return_heatmaps
    
    def _get_coords(self, pred):
        pred_coords = []
        for batch in pred:
            coords = []
            for joint in batch:
                coord = get_peak_inds(joint)[::-1]
                coords.append(coord)
            coords = np.stack(coords, axis=0)
            pred_coords.append(coords)
        pred_coords = np.stack(pred_coords, axis=0) #[bs, nj, 2]

        return pred_coords

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # open csv
            stats_file = open(os.path.join(self.params["com_train_dir"], "training.csv"), 'a', newline='')
            stats_writer = csv.writer(stats_file)
            stats = [epoch]
            # train
            train_stats = self._train_epoch(epoch)

            for k in self.stats_keys:
                stats.append(train_stats[k])
                    
            result_msg = f"Epoch[{epoch}/{self.epochs}] " \
                + "".join(f"train_{k}: {val:.6f} " for k, val in train_stats.items()) 
            
            # validation
            valid_stats= self._valid_epoch(epoch)

            for k in self.stats_keys:
                stats.append(valid_stats[k])
                    
            result_msg = result_msg \
                + "".join(f"val_{k}: {val:.6f} " for k, val in valid_stats.items()) 
            self.logger.info(result_msg)

            # write stats to csv
            stats_writer.writerow(stats)
            stats_file.close()

            # write stats to tensorboard
            for k, v in zip([*self.train_stats_keys, *self.valid_stats_keys], stats[1:]):
                self.writer.add_scalar(k, v, epoch)

            self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()

        # with torch.autograd.set_detect_anomaly(False):
        epoch_loss_dict, epoch_metric_dict = {}, {}
        pbar = tqdm(self.train_dataloader)
        for batch in pbar: 
            self.optimizer.zero_grad()
            
            imgs, gt = batch[0].to(self.device), batch[1].to(self.device)
            # workaround
            if self.params["use_temporal"]:
                gt[[0, 1, 3]] = float('nan')

            pred = self.model(imgs)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            if not self.return_heatmaps:
                pred = spatial_softmax(pred)
                pred = expected_value_2d(pred)
                pred = pred.permute(0, 2, 1)

            total_loss, loss_dict = self.loss.compute_loss(gt, pred, pred)
            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.6f} " for loss, val in loss_dict.items())
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            if len(self.metrics.names) != 0: 
                pred_coords = self._get_coords(pred.detach().cpu().numpy())
                gt_coords = self._get_coords(gt.detach().cpu().numpy()) #[bs, nj, 2]

                pred_coords = np.transpose(pred_coords, (0, 2, 1))
                gt_coords = np.transpose(gt_coords, (0, 2, 1))

                metric_dict = self.metrics.evaluate(pred_coords, gt_coords)
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

        if self.lr_scheduler is not None:
            if self.params["lr_scheduler"]["type"] != "ReduceLROnPlateau":
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
                imgs, gt = batch[0].to(self.device), batch[1].to(self.device)
                
                # workaround
                if self.params["use_temporal"]:
                    gt[[0, 1, 3]] = float('nan')
                
                pred = self.model(imgs)
                if isinstance(pred, tuple):
                    pred = pred[0]

                if not self.return_heatmaps:
                    pred = spatial_softmax(pred)
                    pred = expected_value_2d(pred)
                    pred = pred.permute(0, 2, 1)

                total_loss, loss_dict = self.loss.compute_loss(gt, pred, pred)
                result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.6f} " for loss, val in loss_dict.items())
                pbar.set_description(result)

                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                if len(self.metrics.names) != 0: 
                    pred_coords = self._get_coords(pred.detach().cpu().numpy())
                    gt_coords = self._get_coords(gt.detach().cpu().numpy()) #[bs, nj, 2]

                    pred_coords = np.transpose(pred_coords, (0, 2, 1))
                    gt_coords = np.transpose(gt_coords, (0, 2, 1))

                    metric_dict = self.metrics.evaluate(pred_coords, gt_coords)
                    epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        if self.lr_scheduler is not None:
            if self.params["lr_scheduler"]["type"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(total_loss)
        
        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}
