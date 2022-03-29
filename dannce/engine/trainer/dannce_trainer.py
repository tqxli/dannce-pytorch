import torch
import csv, os
import imageio
from dannce.engine.trainer.base_trainer import BaseTrainer
from dannce.engine.trainer.train_utils import prepare_batch, LossHelper, MetricHelper
import dannce.engine.data.processing as processing
import numpy as np

class DannceTrainer(BaseTrainer):
    def __init__(self, device, train_dataloader, valid_dataloader, lr_scheduler=None, visualize_batch=False, **kwargs):
        super().__init__(**kwargs)

        self.loss = LossHelper(self.params)
        self.metrics = MetricHelper(self.params)
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler

        self.visualize_batch = visualize_batch

        # set up csv file for tracking training and validation stats
        stats_file = open(os.path.join(self.params["dannce_train_dir"], "training.csv"), 'w', newline='')
        stats_writer = csv.writer(stats_file)
        self.stats_keys = [*self.loss.names, *self.metrics.names]
        self.train_stats_keys = ["train_"+k for k in self.stats_keys]
        self.valid_stats_keys = ["val_"+k for k in self.stats_keys]
        stats_writer.writerow(["Epoch", *self.train_stats_keys, *self.valid_stats_keys])
        stats_file.close()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # open csv
            stats_file = open(os.path.join(self.params["dannce_train_dir"], "training.csv"), 'a', newline='')
            stats_writer = csv.writer(stats_file)
            stats = [epoch]
            # train
            train_stats = self._train_epoch(epoch)

            for k in self.stats_keys:
                stats.append(train_stats[k])
                    
            result_msg = f"Epoch[{epoch}/{self.epochs}] " \
                + "".join(f"train_{k}: {val:.4f} " for k, val in train_stats.items()) 
            
            # validation
            valid_stats= self._valid_epoch(epoch)

            for k in self.stats_keys:
                stats.append(valid_stats[k])
                    
            result_msg = result_msg \
                + "".join(f"val_{k}: {val:.4f} " for k, val in valid_stats.items()) 
            self.logger.info(result_msg)

            # write stats to csv
            stats_writer.writerow(stats)
            stats_file.close()

            # write stats to tensorboard
            for k, v in zip([*self.train_stats_keys, *self.valid_stats_keys], stats):
                self.writer.add_scalar(k, v, epoch)

            # save checkpoints after each save period or at the end of training
            if epoch % self.save_period == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch)


    def _train_epoch(self, epoch):
        self.model.train()

        with torch.autograd.set_detect_anomaly(True):
            epoch_loss_dict, epoch_metric_dict = {}, {}
            for batch in self.train_dataloader: 
                volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)
                if self.visualize_batch:
                    self.visualize(epoch, volumes)
                    return

                self.optimizer.zero_grad()
                keypoints_3d_pred, _ = self.model(volumes, grid_centers)
                total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred)
                result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items())
                print(result, end='\r')

                total_loss.backward()
                self.optimizer.step()

                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.cpu().numpy())
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
            for batch in self.valid_dataloader:
                volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)
                keypoints_3d_pred, _ = self.model(volumes, grid_centers)

                _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred)
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.cpu().numpy())
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}
    
    def _update_step(self, epoch_dict, step_dict):
        if len(epoch_dict) == 0:
            for k, v in step_dict.items():
                epoch_dict[k] = [v]
        else:
            for k, v in step_dict.items():
                epoch_dict[k].append(v)
        return epoch_dict
    
    def _average(self, epoch_dict):
        for k, v in epoch_dict.items():
            valid_num = sum([item > 0 for item in v])
            epoch_dict[k] = sum(v) / valid_num
        return epoch_dict
    
    def visualize(self, epoch, volumes):
        tifdir = os.path.join(self.params["dannce_train_dir"], "debug_volumes", f"epoch{epoch}")
        if not os.path.exists(tifdir):
            os.makedirs(tifdir)
        print("Dump training volumes to {}".format(tifdir))
        volumes = volumes.clone().detach().cpu().permute(0, 2, 3, 4, 1).numpy()
        for i in range(volumes.shape[0]):
            for j in range(volumes.shape[1] // self.params["chan_num"]):
                im = volumes[
                    i,
                    :,
                    :,
                    :,
                    j*self.params["chan_num"]: (j+1)self.params["chan_num"],
                ]
                im = processing.norm_im(im) * 255
                im = im.astype("uint8")
                of = os.path.join(
                    tifdir,
                    f"sample{i}" + "_cam" + str(j) + ".tif",
                )
                imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))



