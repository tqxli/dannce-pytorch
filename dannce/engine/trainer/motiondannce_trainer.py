import os
import torch
from tqdm import tqdm

from dannce.engine.trainer.dannce_trainer import DannceTrainer
from dannce.engine.trainer.train_utils import prepare_batch
from dannce.engine.models.motion_discriminator import adv_disc_loss

class MotionDANNCETrainer(DannceTrainer):
    def __init__(self, 
            motion_loader, 
            motion_discriminator, 
            disc_optimizer=None,
            temporal_encoder=None, 
            accumulation_step=4, 
            **kwargs
        ):
        super(MotionDANNCETrainer, self).__init__(**kwargs)

        # we might want to perform multiple sub-batches during one forward pass
        # as mocap sequence length exceeds maximally supported batch size
        self.accumulation_step = accumulation_step

        # temporal encoder over the first stage predictions
        self.temporal_encoder = temporal_encoder
        self.use_temporal_encoder = (temporal_encoder != None)        
    
        # motion discriminator
        self.motion_loader = motion_loader
        self.motion_iter = iter(self.motion_loader)
        self.motion_discriminator = motion_discriminator
        self.masking = torch.LongTensor([8, 12, 15, 19, 16, 20, 7, 11, 3, 5, 4]).to(self.device)

        self.disc_optimizer = disc_optimizer

        # display
        if self.motion_discriminator is not None:
            self.stats_keys += ['DiscriminatorLoss']
            self.train_stats_keys += ["train_DiscriminatorLoss"]
            self.valid_stats_keys += ["val_DiscriminatorLoss"]

    def _train_epoch(self, epoch):
        
        self.model.train()

        epoch_loss_dict, epoch_metric_dict = {}, {}
        pbar = tqdm(self.train_dataloader)

        for batch in pbar:
            volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)

            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()

            inputs = torch.split(volumes, volumes.shape[0] // self.accumulation_step, dim=0)
            grids = torch.split(grid_centers, volumes.shape[0] // self.accumulation_step, dim=0)

            fake_motion_seq = []
            for j in range(self.accumulation_step):
                # regress 3D poses [BS, 3, N_JOINTS]
                keypoints_3d_pred, _ = self.model(inputs[j], grids[j])

                fake_motion_seq.append(keypoints_3d_pred)

            fake_motion_seq = torch.cat(fake_motion_seq, dim=0)
            
            # optional: apply recurrent over initial predictions
            if self.use_temporal_encoder:
                fake_motion_seq = fake_motion_seq.reshape(1, fake_motion_seq.shape[0], -1)
                fake_motion_seq = self.temporal_encoder(fake_motion_seq)
                fake_motion_seq = fake_motion_seq.reshape(*keypoints_3d_gt.shape)

            total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, fake_motion_seq, None, grid_centers, None)

            # select one real motion sequence [1, T, 3*N_JOINTS]
            try:
                real_motion_seq = next(self.motion_iter)
            except StopIteration:
                self.motion_iter = iter(self.motion_loader)
                real_motion_seq = next(self.motion_iter)

            # need to mask non-overlapped joints
            # fake_motion_seq = fake_motion_seq.reshape(*fake_motion_seq.shape[:2], 3, -1)
            overlap_fake_motion_seq = fake_motion_seq[..., self.masking].unsqueeze(0)
            overlap_fake_motion_seq = overlap_fake_motion_seq.reshape(*overlap_fake_motion_seq.shape[:2], -1)
            
            motion_disc_real = self.motion_discriminator(real_motion_seq.to(self.device))
            motion_disc_fake = self.motion_discriminator(overlap_fake_motion_seq)
            
            # compute discriminator loss
            _, _, d_loss = adv_disc_loss(motion_disc_real, motion_disc_fake)

            # compute supervised keypoint loss
            fake_motion_seq = fake_motion_seq.reshape(*keypoints_3d_gt.shape)

            total_loss += d_loss
            loss_dict['DiscriminatorLoss'] = d_loss.item()

            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items())
            # result += "train_{}: {:.4f} ".format("DiscriminatorLoss", d_loss.item()) 
            pbar.set_description(result)

            total_loss.backward()
            # d_loss.backward()
            self.optimizer.step()
            self.disc_optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            metric_dict = self.metrics.evaluate(
                fake_motion_seq.detach().cpu().numpy(), 
                keypoints_3d_gt.clone().cpu().numpy())
            epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

            del total_loss, real_motion_seq, overlap_fake_motion_seq, motion_disc_fake, motion_disc_real, fake_motion_seq, inputs, grids, keypoints_3d_gt

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}   

    def _valid_epoch(self, epoch):
        self.model.eval()

        epoch_loss_dict, epoch_metric_dict = {}, {}
        
        pbar = tqdm(self.valid_dataloader)
        with torch.no_grad():
            for batch in pbar:
                volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)

                inputs = torch.split(volumes, volumes.shape[0] // self.accumulation_step, dim=0)
                grids = torch.split(grid_centers, volumes.shape[0] // self.accumulation_step, dim=0)

                fake_motion_seq = []
                for j in range(self.accumulation_step):
                    # regress 3D poses [BS, 3, N_JOINTS]
                    keypoints_3d_pred, _ = self.model(inputs[j], grids[j])

                    fake_motion_seq.append(keypoints_3d_pred)

                fake_motion_seq = torch.cat(fake_motion_seq, dim=0)
                fake_motion_seq = fake_motion_seq.reshape(1, fake_motion_seq.shape[0], -1)

                # optional: apply recurrent over initial predictions
                if self.use_temporal_encoder:
                    fake_motion_seq = self.temporal_encoder(fake_motion_seq)

                # compute supervised keypoint loss
                fake_motion_seq = fake_motion_seq.reshape(*keypoints_3d_gt.shape)
                _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, fake_motion_seq, None, grid_centers, None)
                
                # no discriminator loss should be computed; display purpose only
                loss_dict['DiscriminatorLoss'] = torch.zeros(()).item()
                
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                metric_dict = self.metrics.evaluate(fake_motion_seq.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

                del inputs, grids, fake_motion_seq, keypoints_3d_gt, volumes, grid_centers

        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}  

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        state = {
            'epoch': epoch,
            'posenet_state_dict': self.model.state_dict(),
            'temporal_encoder_state_dict': self.temporal_encoder.state_dict(),
            'motion_disc_state_dict': self.motion_discriminator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'params': self.params,
        }

        # if self.lr_scheduler is not None:
        #     state["lr_scheduler"] = self.lr_scheduler.state_dict()
        if epoch % self.save_period == 0 or epoch == self.epochs:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint.pth'.format(epoch))
        
        torch.save(state, filename) 