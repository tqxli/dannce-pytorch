import torch
import os, csv
from tqdm import tqdm

from dannce.engine.trainer.dannce_trainer import DannceTrainer
from dannce.engine.trainer.train_utils import prepare_batch
import dannce.engine.models.loss as custom_losses

class GCNTrainer(DannceTrainer):
    def __init__(self, predict_diff=True, multi_stage=False, relpose=True, dual_sup=False, **kwargs):
        super().__init__(**kwargs)

        # GCN-specific training options
        self.predict_diff = predict_diff
        self.multi_stage = multi_stage
        self.relpose = relpose
        self.dual_sup = dual_sup and relpose

        # adjust loss functions and attributes
        if predict_diff or multi_stage or relpose:
            self.loss_sup = custom_losses.L1Loss()

        if predict_diff and (not self.multi_stage): 
            self._add_loss_attr(["L1DiffLoss"])
        elif predict_diff and self.multi_stage:
            self._add_loss_attr(["Stage1L1DiffLoss", "Stage2L1DiffLoss", "Stage3L1DiffLoss"])
        elif self.multi_stage:
            self._add_loss_attr(["Stage1L1Loss", "Stage2L1Loss", "Stage3L1Loss"])
        
        if not self.dual_sup:
            try:
                self._del_loss_attr(["L1Loss"])
                self.loss.loss_fcns.pop("L1Loss")
            except:
                self._del_loss_attr(["WeightedL1Loss"])
                self.loss.loss_fcns.pop("WeightedL1Loss")
    
    def _forward(self, epoch, batch):
        volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)

        if self.visualize_batch:
            self.visualize(epoch, volumes)
            return

        init_poses, keypoints_3d_pred, heatmaps = self.model(volumes, grid_centers)
        
        if not isinstance(keypoints_3d_pred, list):
            keypoints_3d_gt, keypoints_3d_pred, heatmaps = self._split_data(keypoints_3d_gt, keypoints_3d_pred, heatmaps)   

        return init_poses, keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux     

    def _train_epoch(self, epoch):
        self.model.train()

        epoch_loss_dict, epoch_metric_dict = {}, {}
        pbar = tqdm(self.train_dataloader)
        for batch in pbar: 
            self.optimizer.zero_grad()

            init_poses, keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux = self._forward(epoch, batch)

            if self.predict_diff and (not self.multi_stage) and (not self.relpose):
                # predictions are offsets from the initial predictions
                diff_gt = keypoints_3d_gt - init_poses
                loss_sup = self.loss_sup(diff_gt, keypoints_3d_pred)
                total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, init_poses+keypoints_3d_pred, heatmaps, grid_centers, aux)
                total_loss += loss_sup
                loss_dict["L1DiffLoss"] = loss_sup.clone().detach().cpu().item()
            elif self.multi_stage:
                # loss_sup0 = self.loss_sup(keypoints_3d_gt, init_poses)

                if self.predict_diff:
                    gt = keypoints_3d_gt - init_poses
                else:
                    gt = keypoints_3d_gt
                loss_sup1 = self.loss_sup(gt, keypoints_3d_pred[0])
                loss_sup2 = self.loss_sup(gt, keypoints_3d_pred[1])
                loss_sup3 = self.loss_sup(gt, keypoints_3d_pred[2])

                aux_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, init_poses+keypoints_3d_pred[2], heatmaps, grid_centers, aux)

                # loss_dict["Stage0L1Loss"] = loss_sup0.clone().detach().cpu().item()
                if self.predict_diff:
                    loss_dict["Stage1L1DiffLoss"] = loss_sup1.clone().detach().cpu().item()
                    loss_dict["Stage2L1DiffLoss"] = loss_sup2.clone().detach().cpu().item()
                    loss_dict["Stage3L1DiffLoss"] = loss_sup3.clone().detach().cpu().item()
                else:
                    loss_dict["Stage1L1Loss"] = loss_sup1.clone().detach().cpu().item()
                    loss_dict["Stage2L1Loss"] = loss_sup2.clone().detach().cpu().item()
                    loss_dict["Stage3L1Loss"] = loss_sup3.clone().detach().cpu().item()

                total_loss = loss_sup1 + loss_sup2 + loss_sup3 + aux_loss
            elif self.relpose:
                com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1) #[N, 3, 1]
                nvox = round(grid_centers.shape[1]**(1/3))
                vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
                keypoints_3d_gt_rel = (keypoints_3d_gt - com3d) / vsize
                
                if not self.predict_diff:    
                    loss_sup = self.loss_sup(keypoints_3d_gt_rel, keypoints_3d_pred)
                    keypoints_3d_pred = keypoints_3d_pred * vsize + com3d
                    total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux)
                    total_loss += loss_sup
                    loss_dict["L1Loss"] = loss_sup.clone().detach().cpu().item()
                else:
                    diff_gt_rel = (keypoints_3d_gt - init_poses) / vsize
                    diff_loss = self.loss_sup(diff_gt_rel, keypoints_3d_pred)

                    # scale back to original, so that bone length loss can be correctly computed
                    keypoints_3d_pred = keypoints_3d_pred * vsize #+ com3d
                    total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, init_poses + keypoints_3d_pred, heatmaps, grid_centers, aux)
                    total_loss += diff_loss
                    loss_dict["L1DiffLoss"] = diff_loss.clone().detach().cpu().item()
                    
                    if self.dual_sup:
                        init_poses_rel = (init_poses - com3d) / vsize
                        diff_gt_rel = keypoints_3d_gt_rel - init_poses_rel
                        pose_loss = 0.1 * self.loss_sup(keypoints_3d_gt, init_poses)
                        total_loss += pose_loss
                        loss_dict["L1Loss"] = pose_loss.clone().detach().cpu().item()
            else:
                total_loss, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux)
            
            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items())
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)
            
            if len(self.metrics.names) != 0: 
                if self.predict_diff and (not self.multi_stage):
                    metric_dict = self.metrics.evaluate(
                        (init_poses+keypoints_3d_pred).detach().cpu().numpy(), 
                        keypoints_3d_gt.clone().cpu().numpy()
                    )
                elif self.predict_diff and self.multi_stage:
                    metric_dict = self.metrics.evaluate(
                        (init_poses+keypoints_3d_pred[-1]).detach().cpu().numpy(), 
                        keypoints_3d_gt.clone().cpu().numpy()
                        )
                else:
                    metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())
                
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
            
            del total_loss, loss_dict, metric_dict, keypoints_3d_pred, init_poses

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
                init_poses, keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux = self._forward(epoch, batch)

                if self.predict_diff and (not self.multi_stage) and (not self.relpose):
                    # predictions are offsets from the initial predictions
                    diff_gt = keypoints_3d_gt - init_poses
                    loss_sup = self.loss_sup(diff_gt, keypoints_3d_pred)
                    _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, init_poses+keypoints_3d_pred, heatmaps, grid_centers, aux)

                    loss_dict["L1DiffLoss"] = loss_sup.detach().clone().cpu().item()
                elif self.multi_stage:
                    # loss_sup0 = self.loss_sup(keypoints_3d_gt, init_poses)

                    if self.predict_diff:
                        gt = keypoints_3d_gt - init_poses
                    else:
                        gt = keypoints_3d_gt
                    loss_sup1 = self.loss_sup(gt, keypoints_3d_pred[0])
                    loss_sup2 = self.loss_sup(gt, keypoints_3d_pred[1])
                    loss_sup3 = self.loss_sup(gt, keypoints_3d_pred[2])

                    _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, init_poses+keypoints_3d_pred[2], heatmaps, grid_centers, aux)

                    if self.predict_diff:
                        loss_dict["Stage1L1DiffLoss"] = loss_sup1.clone().detach().cpu().item()
                        loss_dict["Stage2L1DiffLoss"] = loss_sup2.clone().detach().cpu().item()
                        loss_dict["Stage3L1DiffLoss"] = loss_sup3.clone().detach().cpu().item()
                    else:
                        loss_dict["Stage1L1Loss"] = loss_sup1.clone().detach().cpu().item()
                        loss_dict["Stage2L1Loss"] = loss_sup2.clone().detach().cpu().item()
                        loss_dict["Stage3L1Loss"] = loss_sup3.clone().detach().cpu().item()
                elif self.relpose:
                    com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1) #[N, 3, 1]
                    nvox = round(grid_centers.shape[1]**(1/3))
                    vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
                    keypoints_3d_gt_rel = (keypoints_3d_gt - com3d) / vsize

                    if not self.predict_diff:
                        loss_sup = self.loss_sup(keypoints_3d_gt_rel, keypoints_3d_pred)
                        keypoints_3d_pred = keypoints_3d_pred * vsize + com3d
                        _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux)
                        loss_dict["L1Loss"] = loss_sup.clone().detach().cpu().item()
                    else:
                        diff_gt_rel = (keypoints_3d_gt - init_poses) / vsize
                        diff_loss = self.loss_sup(diff_gt_rel, keypoints_3d_pred)

                        # scale back to original, so that bone length loss can be correctly computed
                        keypoints_3d_pred = keypoints_3d_pred * vsize #+ com3d
                        _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, init_poses + keypoints_3d_pred, heatmaps, grid_centers, aux)
                        loss_dict["L1DiffLoss"] = diff_loss.clone().detach().cpu().item()
                        if self.dual_sup:
                            init_poses_rel = (init_poses - com3d) / vsize
                            diff_gt_rel = keypoints_3d_gt_rel - init_poses_rel
                            pose_loss = 0.1 * self.loss_sup(keypoints_3d_gt, init_poses)
                            loss_dict["L1Loss"] = pose_loss.clone().detach().cpu().item()
                else:
                    _, loss_dict = self.loss.compute_loss(keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux)
                
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                if len(self.metrics.names) != 0: 
                    if self.predict_diff and (not self.multi_stage):
                        metric_dict = self.metrics.evaluate(
                            (init_poses+keypoints_3d_pred).detach().cpu().numpy(), 
                            keypoints_3d_gt.clone().cpu().numpy()
                        )
                    elif self.predict_diff and self.multi_stage:
                        metric_dict = self.metrics.evaluate(
                            (init_poses+keypoints_3d_pred[-1]).detach().cpu().numpy(), 
                            keypoints_3d_gt.clone().cpu().numpy()
                            )
                    else:
                        metric_dict = self.metrics.evaluate(keypoints_3d_pred.detach().cpu().numpy(), keypoints_3d_gt.clone().cpu().numpy())

                    epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}
