import torch
from dannce.engine.trainer.dannce_trainer import DannceTrainer
from tqdm import tqdm
from dannce.engine.data.ops import spatial_softmax, expected_value_2d

class BackboneTrainer(DannceTrainer):
    def __init__(self, image_shape=(512, 512), heatmap_shape=(128, 128), **kwargs):
        super(BackboneTrainer, self).__init__(**kwargs)
        
        self.image_shape=image_shape
        self.heatmap_shape=heatmap_shape

        self._construct_grid(heatmap_shape)

    def _construct_grid(self, heatmap_shape):
        # TODO: check the heatmap size!
        h, w = heatmap_shape
        x_coord, y_coord = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack((
            x_coord.flatten(), 
            y_coord.flatten()), dim=-1
        ).unsqueeze(0).unsqueeze(-1).to(self.device) #[1, h*w, 2, 1]   

        self.grid = grid   
    
    def _train_epoch(self, epoch):
        self.model.train()

        epoch_loss_dict, epoch_metric_dict = {}, {}

        pbar = tqdm(self.train_dataloader)
        for batch in pbar: 
            images, keypoints_2d_gt = batch
            images, keypoints_2d_gt = images.to(self.device), keypoints_2d_gt.to(self.device)

            self.optimizer.zero_grad()
            
            heatmaps = self.model(images)

            total_loss, loss_dict = self.loss.compute_loss(keypoints_2d_gt, None, heatmaps, None, None)

            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items())
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            # if len(self.metrics.names) != 0: 
            #     metric_dict = self.metrics.evaluate(keypoints_2d_pred.detach().cpu().numpy(), keypoints_2d_gt.clone().cpu().numpy())
            #     epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

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
                images, keypoints_2d_gt = batch
                images, keypoints_2d_gt = images.to(self.device), keypoints_2d_gt.to(self.device)
                
                heatmaps = self.model(images)

                _, loss_dict = self.loss.compute_loss(keypoints_2d_gt, None, heatmaps, None, None)
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                # metric_dict = self.metrics.evaluate(keypoints_2d_pred.detach().cpu().numpy(), keypoints_2d_gt.clone().cpu().numpy())
                # epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}