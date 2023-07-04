import torch.nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lrs

import lightning.pytorch as pl

from models.criterion import get_criterion
from models.gotr import build_model
from util.misc import nested_tensor_from_tensor_list, get_gazed_objects

from metrics.metrics import GazeMetrics, BBoxMetrics

import datasets.gom as gom


class GAZEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train = gom.build_dataset(
                "train", self.config, skip_frames=3, skip_intents=None,
            )
            self.val = gom.build_dataset(
                "test", self.config, skip_frames=8, skip_intents=None,
            )

        if stage == "test":
            self.test = gom.build_dataset("test", self.config, skip_frames=0, skip_intents=None)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.config["batch_size"],
            pin_memory=True,
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.config["batch_size"],
            pin_memory=True,
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        return tuple(batch)


class LITGaMer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.model = build_model(config)  # model
        self.criterion = get_criterion(config)  # loss

        # Gaze Metrics: Gaze Accuracy, Precision, Recall (model and per class)
        self.gaze_metrics_train = GazeMetrics(prefix="train")
        self.gaze_metrics_test = GazeMetrics(prefix="test")
        self.gaze_metrics_val = GazeMetrics(prefix="val")
        
        # self.bbox_metrics_train = BBoxMetrics(prefix="train")
        self.bbox_metrics_test = BBoxMetrics(prefix="test")
        # self.bbox_metrics_val = BBoxMetrics(prefix="val")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.config["model"]["backbone"]["lr"],
            },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.config["optimizer"]["lr"],
            weight_decay=self.config["optimizer"]["weight_decay"],
        )
        scheduler = lrs.StepLR(
            optimizer, step_size=self.config["optimizer"]["scheduler"]["lr_drop"]
        )
        return [optimizer], [scheduler]

    def _common_step(self, prefix, batch, gaze_metric, bbox_metric):
        inputs, targets = batch
        predictions = self(inputs)

        # computation of loss
        loss_dict = self.criterion(predictions, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        log_dict = {f"{prefix}_{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.config["batch_size"],
            sync_dist=True,
        )

        self.log_dict(
            {f"{prefix}_loss": losses},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config["batch_size"],
            sync_dist=True,
        )
        
        gaze_metric.update(predictions, targets)
        if bbox_metric:
            bbox_metric.update(predictions, targets)

        self.log(f"{prefix}_model_accuracy", gaze_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_model_mAP", bbox_metric, on_step=False, on_epoch=True, prog_bar=True)

        return losses

    def training_step(self, train_batch, batch_idx):
        return self._common_step("train", train_batch, self.gaze_metrics_train, None)

    def validation_step(self, val_batch, batch_idx):
        return self._common_step("val", val_batch, self.gaze_metrics_val, None)

    def test_step(self, test_batch, batch_idx):
        # return self._common_step("test", test_batch, self.gaze_metrics_test, self.bbox_metrics_test)
        prefix = "test"
        gaze_metric = self.gaze_metrics_test
        inputs, targets = test_batch
        predictions = self(inputs)

        # save to file
        gazed_objects, _ = get_gazed_objects(predictions, targets, self.device)
        video_files = [target["video_file"] for target in targets]
        frame_numbers = [target["frame_number"] for target in targets]
        with open(f"with_head_from_no_depth.csv", 'a') as f:
            for video_file, frame_number, gazed_object in zip(video_files, frame_numbers, gazed_objects):
                output = ",".join([video_file, str(frame_number), str(gazed_object.item()) + "\n"])
                f.write(output)

        # computation of loss
        loss_dict = self.criterion(predictions, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        log_dict = {f"{prefix}_{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.config["batch_size"],
            # sync_dist=True,
        )

        self.log_dict(
            {f"{prefix}_loss": losses},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config["batch_size"],
            # sync_dist=True,
        )

        gaze_metric.update(predictions, targets)
        # bbox_metric.update(predictions, targets)

        self.log(f"{prefix}_model_accuracy", gaze_metric, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(f"{prefix}_model_mAP", bbox_metric, on_step=False, on_epoch=True, prog_bar=True)

        del predictions

        return losses
        # return self._common_step("test", test_batch, self.gaze_metrics_test)

def get_datamodule(config) -> pl.LightningDataModule:
    return GAZEDataModule(config)


def get_model(config) -> pl.LightningModule:
    return LITGaMer(config)
