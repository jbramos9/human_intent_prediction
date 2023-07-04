import sys, inspect
import copy
from typing import Any, Mapping

import torch.nn as nn
import torch.optim
from torch.optim import SGD
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as pl

from transformer import *
from utils.metrics import HIPHOPAccuracy

__all__ = ["PtTransformer", "InFormer", "LITTransformer"]

def _get_all_models_here(): # deprecated
    res = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and obj.__module__ == __name__:
            res.append(name)
    return res

def get_model(config):
    model_list = __all__
    model_name = config['model_name']
    assert model_name in model_list, (f"model_name must be one of these {model_list}.")

    if model_name == "PtTransformer":
        return PtTransformer(
            src_vocab=config["src_vocab"],
            d_model=config["input_size"],
            num_classes=config["num_classes"],
            N=config["depth"],
            h=config["num_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            loss=config['loss'],
            optimizer=config['optimizer'],
            scheduler=config['scheduler'],
        )
    elif model_name == "InFormer":
        return InFormer(
            src_vocab=config["src_vocab"],
            d_model=config["input_size"],
            num_classes=config["num_classes"],
            N=config["depth"],
            h=config["num_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            loss=config['loss'],
            optimizer=config['optimizer'],
            scheduler=config['scheduler'],
        )
    elif model_name == "LITTransformer":
        return LITTransformer(
            src_vocab=config["src_vocab"],
            d_model=config["input_size"],
            num_classes=config["num_classes"],
            N=config["depth"],
            h=config["num_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
        )
    else:
        raise ValueError('Invalid model name!')


class BaseModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # loss
        self.loss = getattr(nn, self.hparams.loss, nn.CrossEntropyLoss)()

        # Metrics
        self.train_acc = HIPHOPAccuracy()
        self.val_acc = HIPHOPAccuracy()
        self.test_acc = HIPHOPAccuracy()
    
    def configure_optimizers(self):
        optimizer_config = self.hparams.optimizer.copy()
        name = optimizer_config.pop("name")

        if name is None:
            optimizer_class = getattr(torch.optim, "SGD")
            optimizer_config.clear()
            optimizer_config["momentum"] = 0.9
        else:
            optimizer_class = getattr(torch.optim, name)

        optimizer = optimizer_class(self.parameters(), lr=self.hparams.lr, **optimizer_config)
    
        if self.hparams.scheduler["lr_scheduler"] is None:
            return optimizer
        else:
            scheduler_config = self.hparams.scheduler.copy()
            scheduler_name = scheduler_config.pop("lr_scheduler")
            # scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
            # scheduler = scheduler_class(optimizer, **scheduler_config)

            if scheduler_name == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=scheduler_config['lr_decay_steps'],
                                       gamma=scheduler_config['lr_decay_rate'])
            elif scheduler_name == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=scheduler_config['lr_decay_steps'],
                                                  eta_min=scheduler_config['lr_decay_min_lr'])
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        preds = self(inputs)
        loss = self.loss(preds, labels)
        acc = self.train_acc(preds, labels)
        self.log_dict({'train_loss': loss, 'train_acc': acc},
                      on_step=False, on_epoch=True, prog_bar=True )
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        preds = self(inputs)

        loss = self.loss(preds, labels)
        acc = self.val_acc(preds, labels)
        self.log_dict({'val_loss': loss, 'val_acc': acc},
                      on_step=False, on_epoch=True, prog_bar=True )
        return loss


    def test_step(self, batch, batch_idx):
        gaze, labels = batch
        preds = self(gaze)
        loss = self.loss(preds, labels)
        acc = self.test_acc(preds, labels)
        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_epoch=True)
        return loss
    
class InFormer(BaseModule):
    def __init__( self,
        src_vocab : int,
        d_model : int,
        num_classes : int,
        N : int =6,
        h : int =8,
        d_ff : int =512,
        dropout : float =0.1,
        lr : float =1e-3,
        loss : str ="CrossEntropyLoss",
        optimizer : dict ={},
        scheduler : dict ={},
    ):
        """
        Args:
            src_vocab (int) : vocabulary size for embeddings
            d_model (int) : input size of the transformer
            num_classes (int) : number of classes for the Feed Forward Classifier
            N (int) : number of encoders
            h (int) : number of number of heads
            d_ff (int) : dim for Feed Forward
            dropout (float) : dropout probability
            lr (float) : learning rate
            loss (str) : Class name of loss function
            optimizer (dict) : optimizer config
            scheduler (dict) : scheduler config
        """
        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.fc = nn.Linear(d_model, num_classes)


    def forward(self, x):
        x = self.src_embed(x)
        x = self.encoder(x, None)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

class PtTransformer(BaseModule):
    def __init__( self,
        src_vocab : int,
        d_model : int,
        num_classes : int,
        N : int =6,
        h : int =8,
        d_ff : int =512,
        dropout : float =0.1,
        lr : float =1e-3,
        loss : str ="CrossEntropyLoss",
        optimizer : dict ={},
        scheduler : dict ={},
    ):
        """
        Args:
            src_vocab (int) : vocabulary size for embeddings
            d_model (int) : input size of the transformer
            num_classes (int) : number of classes for the Feed Forward Classifier
            N (int) : number of encoders
            h (int) : number of number of heads
            d_ff (int) : dim for Feed Forward
            dropout (float) : dropout probability
            lr (float) : learning rate
            loss (str) : Class name of loss function
            optimizer (dict) : optimizer config
            scheduler (dict) : scheduler config
        """
        super().__init__()

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), position)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=d_ff,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, N, norm=LayerNorm(d_model))
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.src_embed(x)
        x = self.encoder(x, None)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out


class LITTransformer(BaseModule):
    def __init__(
        self, src_vocab, d_model, num_classes, N=6, h=8, d_ff=512, dropout=0.1, lr=1e-3
    ):
        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.fc = nn.Linear(d_model, num_classes)

        # loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.src_embed(x)
        x = self.encoder(x, None)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out
    
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer