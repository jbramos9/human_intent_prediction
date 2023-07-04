import os
import yaml
from argparse import ArgumentParser

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

from dataset import get_datamodule
from models import get_model

def get_parser():
    parser = ArgumentParser()
    # Paths
    parser.add_argument('--data_dir', default="hiphop", type=str)
    parser.add_argument('--chk_dir', default="checkpoints", type=str)

    # Training settings
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--data_augmentation', default={"RandomShift": 0.3, "RandomBlanks": 0.3}, type=dict)

    # Run configs
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', default="IntentClassifier", type=str)
    parser.add_argument('--run_name', required=True, type=str)
    parser.add_argument('--no_test', action='store_true')

    # Model config
    parser.add_argument('--model_name', choices=["InFormer", "PtTransformer", "LITTransformer"], type=str)
    parser.add_argument('--input_size', default=250, type=int)
    parser.add_argument('--src_vocab', default=16, type=int)
    parser.add_argument('--num_classes', default=8, type=int)

    parser.add_argument('--loss', default='CrossEntropyLoss', type=str)
    parser.add_argument('--optimizer', default={"name":"SGD", "momentum":0.9}, type=dict)

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    parser.add_argument('--num_heads', default=10, type=int)
    parser.add_argument('--depth', default=6, type=int)
    parser.add_argument('--d_ff', default=512, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--use_config', action='store_true')

    return parser.parse_args()

def _scheduler_packer(config):
    config["scheduler"] = {
        "lr_scheduler" : config.pop('lr_scheduler'),
        'lr_decay_steps': config.pop('lr_decay_steps'),
        "lr_decay_rate": config.pop('lr_decay_rate'),
        "lr_decay_min_lr": config.pop('lr_decay_min_lr'),
    }


def get_callbacks(config):
    return ModelCheckpoint(
        dirpath=config['chk_dir'],
        filename=config['run_name'],
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

def get_logger(config, run_name):
    loggers = []
    if config['use_wandb']:
        wandb.login()

        project_name = config['project_name']

        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config # Track hyperparameters and run metadata
        )
    
        loggers.append(
            WandbLogger(
                project=project_name,
                name=run_name,
                log_model="all",
        ))
    
    if loggers:
        return loggers
    return None

def load_config():
    with open('config_small.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    run_name = config['run_name']

    data_module = get_datamodule(config)
    model = get_model(config)
    callbacks = get_callbacks(config)
    wandb_logger = get_logger(config, run_name)

    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=config['max_epoch'],
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)

    if not config['no_test']:
        ckpt_path = os.path.join(config['chk_dir'], run_name + '.ckpt')
        trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == "__main__":

    """
    Before running make sure the config file is properly configured
    I extracted a validation set from the train set, please check that at dataset.py
    """

    args = get_parser()

    if args.use_config:
        config = load_config()
    else:
        config = vars(args)
        _scheduler_packer(config)

    config['run_name'] = args.run_name

    main(config)


    