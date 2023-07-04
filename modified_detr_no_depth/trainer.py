import os
import yaml

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

from lightnings import get_datamodule, get_model
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def get_callbacks(config):
    model_checkpoint = ModelCheckpoint(
        dirpath=config["directories"]['chk_dir'],
        filename=config['run_name'],
        save_top_k=1,
        verbose=True,
        monitor='train_loss',
        mode='min',
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', # this should probably the Gaze Accuracy Metrics
        min_delta=0.01, #minimum change in the monitored quantity to qualify as an improvement
        mode='min', # if the monitored value should be decreasing
        patience=3, # number of checks with no improvement after which training will be stopped
    )
    # return [model_checkpoint, early_stopping]
    return [model_checkpoint]


def get_logger(config, run_name):
    loggers = []
    if config['use_wandb']:
        wandb.login()
        project_name = config['project_name']
        # run = wandb.init(
        #     project=project_name,
        #     name=run_name,
        #     group=run_name,
        #     config=config, # Track hyperparameters and run metadata
        #     resume=config['resume']
        # )
        loggers.append(WandbLogger(
                            project=project_name,
                            name=run_name,
                            log_model="all",
                            # id="t9g338ss"
                            ))
    
    if loggers:
        return loggers
    return None

if __name__ == "__main__":

    pl.seed_everything(1)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    run_name = config['run_name']

    data_module = get_datamodule(config)
    model = get_model(config)
    callbacks = get_callbacks(config)
    wandb_logger = get_logger(config, run_name)

    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['num_gpus'],
        max_epochs=config['max_epoch'],
        # precision=config['precision'],
        strategy=config["strategy"],
        logger=wandb_logger,
        callbacks=callbacks,
        # fast_dev_run=5,
        # limit_train_batches=10,
        # limit_val_batches=10,
    )

    if config["resume"]:
        trainer.fit(model, datamodule=data_module, ckpt_path=config["directories"]["chk_path"])
    else:
        trainer.fit(model, datamodule=data_module)

    if not config['no_test']:
        ckpt_path = os.path.join(config['directories']['chk_dir'], run_name + '.ckpt')
        trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

    if config["use_wandb"]:
        wandb.finish()
