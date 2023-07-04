import yaml
from lightnings import LITGaMer, get_datamodule
import lightning.pytorch as pl
from trainer import get_logger

def test():

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    config["batch_size"] = 10

    datamodule = get_datamodule(config)
    model = LITGaMer.load_from_checkpoint("checkpoints/no_depth_v2.ckpt")

    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=1,
        logger=get_logger(config, config["run_name"]),
    )
    trainer.test(model, datamodule=datamodule)

if __name__=="__main__":
    test()
