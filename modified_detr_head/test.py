import yaml
from lightnings import LITGaMer, get_datamodule
import lightning.pytorch as pl
from trainer import get_logger

def test():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    config["batch_size"] = 20

    datamodule = get_datamodule(config)
    model = LITGaMer.load_from_checkpoint("checkpoints/with_head_pretrained.ckpt")

    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=[2],
        # logger=get_logger(config),
    )
    trainer.test(model, datamodule=datamodule)

if __name__=="__main__":
    test()
