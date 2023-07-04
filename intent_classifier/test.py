import yaml
from dataset import get_datamodule
import lightning.pytorch as pl
from models import InFormer
from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True, type=str)
    #checkpoints/trial-v1.ckpt
    return parser.parse_args()

def test(ckpt):

    with open("/content/gdrive/Shareddrives/ECE 199 2s2223/models/intent_classifier/config.yml", 'r') as f:
        config = yaml.safe_load(f)

    config["batch_size"] = 10

    datamodule = get_datamodule(config)
    model = InFormer.load_from_checkpoint(ckpt)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
    )
    trainer.test(model, datamodule = datamodule)

if __name__=="__main__":
    args = get_parser()
    test(args.ckpt_dir)