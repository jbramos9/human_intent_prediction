import yaml
from util.misc import get_gazed_objects
from lightnings import LITGaMer, get_datamodule
from tqdm import tqdm


def main(ckpt, device):

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = LITGaMer.load_from_checkpoint(ckpt)
    model.eval()
    model.to(device)
    datamodule = get_datamodule(config)
    datamodule.setup("test")

    for (batch, targets) in tqdm(datamodule.test_dataloader()):
        batch = batch.to(device)
        # targets = [target.to(device) for target in targets]
        predictions = model(batch)

        # save to file
        gazed_objects, _ = get_gazed_objects(predictions, targets, device)
        video_files = [target["video_file"] for target in targets]
        frame_numbers = [target["frame_number"] for target in targets]
        with open("with_head_pretrained_latest.csv", 'a') as f:
            for video_file, frame_number, gazed_object in zip(video_files, frame_numbers, gazed_objects):
                output = ",".join([video_file, str(frame_number), str(gazed_object.item()) + "\n"])
                f.write(output)
        
        del predictions

if __name__=="__main__":
    ckpt = "checkpoints/with_head_pretrained_latest.ckpt"
    main(ckpt, device="cuda")
