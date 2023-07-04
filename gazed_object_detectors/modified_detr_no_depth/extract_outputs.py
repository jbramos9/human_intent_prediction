import yaml
from util.misc import get_gazed_objects, get_head_box
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
        with open("with_head_pretrained_train.csv", 'a') as f:
            for video_file, frame_number, gazed_object in zip(video_files, frame_numbers, gazed_objects):
                output = ",".join([video_file, str(frame_number), str(gazed_object.item()) + "\n"])
                f.write(output)
        
        del predictions

def extract_heads(ckpt, config, device):

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
        bboxes = get_head_box(predictions, [t["org_size"] for t in targets]) #targets["org_size"])
        
        video_files = [target["video_file"] for target in targets]
        frame_numbers = [target["frame_number"] for target in targets]
        with open("no_depth_boxes.csv", 'a') as f:
            for video_file, frame_number, bbox in zip(video_files, frame_numbers, bboxes):
                bboxes = ",".join([str(box.item()) for box in bbox])
                output = ",".join([video_file, str(frame_number), bboxes + "\n"])
                f.write(output)

        del predictions
        break

if __name__=="__main__":
    ckpt = "checkpoints/no_depth_v2.ckpt"

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    extract_heads(ckpt, config, device="cuda")

