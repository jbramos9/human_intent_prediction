import datasets.visualization as vis
from datasets.gom import build_dataset
import yaml
from lightnings import LITGaMer
from models.gotr import build_model
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import os
import cv2
from metrics.metrics import GazeMetrics
from pprint import pprint


def test_visual(config):
    
    
    ds = build_dataset("train", config)

    image, target = ds[0]

    vis.dataset_vis(image, target)

def test_trained_model(config):
    CKPT_PATH = 'checkpoints\GOTr-v1.ckpt'
    model = LITGaMer.load_from_checkpoint(CKPT_PATH)
    model.eval()
    model.to("cuda")

    ds = build_dataset("test", config, skip_frames=1)
    dloader = DataLoader(ds, batch_size=10, collate_fn=utils.collate_fn, shuffle=True)
    # batch_size should always be 1
    stop=0

    for img, target in dloader:
        image = img.to("cuda")
        pred = model(image)

        frame = get_frame(target[0]["video_file"], target[0]["frame_number"], config)
        vis.pred_vis(frame, pred)

        del image, pred

        stop+=1
        if stop==5: break

def test_metric(config):
    ds = build_dataset("test", config, skip_intents=[])
    dloader = DataLoader(ds, batch_size=2, collate_fn=utils.collate_fn, shuffle=False)

    metrics = GazeMetrics()
    model = build_model(config, pre_trained=False)

    for image, target in dloader:    
        image = image.to("cuda")
        pred = model(image)
        metrics.update(pred, target)
        pprint(metrics.compute())
        input()
    

def get_frame(vid_path, frame_no, config):
    vid_path = os.path.join(config["directories"]["root"], vid_path)  
    vidcap = cv2.VideoCapture(vid_path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, img = vidcap.read()
    vidcap.release()
    return img

if __name__=="__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    test_metric(config)


# with open("config.yaml", 'r') as f:
#     config = yaml.safe_load(f)

# CKPT_PATH = r"data\pretrained_weights\goods-v2.ckpt"
# CKPT_PATH = 'checkpoints\GOTr-v1.ckpt'

# # checkpoint = torch.load(CKPT_PATH)
# # state_dict_ = checkpoint["state_dict"]

# # state_dict = {".".join(k.split(".")[1:]): v  for k,v in state_dict_.items() if "model" in k}

# # model = build_model(config, False)
# # model.load_state_dict(state_dict, strict=True)
# model = LITGaMer.load_from_checkpoint(CKPT_PATH)
# model.eval()
# model.to("cpu")

# ds = build_dataset("train", config)
# dloader = DataLoader(ds, batch_size=1, collate_fn=utils.collate_fn, shuffle=True)

# stop=0
# for img, target in dloader:
#     # # img = img.to("cuda")
#     # pred = model(img)

#     image, mask = img.decompose()
#     # vis.pred_vis(image[0], pred)

#     vis.dataset_vis(image[0], target)

#     break

#     stop+=1
#     break
#     if stop==20:
#         break
