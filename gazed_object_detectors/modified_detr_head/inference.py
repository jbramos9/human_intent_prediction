import cv2
import os
import numpy as np
from PIL import Image
import yaml

import torch

from datasets.gom_transforms import Compose, Normalize, ToTensor
from lightnings import LITGaMer

def make_normalize_transform():
    normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406, 0.41], [0.229, 0.224, 0.225, 0.067]),
        ])
    return normalize

def get_model():
    CKPT_PATH = r"checkpoints\GOTr-v1.ckpt"
    model = LITGaMer.load_from_checkpoint(CKPT_PATH)
    return model

if __name__=="__main__":

    model = get_model()
    model.eval()

    gaze_seq = []

    vid_path = r"D:\HIP-HOP\DATASET\VIDEOS\P16\V7\P16_V7.mp4"
    vidcap = cv2.VideoCapture(vid_path)

    if not vidcap.isOpened():
        print("Error opening video file")
        exit()

    ret, img = vidcap.read()

    frame_number = 0
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        frame_number+=1

        depth_path = f"D:\HIP-HOP\DATASET\DEPTHS\P16\V7\P16_V7_F{frame_number:03d}_DEPTH.png"
        depth = cv2.imread(depth_path)[:,:,:1]

        rgbd = np.concatenate((img, depth), axis=2)
        rgbd = Image.fromarray(rgbd).convert("RGBA")
        rgbd, _ = make_normalize_transform()(rgbd, None)
        rgbd = rgbd.to("cuda")

        pred = model([rgbd])

        objects = torch.argmax(pred["pred_object_logits"], dim=2)[0]
        gazes = torch.argmax(pred["pred_isgazed_logits"], dim=2)[0]
        
        gazed_index = [objects[x] for x, v in enumerate(gazes) if v == 1]


        if len(gazed_index) == 0:
            gazed_object = 0
        else:
            gazed_object = objects[gazed_index[0]] + 1 # [0] only get the first gazed_object, not chronologically
        # print(gazed_object)
        
        gaze_seq.append(gazed_object)

        del rgbd, pred

        print(f"F{frame_number:03d}: {gazes} -> gazed:{gazed_object}")

    vidcap.release()

    print(gaze_seq)



