import torch
import cv2
import pandas as pd
import os
import numpy as np
from PIL import Image
import json

from torchvision.ops import box_convert

from .gotr import build_model as build_gotr
from .InFormer import get_model as build_informer
from .gom_transforms import make_gom_transforms
from util.misc import get_gazed_objects
from visualization import pred_vis


OBJECT_NAMES = ['Head', 'Bag', 'Book', 'Bottle', 'Bowl', 'Broom', 'Chair', 'Cup', 'Fruits', 'Laptop',
                'Pillow', 'Racket', 'Rug', 'Sandwich', 'Umbrella', 'Utensils']

GAZED_OBJECT = ['Head', 'Bag', 'Book', 'Bottle', 'Bowl', 'Broom', 'Chair', 'Cup', 'Fruits', 'Laptop',
        'Pillow', 'Racket', 'Rug', 'Sandwich', 'Umbrella', 'Utensils', 'none']

class MHICS():

    def __init__(self, gaze_config, gaze_ckpt, intent_config, intent_ckpt, head_config=None, head_ckpt=None, device="cpu") -> None:

        self.device = device

        self.gaze_model = build_gotr(gaze_config, pre_trained=False, depth=True).to(device)
        gaze_weights = {name[6:]: weight for name, weight in torch.load(gaze_ckpt)["state_dict"].items() if "model" in name}
        self.gaze_model.load_state_dict(gaze_weights, strict=True)

        self.intent_model = build_informer(intent_config).to(device)
        informer_weights = torch.load(intent_ckpt)["state_dict"]
        self.intent_model.load_state_dict(informer_weights, strict=True)
        # print(self.intent_model)
    
        if head_config and head_ckpt:
            self.head_detector = build_gotr(head_config, pre_trained=False, depth=False).to(device)
        else:
            self._head_box = pd.read_csv(r"data\head_bboxes.csv", header=None)
            self._head_box.columns = ["video_file", "frame_number", "cx", "cy", "w", "h"]

        annFile = r"data\gaze_dataset_2.json"
        self._gaze_target = _prep_gaze_target(annFile)
        self.max_len = 280

    def __call__(self, video_file, save=False):

        self.gaze_model.eval()
        self.intent_model.eval()
        
        root = r"D:\Dataset\hiphop"
        video_path = os.path.join(root, video_file)
        video = cv2.VideoCapture(video_path)

        # Check if the video file was successfully opened
        if not video.isOpened():
            print("Failed to open the video file.")
            exit()

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the output video file path and codec
        output_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        gazed_seq = []
        frame_number = 0
        while True:
            # Read the next frame
            ret, img = video.read()

            # Check if the frame was successfully read
            if not ret: break
            else: frame_number += 1

            rgbd, head, target = self._prep_gaze_input(root, video_file, frame_number, img)
            gaze_output = self.gaze_model(rgbd, head)

            pred_vis(img, gaze_output, target, writer=video_writer)

            gazed_object = GAZED_OBJECT[get_gazed_objects(gaze_output, None, self.device).item()]
            gazed_seq.append(gazed_object)

            del gaze_output
        

        # Release the video file and close windows
        video.release()
        video_writer.release()
        cv2.destroyAllWindows()

        gazed_seq = self._prep_intent_input(gazed_seq)

        intent_logits = self.intent_model(gazed_seq)
        return intent_logits.argmax(-1)

    def _prep_gaze_input(self, root, video_file, frame_number, img):

        frame = self._gaze_target[(self._gaze_target['video_file'] == video_file) & (self._gaze_target['frame_number'] == frame_number)].to_dict(orient='records')[0]

        objects, boxes = zip(*frame["object_bbs"].items())
        boxes = torch.tensor(boxes)
        isGazed = torch.tensor([1 if frame['gazed_object'] == o else 0 for o in objects])
        objects = torch.tensor([OBJECT_NAMES.index(o) for o in objects])

        target = {
            "boxes": box_convert(boxes, 'xywh', 'xyxy'),
            "objects": objects,
            "isgazed": isGazed,
            "video_file": frame['video_file'], 
            "frame_number": frame['frame_number'],
            "intent": frame["intent"],
            "org_size": img.size
        }
        target = [target, ]

        depth_path = os.path.join(root, "DEPTHS", f"{video_file[7:-4]}_F{frame_number:03d}_DEPTH.png")
        depth = cv2.imread(depth_path)[:,:,:1]

        rgbd = np.concatenate((img, depth), axis=2)
        rgbd = Image.fromarray(rgbd).convert("RGBA")
        gom_transforms = make_gom_transforms("test")
        rgbd, target = gom_transforms(rgbd, target)
        rgbd = rgbd.to(self.device)

        head_row = self._head_box[(self._head_box['video_file'] == video_file) & (self._head_box['frame_number'] == frame_number)]
        cx = head_row['cx'].values[0]
        cy = head_row['cy'].values[0]
        w = head_row['w'].values[0]
        h = head_row['h'].values[0]

        _, height, width  = rgbd.shape
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)

        head = rgbd[:3, y1:y2, x1:x2]

        rgbd = torch.unsqueeze(rgbd, 0)
        head = torch.unsqueeze(head, 0)

        return rgbd, head, target

    def _prep_intent_input(self, gazed_seq):
        if len(gazed_seq) < self.max_len:
            pad_len = self.max_len - len(gazed_seq)
            gazed_seq = gazed_seq + ["none" for _ in range(pad_len)]
        elif len(gazed_seq) > self.max_len:
            gazed_seq = gazed_seq[:self.max_len]
        
        processed_src = torch.cat(
                [torch.tensor(
                        [GAZED_OBJECT.index(o) for o in gazed_seq],
                        dtype=torch.int64, device=self.device
                    )
                ],
                0,
            )

        return processed_src

def _prep_gaze_target(annFile, image_set="test"):
    with open(annFile, 'r') as f:
        json_file = json.load(f)

    json_file = json.loads(open(annFile, 'r').read())
    chosen_set = json_file['hiphop']['gaze'][image_set]

    rows = []

    for item in chosen_set:
        # if participants: # skipping participants NOT in participants
        #     if item["video"].split("/")[1] not in participants:
        #         continue
        # if skip_intents: # skipping intents in skip_intents
        #     if item["intent"] in skip_intents:
        #         continue

        # just making sure gaze_seq ~ bbox ~ depth
        assert len(item['gaze_seq']) == len(item['bbox']), f"Inconsistency in {item['video']}"
        
        # item['video'] = "VIDEOS/P4/V4/P4_V4.mp4"
        P, V = item["video"].split("/")[1:3]
        
        for frame_no in range(0, len(item['gaze_seq'])):
            frame_dict = {
                "video_file": item["video"],
                "frame_file": f"{P}_{V}_F{frame_no+1:03d}.png",
                "depth_file": item['depth'][frame_no],
                "object_bbs": item['bbox'][frame_no],
                "gazed_object": item['gaze_seq'][frame_no],
                "intent": item['intent'],
                "participant_no":int(P[1:]),
                "video_no": int(V[1:]),
                "frame_number": frame_no+1,
            }
            rows.append(frame_dict)

    return pd.DataFrame(rows)
