import torch
from torchvision.datasets import VisionDataset
from torchvision.ops import box_convert

import os
import json
import pandas as pd
import cv2
from PIL import Image
import numpy as np

from datasets.gom_transforms import make_gom_transforms

OBJECT_NAMES = ['Head', 'Bag', 'Book', 'Bottle', 'Bowl', 'Broom', 'Chair', 'Cup', 'Fruits', 'Laptop',
                'Pillow', 'Racket', 'Rug', 'Sandwich', 'Umbrella', 'Utensils']

class  GOMDataset(VisionDataset):
    """
    Class for making GOMDataset
    Args:
        root (string): Root directory of dataset.
        annFile (string): Path of the annotation file
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        image_set (string): Which set to use. Either "train" or "test" only
        skip_frames (int): Number of frames to be skipped when extracting the video file.
        skip_intents (List [str]): List of intents that not to be included in the dataset.
        participants (List [str]): List of participants to be included in the dataset.
    """
    def __init__(
            self,
            root,
            annFile,
            transform=None, 
            target_transform=None,
            transforms=None,
            image_set='train',
            skip_frames=5,
            skip_intents=[],
            participants=[f"P{i}" for i in range(1,21)]
    ):
        
        assert image_set in ['train', 'test'], image_set
        self.image_set = image_set
        super().__init__(root, transforms, transform, target_transform)

        with open(annFile, 'r') as f:
            json_file = json.load(f)

        json_file = json.loads(open(annFile, 'r').read())
        chosen_set = json_file['hiphop']['gaze'][image_set]

        rows = []
    
        for item in chosen_set:
            if participants: # skipping participants NOT in participants
                if item["video"].split("/")[1] not in participants:
                    continue
            if skip_intents: # skipping intents in skip_intents
                if item["intent"] in skip_intents:
                    continue

            # just making sure gaze_seq ~ bbox ~ depth
            assert len(item['gaze_seq']) == len(item['bbox']), f"Inconsistency in {item['video']}"
            
            # item['video'] = "VIDEOS/P4/V4/P4_V4.mp4"
            P, V = item["video"].split("/")[1:3]
            
            for frame_no in range(0, len(item['gaze_seq']), skip_frames+1):
                frame_dict = {
                    "video_file": item["video"],
                    "frame_file": f"{P}_{V}_F{frame_no+1:03d}.png",
                    "depth_file": item['depth'][frame_no],
                    "object_bbs": item['bbox'][frame_no],
                    "gazed_object": item['gaze_seq'][frame_no],
                    "intent": item['intent'],
                    "participant_no":int(P[1:]),
                    "video_no": int(V[1:]),
                    "frame_number": frame_no,
                }
                rows.append(frame_dict)

        self._walker = pd.DataFrame(rows)

    def __getitem__(self, index: int):
        """
        Returns:
            rgbd (Tensor) : the image input, with depth, a tensor of shape (4, H, W)
            target (dict) :
                "boxes": a tensor of shape (num_objects, 4), bb format: [x y x y] (with Normalize transforms -> cxcywh)
                "objects": a tensor of shape (num_objects, ) containing int labels of corresponding bbox in "boxes"
                "isgazed": a tensor of shape (num_objects, ) containing 0 or 1 if corresponding object is gazed or not
                OPTIONALS
                "video_file": path of the video file. ex: "VIDEOS/P4/V4/P4_V4.mp4"
                "frame_number": frame number
                "intent": intent
                "org_size": original size of the image. Needed for de-normalizing
        """
        frame = self._walker.iloc[index]
        
        # check cache
        cache_file = f"{frame['video_file'].split('/')[-1]}_{frame['frame_number']}.png"
        if os.path.exists(f"dataset_cache/{cache_file}"):
            img = cv2.imread(f"dataset_cache/{cache_file}")
        else:
            vid_path = os.path.join(self.root, frame['video_file'])
            vidcap = cv2.VideoCapture(vid_path)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame['frame_number'])
            ret, img = vidcap.read()
            cv2.imwrite(f"dataset_cache/{cache_file}", img)
            vidcap.release()

        depth_path = os.path.join(self.root, frame['depth_file'])
        depth = cv2.imread(depth_path)[:,:,:1]

        rgbd = np.concatenate((img, depth), axis=2)
        rgbd = Image.fromarray(rgbd).convert("RGBA")

        objects, boxes = zip(*frame["object_bbs"].items())
        boxes = torch.tensor(boxes)
        isGazed = torch.tensor([1 if frame['gazed_object'] == o else 0 for o in objects])
        objects = torch.tensor([OBJECT_NAMES.index(o) for o in objects])

        target = {
            "boxes": box_convert(boxes, 'xywh', 'xyxy'),
            "objects": objects,
            "isgazed": isGazed,
            "video_file": frame['video_file'], 
            "frame_number": frame['frame_number'] + 1,
            "intent": frame["intent"],
            "org_size": rgbd.size
        }

        if self.transforms is not None:
            rgbd, target = self.transforms(rgbd, target, self.image_set)

        return rgbd, target

    def __len__(self) -> int:
        return len(self._walker)

def build_dataset(
    image_set, 
    config, 
    skip_intents = None, 
    participants = [f"P{i}" for i in range(1,21)],
    skip_frames = 5, 
):
    assert image_set in ['train', 'test'], image_set
    annotation_file = './data/gaze_dataset_2.json'

    transforms = make_gom_transforms(image_set, test_scale=-1)

    return GOMDataset(
        root=config["directories"]["root"],
        annFile=annotation_file,
        transforms=transforms,
        image_set=image_set,
        skip_frames=skip_frames,
        skip_intents=skip_intents,
        participants=participants
    )

"""
gaze_dataset.json JSON Structure
hiphop
    gaze
        train
            list of dictionaries
                video: str "VIDEOS/P4/V4/P4_V4.mp4"
                gaze_seq: list of str, corresponding to each frame
                bbox: list of dictionaries, corresponding to each frame
                depth: list of str, corresponding to each frame "DEPTHS/P3/V22/P3_V22_F007_DEPTH.png"


gaze_dataset_2 JSON STRUCTURE
hiphop
    gaze
        train
            listof dictionaries
                video: filename
                gaze_seq:
                bbox:
                depth:
                intent:
        test

"""
