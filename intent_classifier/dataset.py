import os
import json
import pandas as pd
from typing import List, Tuple

import torch
from torch.utils.data import random_split
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.dataset_keys import *
import utils.gaze_transforms
from utils.gaze_transforms import GazeTransform


class INTENTDataset(Dataset):

    def __init__(self, data_dir, subset:str, max_length:int, transforms:GazeTransform=None) -> None:
        super().__init__()

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from " + "{'training', 'validation', 'testing'}."
        )

        self.max_len = max_length
        self.transforms = transforms

        self.data_dir = os.path.join(data_dir, 'intent_dataset_new.json')
        with open(self.data_dir, 'r') as f:
            data = json.load(f)


        if subset=="training":
            dset = data['hiphop']['intentions']['train']
        else:
            dset = data['hiphop']['intentions']['test']
        
        rows = []
        for intent in dset:
            seqs = dset[intent]
            for seq in seqs:
                rows.append([intent, seq])
            
        self._walker = pd.DataFrame(rows, columns=[INTENT, GAZE_SEQ])


    def __len__(self) -> int:
        return len(self._walker)
    
    def __getitem__(self, n) -> Tuple[List[str], str]:
        intent = self._walker.loc[n, INTENT]
        seq = self._walker.loc[n, GAZE_SEQ]

        if self.transforms:
            for t in self.transforms:
                seq = t(seq)
        
        if len(seq) < self.max_len:
            pad_len = self.max_len - len(seq)
            seq = seq + [GAZED_OBJECT_MAPPING_R[0.0] for _ in range(pad_len)]
        elif len(seq) > self.max_len:
            seq = seq[:self.max_len]
        
        return seq, intent

class GazeOutputDataset(Dataset):
    def __init__(self, csvFile, targetFile, max_length=250) -> None:
        super().__init__()

        raw = pd.read_csv(csvFile, header=None)
        raw.columns = ["video", "frame", "gazed_object"]
        self.max_len = max_length
        
        raw = raw.replace(np.nan, "none")
        videos = raw['video'].unique()

        with open(targetFile, 'r') as f:
            intent_dict = json.load(f)["hiphop"]["videos"]

        data = []

        gazed_object_mapping = ['Head', 'Bag', 'Book', 'Bottle', 'Bowl', 'Broom', 'Chair', 'Cup', 'Fruits', 'Laptop',
                'Pillow', 'Racket', 'Rug', 'Sandwich', 'Umbrella', 'Utensils', 'none']
        
        for video in videos:
            # if any([p in video for p in [f"P{i}_" for i in [1, 2, 3, 20]]]): continue
            vid_pd = raw[raw['video'] == video]
            no_frames = len(vid_pd)
            gaze_seq = ["none" for _ in range(len(vid_pd))]
            for _, row in vid_pd.iterrows():
                gaze_seq[row["frame"] - 1] = gazed_object_mapping[row["gazed_object"]]
            vid_file = video.split("/")[-1]
            data.append([vid_file, gaze_seq, intent_dict[vid_file]["intent"]])

        self._walker = pd.DataFrame(data, columns=["video", "gaze_seq", "intent"])
    
    def __getitem__(self, index) -> tuple:
        intent = self._walker.loc[index, "intent"]
        seq = self._walker.loc[index, "gaze_seq"]
        
        if len(seq) < self.max_len:
            pad_len = self.max_len - len(seq)
            seq = seq + ["none" for _ in range(pad_len)]
        elif len(seq) > self.max_len:
            seq = seq[:self.max_len]
        
        return seq, intent

    def __len__(self):
        return len(self._walker)

def collate_batch(batch):
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [torch.tensor(
                    [GAZED_OBJECT_MAPPING[o] for o in _src],
                    dtype=torch.int64
                )
            ],
            0,
        )
        processed_tgt = torch.tensor(
            INTENTIONS_MAPPING[_tgt],
            dtype=torch.int64
        )
        src_list.append(processed_src)
        tgt_list.append(processed_tgt)
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

class INTENTDataModule(pl.LightningDataModule):
    def __init__( self,
            data_path: str = "dataset",
            input_size: int = 250,
            batch_size: int = 100,
            num_workers: int = 0,
            transforms: List[GazeTransform] = None
    ):
        super().__init__()
        self.data_path = data_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augments = transforms
    
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train = INTENTDataset(self.data_path, "training", self.input_size, self.augments)
            self.val = INTENTDataset(self.data_path, "testing", self.input_size, self.augments)
            # ds = INTENTDataset(self.data_path, "training", self.input_size, self.augments)
            # self.train, self.val = random_split(ds, [700, 100])
            
        if stage == "test":
            self.test = INTENTDataset(self.data_path, "testing", self.input_size)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=collate_batch
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_batch
        )

def _get_transforms(gazetransforms) -> List[GazeTransform]:
    res = []
    for t in gazetransforms:
        gazetransform = getattr(utils.gaze_transforms, t)
        p = gazetransforms[t]
        res.append(gazetransform(p))
    return res

def get_datamodule(config) -> pl.LightningDataModule:
    return INTENTDataModule (
        data_path = config['data_dir'],
        input_size = config['input_size'],
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        transforms = _get_transforms(config['data_augmentation'])
    )

