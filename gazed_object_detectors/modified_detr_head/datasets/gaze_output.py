from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd
import json
import numpy as np

from intent_dataset_keys import *

class GazeOutputDataset(Dataset):
    def __init__(self, csvFile, targetFile) -> None:
        super().__init__()

        raw = pd.read_csv(csvFile, header=None)
        raw.columns = ["video", "frame", "gazed_object"]
        
        raw = raw.replace(np.nan, "none")
        videos = raw['video'].unique()

        with open(targetFile, 'r') as f:
            intent_dict = json.load(f)["hiphop"]["videos"]

        data = []
        
        for video in videos:
            vid_pd = raw[raw['video'] == video]
            no_frames = len(vid_pd)
            gaze_seq = ["none" for _ in range(len(vid_pd))]
            for _, row in vid_pd.iterrows():
                gaze_seq[row["frame"] - 1] = row["gazed_object"]
            vid_file = video.split("/")[-1]
            data.append([vid_file, gaze_seq, intent_dict[vid_file]["intent"]])

        self._walker = pd.DataFrame(data, columns=["video", "gaze_seq", "intent"])
    
    def __getitem__(self, index) -> tuple:
        intent = self._walker.loc[index, "intent"]
        seq = self._walker.loc[index, "gaze_seq"]
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

if __name__=="__main__":

    ds = GazeOutputDataset("fewer_skips_gaze_output.csv", "data\intent_ann_new.json")

    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch)

    for batch in dl:
        print(batch)
        break