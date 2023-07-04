# Brute testing
import yaml
from dataset import get_datamodule, GazeOutputDataset, collate_batch
from models import InFormer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True, type=str)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--gaze_outputs', type=str)
    #checkpoints/trial-v1.ckpt
    return parser.parse_args()

def aprf(preds, targets):
    preds = preds.to("cpu")
    targets = targets.to("cpu")
    a = accuracy_score(y_true=targets, y_pred=preds)
    p = precision_score(y_true=targets, y_pred=preds, average="macro")
    r = recall_score(y_true=targets, y_pred=preds, average="macro")
    f = f1_score(y_true=targets, y_pred=preds, average="macro")
    b = balanced_accuracy_score(y_true=targets, y_pred=preds)

    cm = confusion_matrix(y_true=targets, y_pred=preds)
    print(cm)
    del preds, targets
    return (a, p, r, f, b, cm)

def evaluate(args):

    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)
    config["batch_size"] = 100

    datamodule = get_datamodule(config)
    datamodule.setup("test")

    model = InFormer.load_from_checkpoint(args.ckpt_dir, ).to(args.device)
    model.eval()

    preds = []
    targets = []

    for seq, target in datamodule.test_dataloader():
        seq = seq.to(args.device)
        target = target.to(args.device)
        pred = model(seq).softmax(-1).argmax(-1)
        preds.append(pred.clone())
        targets.append(target.clone())
        del pred

    preds = torch.hstack(preds)
    targets = torch.hstack(targets)

    return aprf(preds, targets)

def eval_overall(args):
    model = InFormer.load_from_checkpoint(args.ckpt_dir,  map_location=torch.device('cpu')).to(args.device)
    model.eval()

    dataset = GazeOutputDataset(
        args.gaze_outputs,
        r"hiphop/intent_ann_new.json"
    )

    dataloader = DataLoader(dataset, batch_size = 10, shuffle=False, collate_fn=collate_batch)

    preds, targets = [], []

    for seq, target in dataloader:
        seq = seq.to(args.device)
        target = target.to(args.device)

        pred = model(seq).softmax(-1).argmax(-1)
        preds.append(pred.clone())
        targets.append(target.clone())
        del pred

    preds = torch.hstack(preds)
    targets = torch.hstack(targets)

    return aprf(preds, targets)


if __name__=="__main__":
    args = get_parser()
    a, p, r, f, b, cm = eval_overall(args)

    print("-----Results------")
    print("Accuracy:", round(a*100, 3))
    print("Precision:", round(p*100, 3))
    print("Recall:", round(r*100, 3))
    print("F1 Score:", round(f*100, 3))
    print("Balanced Accuracy:", round(b*100, 3))

    