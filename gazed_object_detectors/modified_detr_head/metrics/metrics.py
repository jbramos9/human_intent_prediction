from typing import Any
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision
from torch import masked_select, stack, ones, arange, zeros, where, tensor, zeros_like
import torch

from datasets.gom import OBJECT_NAMES

OBJECTS = OBJECT_NAMES + ["None"]

class BBoxMetrics(MeanAveragePrecision):
    """
    Computes the mean average precision (mAP) for object detection.
    """

    def __init__(self, prefix="", **kwargs: Any) -> None:
        super().__init__(box_format='cxcywh', **kwargs)

        self.prefix = prefix + "_" if prefix else prefix

    def _format(self, outputs, targets):
        queries = outputs['pred_boxes'][0].shape[0]
        
        outputs = [
            {
                "boxes": outputs['pred_boxes'][index],
                "scores": outputs['pred_object_logits'][index][arange(queries), :-1].softmax(dim=1).max(-1).values,
                "labels": outputs['pred_object_logits'][index][arange(queries), :-1].argmax(dim=1)
            }
            for index in range(len(targets))
        ]

        targets = [
            {
                "boxes": batch['boxes'],
                "labels": batch['objects']
            }
            for batch in targets
        ]

        return outputs, targets

    def update(self, outputs, targets) -> None:
        outputs, targets = self._format(outputs, targets)
        super().update(outputs, targets)

    def compute(self) -> dict:
        return super().compute()["map"]


class GazeMetrics(Metric):
    """
    Computes the weighted multiclass `accuracy`, `precision`, and `recall`.
    for gaze classification.
    """

    # Fixed number of classes
    # num_classes = len(OBJECTS)

    def __init__(self, prefix, num_classes, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes + 1

        self.prefix = prefix + "_" if prefix else prefix

        self.add_state(
            "true_positives",
            default=zeros(self.num_classes).to(self.device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_positives",
            default=zeros(self.num_classes).to(self.device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_negatives",
            default=zeros(self.num_classes).to(self.device),
            dist_reduce_fx="sum",
        )

    def _format(self, outputs, targets):
        
        scores = outputs["pred_isgazed_logits"].softmax(-1)
        isgazed = outputs["pred_isgazed_logits"].argmax(-1)

        mask = where(isgazed == 1, isgazed, zeros_like(isgazed))
        isgazed_scores = scores[:, :, 1] * mask

        objects = outputs["pred_object_logits"].argmax(-1)
        gazed_object = []
        for gaze, obj in zip(isgazed_scores, objects):
            if sum(gaze) == 0:
                gazed_object.append(tensor(self.num_classes-1).to(self.device))
            else:
                gazed_object_idx = gaze.argmax()
                gazed_object.append(obj[gazed_object_idx])
        gazed_object = stack(gazed_object)

        # Mask out non-gazed queries
        gazes = [
            masked_select(target["objects"], target["isgazed"] > 0)
            for target in targets
        ]
        # Add default gazed class ID 16 to account for null class
        gazes = [
            gaze if gaze.nelement() else self.num_classes-1 * ones(1).to(self.device)
            for gaze in gazes
        ]

        targets = stack(gazes).squeeze(1)

        return gazed_object, targets

    @property
    def precision(self):
        return self.true_positives / (
            self.true_positives + self.false_positives + 1e-8
        ).to(self.device)

    @property
    def recall(self):
        return self.true_positives / (
            self.true_positives + self.false_negatives + 1e-8
        ).to(self.device)

    @property
    def accuracy(self):
        return (self.recall + self.precision) / 2

    @property
    def class_precision(self):
        return {
            f"{self.prefix}{name}_precision": self.precision[index]
            for index, name in enumerate(OBJECTS)
        }

    @property
    def class_recall(self):
        return {
            f"{self.prefix}{name}_recall": self.recall[index]
            for index, name in enumerate(OBJECTS)
        }

    @property
    def class_accuracy(self):
        return {
            f"{self.prefix}{name}_accuracy": self.accuracy[index]
            for index, name in enumerate(OBJECTS)
        }

    @property
    def model_precision(self):
        return self.precision.mean()

    @property
    def model_recall(self):
        return self.recall.mean()

    @property
    def model_accuracy(self):
        return self.accuracy.mean()

    def update(self, outputs, targets):
        outputs, targets = self._format(outputs, targets)

        for class_type in range(self.num_classes):
            class_type_outputs = outputs == class_type
            class_type_targets = targets == class_type

            self.true_positives[class_type] += (
                (class_type_outputs & class_type_targets).sum().float()
            )
            self.false_positives[class_type] += (
                (class_type_outputs & ~class_type_targets).sum().float()
            )
            self.false_negatives[class_type] += (
                (~class_type_outputs & class_type_targets).sum().float()
            )

    def compute(self):
        return self.model_accuracy
