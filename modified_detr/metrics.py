from typing import Any, Optional
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, 
    BinaryAccuracy, BinaryPrecision, BinaryRecall)
from torch import masked_select, stack, ones, arange
from torchmetrics import MetricCollection

from datasets.gom import OBJECT_NAMES

OBJECTS = OBJECT_NAMES + ["None"]

class DetectionMAP(MeanAveragePrecision):
    """
    Computes the mean average precision (mAP) for object detection.
    """
        
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(box_format='cxcywh', class_metrics=True, **kwargs)    
        
    def _format(self, outputs, targets):
                
        print(outputs['pred_object_logits'])

        outputs = [
            {
                "boxes": outputs['pred_boxes'][index].to(self.device),
                "scores": outputs['pred_object_logits'][index].max(dim=1)[0],
                "labels": outputs['pred_object_logits'][index][arange(len(OBJECT_NAMES)), :-1].argmax(dim=1).to(self.device)
            }
            for index in range(len(targets))
        ]

        print(outputs["labels"])

        targets = [
            { 
                "boxes": batch['boxes'].to(self.device), 
                "labels": batch['objects'].to(self.device)
            }
            for batch in targets
        ]
                
        return outputs, targets
    
    def update(self, outputs, targets) -> None:
        outputs, targets = self._format(outputs, targets)
        super().update(outputs, targets)
        
    def compute(self) -> dict:
        mean_average_precision = super().compute()
        return {
            "mean_average_precisions": mean_average_precision["map"], 
            "average_precisions": mean_average_precision["map_per_class"]
        }

    @staticmethod
    def unpack(metrics: dict, prefix=""):
        """Returns all of the metrics into a flat dictionary. """

        multiclass_maps = {
            f"{prefix}_mAP_{object_type}": map
            for map, object_type in zip(metrics["average_precisions"], OBJECTS)   
        }

        multiclass_maps[f"{prefix}_mAP"] = metrics["mean_average_precisions"]

        return multiclass_maps

class GazeMetrics(MetricCollection):
    """
        torchmetrics.MetricCollection computing the following metrics:
        
        Gaze Classification:
            - The Accuracy, Precision, and Recall Score (weighted)
              for every object class and the entirety of the model
    """
    
    num_classes = len(OBJECT_NAMES)+1

    def __init__(self, prefix="", **kwargs: Any) -> None:

        # Generate recall, precision, accuracy metrics per class
        metrics_per_class = MetricCollection([
            GazeMetric(name, **kwargs) for name in OBJECTS
        ])

        # Generate recall, precision, accuracy for entire model
        metrics_model = MetricCollection([
            MulticlassRecall(num_classes=self.num_classes, average="weighted", **kwargs),
            MulticlassPrecision(num_classes=self.num_classes, average="weighted", **kwargs),
            MulticlassAccuracy(num_classes=self.num_classes, average="weighted", **kwargs),
        ], prefix="_model_", **kwargs)

        super().__init__(
            MetricCollection([metrics_per_class, metrics_model]),
            prefix=prefix, **kwargs)

    def _format(self, outputs, targets):   
            
        # Determine most likely gazed query
        is_gazed_index = 1
        max_indices = outputs['pred_isgazed_logits'][:, :, is_gazed_index]
        max_indices = max_indices.argmax(1)
        
        # Determine most likely object id for each gazed query
        # Note: Assumes output
        batches = arange(max_indices.shape[0])
        outputs = outputs['pred_object_logits'][batches, max_indices, :]

        # Mask out non-gazed queries
        gazes = [masked_select(target['objects'], target['isgazed'] > 0) 
                 for target in targets]
        # Add default gazed class ID 16 to account for null class
        gazes = [gaze if gaze.nelement() else 16*ones(1).to(self.device) 
                 for gaze in gazes]

        targets = stack(gazes).squeeze(1) 
        
        return outputs, targets

    def update(self, outputs, targets):
        outputs, targets = self._format(outputs, targets)
        
        super().update(outputs, targets)

class GazeMetric(MetricCollection):
    """
        torchmetrics.MetricCollection computing the following metrics:
        
        Gaze Classification:
            - The Accuracy, Precision, and Recall Score
              for a single object class
    """
    
    def __init__(self, name, **kwargs: Any) -> None:
        super().__init__(
            [
                BinaryAccuracy(**kwargs),
                BinaryPrecision(**kwargs),
                BinaryRecall(**kwargs),
            ], 
            prefix=f"_{name}_", 
            **kwargs
        )

        self.name = name
        self.index = OBJECTS.find(name)
        
    def _format(self, outputs, targets):   
        """
            Convert arguments into binary with respect to self.name

            Args:
                outputs (torch.tensor): Shape (N, )
                targets (torch.tensor): Shape (N, len(OBJECTS))

            Returns
                outputs (torch.tensor): Shape (N, )
                targets (torch.tensor): Shape (N, )
        """    
        
        # Determine most likely gazed query
        is_gazed_index = 1
        max_indices = outputs['pred_isgazed_logits'][:, :, is_gazed_index]
        max_indices = max_indices.argmax(1)
        
        # Determine most likely object id for each gazed query
        # Note: Assumes output
        batches = arange(max_indices.shape[0])
        outputs = outputs['pred_object_logits'][batches, max_indices, :]

        # Mask out non-gazed queries
        gazes = [masked_select(target['objects'], target['isgazed'] > 0) 
                 for target in targets]
        # Add default gazed class ID 16 to account for null class
        gazes = [gaze if gaze.nelement() else 16*ones(1).to(self.device) 
                 for gaze in gazes]

        targets = stack(gazes).squeeze(1)

        outputs = outputs.argmax(dim=1) == self.index
        targets = targets == self.index
        
        return outputs, targets
        
    def update(self, outputs, targets) -> None:
        outputs, targets = self._format(outputs, targets)

        super().update(outputs, targets)

class GazeMultiClassAccuracy(MulticlassAccuracy):
    
    def __init__(self, num_classes: int = len(OBJECT_NAMES)+1, ) -> None:
        super().__init__(num_classes, average='weighted')
        
    def update(self, outputs, targets)-> None:

        # Determine most likely gazed query for each batch
        max_indices = outputs['pred_isgazed_logits'][:, :, 1].argmax(1)
        
        # Determine most likely object id for each gazed query
        #outputs = outputs['pred_object_logits'][:, max_indices[0]]
        #outputs = torch.index_select(outputs['pred_object_logits'], dim=1, index=max_indices)
        batches = arange(max_indices.shape[0])
        outputs = outputs['pred_object_logits'][batches, max_indices, :]

        # Mask out non-gazed queries 
        gazes = [masked_select(target['objects'], target['isgazed'] > 0) 
                 for target in targets]
        # Add default gazed class ID 16 to account for null class
        gazes = [gaze if gaze.nelement() else 16*ones(1).to(self.device) 
                 for gaze in gazes]
        targets = stack(gazes).squeeze(1) 
                
        super().update(outputs, targets)

