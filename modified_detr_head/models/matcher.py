# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_gaze: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        
        Args:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bbox coordinates in the matching cost
            cost_giou: This si the relative weight of the giou loss of the bbox in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_gaze = cost_gaze
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        
        Args:
            outputs: This is a dict
                "pred_object_logits": tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_isgazed_logits": tensor of dim [batch_size, num_queries, num_actions] with the classfiication logits
                "pred_boxes": tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of target dicts (len(targets)==batch_size) where each target dict contains
                "boxes": tensor of dim [num_objects, 4] containing bb coordinates in this format [cx, cy, width, height]
                "objects": tensor of dim [num_objects, ] containing int labels of corresponding bbox in "boxes"
                "isgazed": tensor of dim [num_objects, ] containing 1 or 0 if corresponding object is gazed or not
        
        Returns:
            A List of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_object_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_object_prob = outputs["pred_object_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_isgazed_prob = outputs["pred_isgazed_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_objects = torch.cat([v["objects"] for v in targets])
        tgt_isgazed = torch.cat([v["isgazed"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_objects = -out_object_prob[:, tgt_objects]
        cost_isgazed = -out_isgazed_prob[:, tgt_isgazed]

        # # Compute the isgazed classification cost.
        # alpha = 0.25
        # gamma = 2.0
        # neg_cost_class = (1 - alpha) * (out_isgazed_prob ** gamma) * (-(1 - out_isgazed_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_isgazed_prob) ** gamma) * (-(out_isgazed_prob + 1e-8).log())
        # cost_isgazed = pos_cost_class[:, tgt_isgazed] - neg_cost_class[:, tgt_isgazed]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        # C = l_class + l_gaze + l_bbox + l_giou
        l_cls = (self.cost_class * cost_objects) + (self.cost_gaze * cost_isgazed)
        l_box = (self.cost_bbox * cost_bbox) + (self.cost_giou * cost_giou)
        C =  l_cls  + l_box

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(config):
    return HungarianMatcher(
        cost_class=config["matcher"]["cost_object"],
        cost_gaze=config["matcher"]["cost_isgaze"],
        cost_bbox=config["matcher"]["cost_bbox"],
        cost_giou=config["matcher"]["cost_giou"],
    )
