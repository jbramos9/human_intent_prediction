import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from .matcher import build_matcher

class SetCriterion(nn.Module):
    """
    This class computes the loss for GaMer
    The process involves two steps:
        1) Computation of the Hungarian assignment between ground truth boxes
        2) Supervise each pair of matched ground-truth / prediction
    """

    def __init__( self, num_classes:int, num_actions:int, matcher:nn.Module, weight_dict:dict, eos_coef:float, losses,):
        """Create the criterion.
        Args:
            num_classes: number of object categories (excluding the no_object category)
            num_actions: #TODO: rename this
            matcher: module able to compute the matching between targets and predictions
            weight_dict: dict containing
                name of the losses: relative weight
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_actions = num_actions
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        object_empty_weight = torch.ones(self.num_classes + 1)
        gazed_empty_weight = torch.ones(self.num_actions + 1)
        object_empty_weight[-1] = self.eos_coef
        gazed_empty_weight[-1] = self.eos_coef
        self.register_buffer('object_empty_weight', object_empty_weight)
        self.register_buffer('gazed_empty_weight', gazed_empty_weight)
    
    def forward(self, outputs, targets):
        """
        This performs the loss computation
        Args:
            outputs: output from the GaMer, 
            targets: list of (target) dicts, such that len(targets) == batch_size
                ``` From hiphop.py
                HIPHOPDataset.__getitem__() target contains
                    "boxes": a tensor of shape (num_objects, 4), bb format: [cx, cy, width, height]
                    "objects": a tensor of shape (num_objects, ) containing int labels of corresponding bbox in "boxes"
                    "isGazed": a tensor of shape (num_objects, ) containing 0 or 1 if corresponding object is gazed or not
                ```
        """
        # get the last layer outputs (which are outputs with no 'aux_outputs')
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["objects"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        #Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses
        # contains  **see loss_map in self.get_loss**
            # loss_object, loss_isgazed
            # loss_bbox, loss_giou
            # cardinality loss
            # auxiliary losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Negative Log Likelihood)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_object_logits' in outputs
        assert 'pred_isgazed_logits' in outputs
        object_logits = outputs['pred_object_logits']
        isgazed_logits = outputs['pred_isgazed_logits']

        idx = self._get_src_permutation_idx(indices)
        target_objects_o = torch.cat([t["objects"][J] for t, (_, J) in zip(targets, indices)])
        target_objects = torch.full(object_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=object_logits.device)
        target_objects[idx] = target_objects_o.long()

        target_isgazed_o = torch.cat([t["isgazed"][J] for t, (_, J) in zip(targets, indices)])
        target_isgazed= torch.full(isgazed_logits.shape[:2], self.num_actions,
                                    dtype=torch.int64, device=isgazed_logits.device)
        target_isgazed[idx] = target_isgazed_o
        # target_isgazed_onehot = torch.zeros([isgazed_logits.shape[0], isgazed_logits.shape[1], isgazed_logits.shape[2] + 1],
        #                                     dtype=isgazed_logits.dtype, layout=isgazed_logits.layout, device=isgazed_logits.device)
        # target_isgazed_onehot.scatter_(2, target_isgazed.unsqueeze(-1), 1)

        # target_isgazed_onehot = target_isgazed_onehot[:,:,:-1]
        

        object_ce = F.cross_entropy(object_logits.transpose(1, 2), target_objects, self.object_empty_weight)
        isgazed_ce = F.cross_entropy(isgazed_logits.transpose(1, 2), target_isgazed, self.gazed_empty_weight)
        # isgazed_ce = sigmoid_focal_loss(isgazed_logits, target_isgazed_onehot, num_boxes, 0.25, gamma=2) * isgazed_logits.shape[1]

        losses = {
            'loss_object': object_ce,
            'loss_isgazed': isgazed_ce
        }

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['object_class_error'] = 100 - accuracy(object_logits[idx], target_objects_o)[0]
            losses['is_gazed_class_error'] = 100 - accuracy(isgazed_logits[idx], target_isgazed_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_isgazed_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["isgazed"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
   
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def get_criterion(config):

    matcher = build_matcher(config)
    weight_dict = {
        'loss_object': config['loss_coefs']['class_loss_coef'],
        'loss_isgazed': config['loss_coefs']['isgazed_loss_coef'],
        'loss_bbox': config['loss_coefs']['bbox_loss_coef'],
        'loss_giou': config['loss_coefs']['giou_loss_coef']
    }

    # TODO this is a hack
    if config['model']['aux_loss']:
        aux_weight_dict = {}
        for i in range(config['model']['transformer']['num_decoder_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(
        num_classes=config['num_classes'],
        num_actions=config['num_actions'],
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config['loss_coefs']['eos_coef'],
        losses=losses
    )

    return criterion