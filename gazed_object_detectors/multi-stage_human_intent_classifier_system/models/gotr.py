"""
GaMer (Gaze Target Transformer) model and criterion class
"""

import torch
import torch.nn.functional as F
from torch import nn
from models.backbone import build_backbone
from models.transformer import build_transformer

from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

class GOTr(nn.Module):
    """ This is the GaMer that performs Head, Object and Gaze Target Detection"""

    def __init__(self,
        backbone,
        transformer,
        num_classes,
        num_actions,
        num_queries,
        aux_loss=False,      
    ):
        """ Initiliazes the model.
        Args:
            backbone: torch module of the backbone to be used
            transformer: torch module of the transformer architecture
            num_classes: number of object classes
            num_queries: maximal number of objects the model can detect in a single image
            aux_loss: True if auxiliary decoding losses (at each decoder layer) are to be used
        """
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.aux_loss = aux_loss

        self.transformer = transformer
        
        # for the outputs
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # self.gazed_embed = nn.Linear(hidden_dim, num_actions + 1)
        self.gazed_embed = MLP(hidden_dim, hidden_dim, num_actions + 1, 3) # make the self.gazed_embed deeper
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # input embeddings for the transformer
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # for fitting features into transformer input_dim
        # positional encoding also in input_dim channels
        self.scene_proj = nn.Conv2d(backbone.scene_num_channels, hidden_dim, kernel_size=1)
        self.head_proj = nn.Conv2d(backbone.head_num_channels, hidden_dim, kernel_size=1)
        
        # backbone includes the positional emnedding
        self.backbone = backbone
    
    def forward(self, samples:NestedTensor, heads:NestedTensor=None):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 4 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        It returns a dict with the following elements:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                            (center_x, center_y, height, width). These values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if isinstance(heads, (list, torch.Tensor)):
            heads = nested_tensor_from_tensor_list(heads)
        
        if list(samples.tensors[0,:,0,0].size())[0] == 3: # check if samples.tensor is RGB  if samples.decompose()[0].shape[1] == 3
            samples.tensors = torch.cat((samples.tensors, torch.zeros_like(samples.tensors[:, :1])), dim=1) # add depth full of zeros

        # out1, out2, pos1, pos2 = self.backbone(samples)
        out1, out2, pos1, pos2 = self.backbone(samples, heads)
        scene_feat, scene_mask = out1[-1].decompose()
        head_feat, head_mask = out2[-1].decompose()

        scene_src = self.scene_proj(scene_feat)
        head_src = self.head_proj(head_feat)

        # features, _, pos, _ = self.backbone(samples)
        #     # features -> output of the backbone
        #     # pos -> output of the positional encoder
        #     ###  See Joiner in backbone.py 
        
        # # need [-1] because of IntermediateLayerGetter, [-1] gives the output of the last layer
        # src, mask = features[-1].decompose() # see decompose in NestedTensor class in util.misc

        # # print(src.shape)
        # # print(self.input_proj(src).shape)

        assert scene_mask is not None and head_mask is not None

        hs = self.transformer(scene_src, head_src, scene_mask, head_mask, self.query_embed.weight, pos1[-1], pos2[-1])

        output_objects = self.class_embed(hs)
        output_isgazed = self.gazed_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            'pred_object_logits': output_objects[-1],
            'pred_isgazed_logits': output_isgazed[-1],
            'pred_boxes': outputs_coord[-1]
            }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_objects, output_isgazed, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, output_isgazed, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_object_logits': a,'pred_isgazed_logits':b, 'pred_boxes': c }
                for a, b, c in zip(outputs_class[:-1], output_isgazed[:-1], outputs_coord[:-1])]

class GOTrNoDepth(nn.Module):
    """ This is the GaMer that performs Head, Object and Gaze Target Detection"""

    def __init__(self,
        backbone,
        transformer,
        num_classes,
        num_actions,
        num_queries,
        aux_loss=False,
    ):
        """ Initiliazes the model.
        Args:
            backbone: torch module of the backbone to be used
            transformer: torch module of the transformer architecture
            num_classes: number of object classes
            num_queries: maximal number of objects the model can detect in a single image
            aux_loss: True if auxiliary decoding losses (at each decoder layer) are to be used
        """
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.aux_loss = aux_loss

        self.transformer = transformer

        # for the outputs
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.gazed_embed = nn.Linear(hidden_dim, num_actions + 1)
        #self.gazed_embed = MLP(hidden_dim, hidden_dim, num_actions + 1, 3) # make the self.gazed_embed deeper
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # input embeddings for the transformer
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # for fitting features into transformer input_dim
        # positional encoding also in input_dim channels
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # backbone includes the positional emnedding
        self.backbone = backbone

    def forward(self, samples:NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 4 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        It returns a dict with the following elements:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                            (center_x, center_y, height, width). These values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        #if list(samples.tensors[0,:,0,0].size())[0] == 3: # check if samples.tensor is RGB  if samples.decompose()[0].shape[1] == 3
        #    samples.tensors = torch.cat((samples.tensors, torch.zeros_like(samples.tensors[:, :1])), dim=1) # add depth full of zeros

        features, pos = self.backbone(samples)
            # features -> output of the backbone
            # pos -> output of the positional encoder
            ###  See Joiner in backbone.py

        # need [-1] because of IntermediateLayerGetter, [-1] gives the output of the last layer
        src, mask = features[-1].decompose() # see decompose in NestedTensor class in util.misc
        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        output_objects = self.class_embed(hs)
        output_isgazed = self.gazed_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            'pred_object_logits': output_objects[-1],
            'pred_isgazed_logits': output_isgazed[-1],
            'pred_boxes': outputs_coord[-1]
            }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_objects, output_isgazed, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, output_isgazed, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_object_logits': a,'pred_isgazed_logits':b, 'pred_boxes': c }
                for a, b, c in zip(outputs_class[:-1], output_isgazed[:-1], outputs_coord[:-1])]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_model(config, pre_trained=True, depth=True):

    backbone = build_backbone(config)
    transformer = build_transformer(config)

    if depth:
        model = GOTr(
            backbone,
            transformer,
            config["num_classes"],
            config["num_actions"],
            config["num_queries"],
            config["model"]["aux_loss"]
        )
    else:
        model = GOTrNoDepth(
            backbone,
            transformer,
            config["num_classes"],
            config["num_actions"],
            config["num_queries"],
            config["model"]["aux_loss"]
    )

    if pre_trained:
        ckpt_path = "" #r"checkpoints/best_small_scale.ckpt"
        if ckpt_path.split(".")[-1] == "ckpt":
            weights = torch.load(ckpt_path)["state_dict"]
            weights = {".".join(name.split(".")[1:]): value for name,value in weights.items() if "model" in name}
            model.load_state_dict(weights, strict=True)
        else:
            state_dict = torch.load('data/pretrained_weights/detr-r50-e632da11.pth')
            weights = state_dict["model"]

            weights.pop("class_embed.bias")
            weights.pop("class_embed.weight")
            weights.pop("query_embed.weight")
            conv1_weights = weights.pop("backbone.0.body.conv1.weight")

            # [print(name) for name, value in weights.items() if "backbone" in name]

            model.load_state_dict(weights, strict=False)

            # load backbone.0.body.conv1.weight
            weight = conv1_weights.clone()
            with torch.no_grad():
                model.backbone[0].body.conv1.weight[:, :3] = weight
                model.backbone[0].body.conv1.weight[:, 3] = model.backbone[0].body.conv1.weight.mean(1)

    return model
