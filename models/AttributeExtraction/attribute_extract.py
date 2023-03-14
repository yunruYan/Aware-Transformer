import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

import lib.utils as utils
from models.basic_model import BasicModel

from models.backbone.swin_transformer_backbone import SwinTransformer as STBackbone


class attribute_extract(BasicModel):
    def __init__(self):
        super(attribute_extract, self).__init__()
        # self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1

        self.backbone = STBackbone(
            img_size=384,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            num_classes=1000
        )
        print('load pretrained weights!')
        self.backbone.load_weights(
            './swin_large_patch4_window12_384_22kto1k_no_head.pth'
        )
        # Freeze parameters
        for _name, _weight in self.backbone.named_parameters():
            _weight.requires_grad = False
        if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
            self.att_embed = nn.Identity()
        else:
            self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)
            )
        self.classify=nn.Sequential(
            nn.Linear(cfg.MODEL.ATT_FEATS_EMBED_DIM,cfg.MODEL.VOCAB_SIZE+1),
            nn.Sigmoid()
        )


    def forward(self,att_feats):
        # att_feats=kwargs[cfg.PARAM.ATT_FEATS]
        att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx = att_feats.mean(1)
        result=self.classify(gx)
        # print(att_feats.shape)
        return result
