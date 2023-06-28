#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224
from project.VTN import vtn_helper


class VTN(nn.Module):
    """
    VTN model builder. It uses ViT-Base as the backbone.

    Daniel Neimark, Omri Bar, Maya Zohar and Dotan Asselmann.
    "Video Transformer Network."
    https://arxiv.org/abs/2102.00719
    """

    def __init__(self, args):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(VTN, self).__init__()
        self._construct_network(args)

    def _construct_network(self, args):
        """
        Builds a VTN model, with a given backbone architecture.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        if args.backbone_name == "VIT":
            self.backbone = vit_base_patch16_224(pretrained=args.pretrained,
                                                 num_classes=0,
                                                 drop_path_rate=args.drop_path_rate,
                                                 drop_rate=args.drop_rate)

        embed_dim = self.backbone.embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.temporal_encoder = vtn_helper.VTNLongformerModel(
            embed_dim=embed_dim,
            max_position_embeddings=args.max_position_embeddings,
            num_attention_heads=args.num_attention_heads,
            num_hidden_layers=args.num_hidden_layers,
            attention_mode=args.attention_mode,
            pad_token_id=args.pad_token_id,
            attention_window=args.attention_window,
            intermediate_size=args.intermediate_size,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            hidden_dropout_prob=args.hidden_dropout_prob)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim,args.mip_dim),
            nn.GELU(),
            nn.Dropout(args.drop_out_rate),
            nn.Linear(args.mip_dim, args.num_classes)
        )

    def forward(self, x, bboxes=None):

        x, position_ids = x

        # spatial backbone
        B, C, F, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * F, C, H, W)
        x = self.backbone(x)   #(8,768)
        x = x.reshape(B, F, -1)

        # temporal encoder (Longformer)
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        cls_atten = torch.ones(1).expand(B, -1).to(x.device)
        attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        attention_mask[:, 0] = 2
        x, attention_mask, position_ids = vtn_helper.pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id)
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1

        # position_ids
        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1

        x = self.temporal_encoder(input_ids=None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=None)
        # MLP head
        x = x["last_hidden_state"]
        x = self.mlp_head(x[:, 0])
        return x