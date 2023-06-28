import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import cv2
import random
from vit import vit_base_patch16_224_in21k as base
# from timm.models import vit_base_patch16_224_in21k as base
from full_attention_cls_abs_windos_SpaTempInter_no_random import vit_base_patch16_224_in21k as plus
import matplotlib.pyplot as plt

class FullModel(nn.Module):
    def __init__(self, args, num_features=768):
        # self.energy_thr = energy_thr
        super(FullModel, self).__init__()
        self.base = base(batch_size=args.batch_size, num_segments=args.num_segments, depth=12, num_classes=args.num_classes, has_logits=False, )
        self.plus = plus(batch_size=args.batch_size, num_segments=args.num_segments, num_classes=args.num_classes,
                         window_size=args.window_size, has_logits=False)
        self.num_features = num_features

        self.sigmoid = nn.Sigmoid()
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def featmap_norm(self, feat_map):
        feat_map = feat_map.sum(dim=1).unsqueeze(dim=1)
        feat_map = F.upsample(feat_map, size=(25, 25), mode='bilinear', align_corners = True).squeeze(dim=1)
        feat_b, feat_h, feat_w = feat_map.size(0), feat_map.size(1), feat_map.size(2)

        feat_map = feat_map.view(feat_map.size(0), -1)
        feat_map_max, _ = torch.max(feat_map, dim=1)
        feat_map_min, _ = torch.min(feat_map, dim=1)
        feat_map_max = feat_map_max.view(feat_b, 1)
        feat_map_min = feat_map_min.view(feat_b, 1)
        feat_map = (feat_map - feat_map_min) / (feat_map_max - feat_map_min)
        feat_map = feat_map.view(feat_b, 1, feat_h, feat_w)
        return feat_map

    def bounding_box(self, feat_map, is_training):
        feat_map = feat_map.squeeze(dim=1)
        feat_b = feat_map.size(0)
        feat_vec_h = feat_map.sum(dim=2)
        feat_vec_w = feat_map.sum(dim=1)

        if not is_training:
            h_str, h_end = self.structured_searching(feat_vec_h)
            w_str, w_end = self.structured_searching(feat_vec_w)
        else:
            h_str = np.zeros(shape=feat_b, dtype=float)
            h_end = np.zeros(shape=feat_b, dtype=float)
            w_str = np.zeros(shape=feat_b, dtype=float)
            w_end = np.zeros(shape=feat_b, dtype=float)
            for i in range(feat_b):
                h_str[i] = random.uniform(0, 1-0.5)
                h_end[i] = h_str[i] + 0.5
                w_str[i] = random.uniform(0, 1-0.5)
                w_end[i] = w_str[i] + 0.5

        return [h_str, h_end, w_str, w_end]

    def img_sampling(self, img, h_str, h_end, w_str, w_end):
        img_b, img_c, img_h, img_w = img.size()
        img_sampled = torch.zeros(img_b, img_c, int(img_h/2), int(img_w/2)).cuda()
        h_str = (h_str*img_h).astype(int)
        h_end = (h_end*img_h).astype(int)
        w_str = (w_str*img_w).astype(int)
        w_end = (w_end*img_w).astype(int)
        for i in range(img_b):
            img_sampled_i = img[i, :, h_str[i]:h_end[i], w_str[i]:w_end[i]].unsqueeze(dim=0)
            img_sampled[i, :] = F.upsample(img_sampled_i, size=(int(img_h/2), int(img_w/2)), mode='bilinear', align_corners = True)

        return img_sampled

    def forward(self, x, is_training=True):
        bt = x.size(0)
        heat_map, [h_str, h_end, w_str, w_end] = None, [None, None, None, None]

        with torch.no_grad():
            # x [bt,c,h,w]
            feat_map_1 = self.base(x)   #[bt,768,14,14]
            # logits_s1 = self.fc_s1(self.pooling(feat_map_1).view(bt, -1))
            heat_map = self.featmap_norm(feat_map_1)    #[bt,1,25,25]
            h_str, h_end, w_str, w_end = self.bounding_box(heat_map, is_training)
            img_s2 = self.img_sampling(x, h_str, h_end, w_str, w_end)
            #   plt.imshow(img_s2[26,:,:].permute(1,2,0).cpu())
        x = self.plus(x, h_str, h_end, w_str, w_end)

