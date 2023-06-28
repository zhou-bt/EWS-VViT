import os
from collections import OrderedDict

import torch

os.environ['CUDA_VISIBLE_DEVICES'] ='1'

device = torch.device('cuda:{}'.format(0))

IN21K_check = "vit_base_patch16_224_in21k.pth"
k600_check = "TimeSformer_divST_8x32_224_K600.pyth"

IN21K_model_dict = torch.load(IN21K_check, map_location=device)
k600_model_dict = torch.load(k600_check, map_location=device)['model_state']

IN21K_model_dict['time_embed'] = k600_model_dict['model.time_embed']
for k,v in k600_model_dict.items():
    if('temporal' in k):
        IN21K_model_dict[k[6:]] = v

torch.save(IN21K_model_dict, "vit_base_p16_224_IN21K_k600_T8")