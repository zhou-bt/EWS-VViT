import os
from collections import OrderedDict

import torch

os.environ['CUDA_VISIBLE_DEVICES'] ='1'

device = torch.device('cuda:{}'.format(0))

one_stage = "model-best_vit_8f_71.7.pth.pth"
IN21K_check = "vit_base_patch16_224_in21k.pth"
k600_check = "TimeSformer_divST_8x32_224_K600.pyth"

ONE_stage_dict = torch.load(one_stage, map_location=device)
IN21K_model_dict = torch.load(IN21K_check, map_location=device)
k600_model_dict = torch.load(k600_check, map_location=device)['model_state']

dict = OrderedDict()
del_keys = ['head.weight', 'head.bias']
for k in del_keys:
    del IN21K_model_dict[k]
    del ONE_stage_dict['module.'+k]

for k,v in ONE_stage_dict.items():
    k = k[7:]
    dict['base.' + k] = v
    # if('blocks' in k):
    #     if(int(k[7])<=5 and k[8] == '.'):
    #         dict['base.' + k] = v
    # else:
    #     if ('cls' in k or 'embed' in k or 'norm' in k):
    #         dict['base.' + k] = v
for k,v in IN21K_model_dict.items():
    dict['plus.' + k] = v

dict['plus.time_embed'] = k600_model_dict['model.time_embed']

torch.save(dict, "two_stage_deprh12_checkpoint")