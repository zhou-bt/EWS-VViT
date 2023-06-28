import os
from collections import OrderedDict

import torch

os.environ['CUDA_VISIBLE_DEVICES'] ='2'

device = torch.device('cuda:{}'.format(0))

IN21K_check = "vit_base_p16_224_IN21K_k600_T8"
wta_check = "/data/zhoubotong/project/0_mframe_input_module/project/caiT/runs/2022-10-19-fva-wta-cbam-73.7/0.7348703170028819_model-best.pth"

IN21K_model_dict = torch.load(IN21K_check, map_location=device)
wta_check_dit = torch.load(wta_check, map_location=device)

for k, v in wta_check_dit.items():
    if('conv' in k):
        IN21K_model_dict[k[7:]] = v

torch.save(IN21K_model_dict, "vit_base_p16_224_IN21K_k600_T8_wta")