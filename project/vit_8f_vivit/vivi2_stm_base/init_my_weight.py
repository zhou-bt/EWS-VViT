import torch

weights_dict_s = torch.load('../vit_base_patch16_224_in21k.pth')
weights_dict_timesformer = torch.load('TimeSformer_divST_8x32_224_K600.pyth')['model_state']
weights_dict_t = weights_dict_timesformer['model_state']
weights_dict_2 = {}
for k, v in weights_dict_s.items():
    if 'attn' in k:  # 'blocks.0.attn.proj.bias'
        for i in range(12, 14):
            if k[i] == 'n':
                new = list(k)
                new.insert(i + 1, '2')
                new = ''.join(new)
                weights_dict_2[new] = v
                break
    weights_dict_2[k] = v

torch.save(weights_dict_2, "/data/zhoubotong/project/0_mframe_input_module/project/vit_8f_vivit/vivi2_stm_base/my_init_wight.pth")