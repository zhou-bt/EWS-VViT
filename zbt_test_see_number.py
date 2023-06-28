import json
import os
import argparse
import sys
from tqdm import tqdm
from pack.ConfusionMatrix import ConfusionMatrix
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import dataset_config
from my_dataset import TSNDataSet
# from model.my_vtn_model import VTN as create_model
# from  my_vit_model import vit_base_patch16_224_in21k as create_model
from project.vit_8f.my_vit_8f_model import vit_base_patch16_224_in21k as create_model

os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'

def main(args):
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset_name,
                                                                                                      args.modality)
    batch_size = args.batch_size
    device = torch.device('cuda:{}'.format(args.device[0]) if torch.cuda.is_available() else "cpu")

    model = create_model(args)
    model = torch.nn.DataParallel(model.to(device), args.device)
    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化验证数据集
    val_dataset = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                             modality=args.modality,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=data_transform["val"])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=True)

    #load weight
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        # del_keys = ['head.weight', 'head.bias'] if  model.has_logits \
        #     else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']

        # del_keys = ['head.weight', 'head.bias']
        # for k in del_keys:
        #     del weights_dict[k]
        model.load_state_dict(weights_dict,False)



    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=25, labels=labels)

    with torch.no_grad():
        for epoch in range(args.epochs):
            model.eval()
            accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
            sample_num = 0
            val_loader = tqdm(val_loader, file=sys.stdout)
            for step, (images, labels, index) in enumerate(val_loader):
                images = torch.cat(images).reshape(-1, batch_size, 3, 224, 224).to(device)  # (T*B,C,H,W)
                index.to(device)
                images = [images, index]
                sample_num += batch_size
                pred = model(images)
                pred_classes = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()
                val_loader.desc = "[valid epoch {}],{},{},acc: {:.3f}".format(epoch, accu_num.item(),sample_num,
                                                                           accu_num.item() / sample_num)
                confusion.update(pred_classes.to("cpu").numpy(), labels.to("cpu").numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--device', default=[3], help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--drop_out_rate', type=int, default=0.5)

    parser.add_argument('--num_segments',type=int,default=8)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--pin_memory',type=bool,default=True)
    parser.add_argument('--DETECTION.ENABLE',type=bool,default=False)

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/data/zhoubotong/project/0_mframe_input_module/weights/era_lr0.00025_batch1_vit_base_patch16_224_in21k.pth/model-best_72.8_long.pth',
                        help='initial weights path')
    # 数据集所在根目录
    parser.add_argument('--dataset_name', default='era', help='create model name')


    parser.add_argument('--modality', type=str, default='RGB')
    parser.add_argument('--non_local',type = bool,default=False)
    # ===============VTN parameter===================
    parser.add_argument('--pretrained', type=bool, default='True')
    parser.add_argument('--drop_path_rate', type=int, default=0)
    parser.add_argument('--drop_rate', type=int, default=0)
    parser.add_argument('--backbone_name', type=str, default='VIT')
    # ================ Longformer =======================
    parser.add_argument('--max_position_embeddings', type=int, default=228)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--attention_mode', type=str, default='sliding_chunks')
    parser.add_argument('--pad_token_id', type=int, default=-1)
    parser.add_argument('--attention_window', type=list, default=[18, 18, 18])
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--mip_dim', type=int, default=768)
    parser.add_argument('--dropout_rate', type=float, default=0.5)


    opt = parser.parse_args()

    main(opt)