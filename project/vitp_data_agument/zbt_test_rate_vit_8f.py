import os
import argparse
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import dataset_config
from dataset import TSNDataSet
# from model.my_vtn_model import VTN as create_model
# from  my_vit_model import vit_base_patch16_224_in21k as create_model
from my_vit_8f_model import vit_base_patch16_224_in21k as create_model

os.environ['CUDA_VISIBLE_DEVICES'] ='2'

def main(args):
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset_name,
                                                                                                      args.modality)
    batch_size = args.batch_size
    device = torch.device('cuda:{}'.format(args.device[0]) if torch.cuda.is_available() else "cpu")

    model = create_model(batch_size,num_segments=8,num_classes=25, has_logits=False)

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
                                             shuffle=True,
                                             pin_memory=False,
                                             num_workers=nw,
                                             drop_last=True)

    #load weight
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # del_keys = ['head.weight', 'head.bias']
        # for k in del_keys:
        #     del weights_dict[k]
        if 'module' in list(weights_dict.keys())[0]:
            model = model.cuda()
            # model = torch.nn.DataParallel(model, device_ids=[x for x in range(len(args.device))])
            model = torch.nn.DataParallel(model.to(device), args.device)
            model.load_state_dict(weights_dict)
        else:
            model.load_state_dict(weights_dict)
            model = model.cuda()
            # model = torch.nn.DataParallel(model, device_ids=[x for x in range(len(args.device))])
            model = torch.nn.DataParallel(model.to(device), args.device)

    video_labels = []
    video_pred = []
    with torch.no_grad():
        for epoch in range(args.epochs):
            model.eval()
            accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
            sample_num = 0
            val_loader = tqdm(val_loader, file=sys.stdout)
            for step, (images, labels) in enumerate(val_loader):
                images = torch.cat(images).reshape(-1, batch_size, 3, 224, 224)  # (T*B,C,H,W)
                images = images.permute(1, 0, 2, 3, 4).reshape(-1, 3, 224, 224).to(device)
                video_labels.append(labels.to("cpu"))
                sample_num += batch_size
                pred = model(images)
                pred_classes = torch.max(pred, dim=1)[1]
                video_pred.append(pred_classes.to("cpu"))
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()
                val_loader.desc = "[valid epoch {}],{},{},acc: {:.3f}".format(epoch, accu_num.item(),sample_num,
                                                                           accu_num.item() / sample_num)

    #congfusion matrix
    root_path = args.root_path
    labels = []
    file = open(os.path.join(root_path, 'categories.txt'), 'r')
    lines = file.readlines()
    for line in lines:
        labels.append(line.strip())
    file.close()
    tick_marks = np.array(range(len(labels))) + 0.5
    def plot_confusion_matrix(cm, title='Confusion', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=-60)
        plt.tick_params(axis='x', labelsize=7)
        plt.yticks(xlocations, labels)
        plt.tick_params(axis='y', labelsize=7)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    video_pred = torch.cat(video_pred).numpy()
    video_labels =torch.cat(video_labels).numpy()
    cf = confusion_matrix(video_labels, video_pred).astype(float)
    # cf = confusion_matrix(video_pred, video_labels).astype(float)
    np.set_printoptions(precision=2)
    cf_normalized = cf / cf.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cf_normalized[y_val][x_val]
        if c >= 0.5:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='w', fontsize=7, va='center', ha='center')
        elif c >= 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=7, va='center', ha='center')

    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cf_normalized, title='confusion matrix')
    save_path = os.path.join(root_path, 'confusion_matrix')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, 'vit_8f_'), format='png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--device', default=[1,3], help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--drop_out_rate', type=int, default=0.5)

    parser.add_argument('--num_segments',type=int,default=8)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--pin_memory',type=bool,default=True)
    parser.add_argument('--DETECTION.ENABLE',type=bool,default=False)

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/data/zhoubotong/project/0_mframe_input_module/weights/era_lr0.002_batch8_vit_base_patch16_224_in21k.pth/model-best_vit_8f_77.6.pth',
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
