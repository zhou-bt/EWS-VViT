import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import dataset_config
from dataset import TSNDataSet
from T2T_ViT_onyl_replace_embading import vit_base_patch16_224_in21k as create_model
from my_8f_utils import train_one_epoch, evaluate

os.environ['CUDA_VISIBLE_DEVICES'] ='3'

def main(args):
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset_name,
                                                                                                      args.modality)
    batch_size = args.batch_size
    device = torch.device('cuda:{}'.format(args.device[0]) if torch.cuda.is_available() else "cpu")

    if os.path.exists("../../weights") is False:
        os.makedirs("../../weights")

    tb_writer = SummaryWriter()

    model = create_model(batch_size, num_segments=8, num_classes=25, has_logits=False)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = TSNDataSet(args.root_path, args.train_list,
                               num_segments=args.num_segments,
                               modality=args.modality,
                               image_tmpl=prefix,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                             modality=args.modality,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=data_transform["val"])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=True)

    val_acc_best = 0
    _,pre_wigth = args.weights.split('/')
    best_path = './weights/{}_lr{}_batch{}_{}'.format(args.dataset_name,args.lr,batch_size,pre_wigth)
    if os.path.exists(best_path) is False:
        os.makedirs(best_path)

    #load weight
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict1 = torch.load(args.weights, map_location=device)
        weights_dict2 = torch.load(args.weights2, map_location=device)['state_dict_ema']
        weights_dict = model.state_dict()
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
            del weights_dict2[k]
        for k,v in weights_dict1.items():   #vit
            if 'block' in k:
                weights_dict[k] = v

        for k,v in weights_dict2.items():   #embed
            if 'block' not in k:
                weights_dict[k] = v

        if 'module' in list(weights_dict.keys())[0]:
            model = model.cuda()
            # model = torch.nn.DataParallel(model, device_ids=[x for x in range(len(args.device))])
            model = torch.nn.DataParallel(model.to(device), args.device)
            model.load_state_dict(weights_dict,False)
        else:
            model.load_state_dict(weights_dict,False)
            model = model.cuda()
            # model = torch.nn.DataParallel(model, device_ids=[x for x in range(len(args.device))])
            model = torch.nn.DataParallel(model.to(device), args.device)

    #freeze
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                batch_size=batch_size)

        scheduler.step()

        # validate

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     batch_size=batch_size,)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # if epoch%5 == 0:
        #     torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            torch.save(model.state_dict(), best_path+'/'+"model-best.pth")
            print('------best model updata ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--device', default=[0], help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--drop_out_rate', type=int, default=0.5)

    parser.add_argument('--num_segments',type=int,default=8)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--pin_memory',type=bool,default=True)
    parser.add_argument('--DETECTION.ENABLE',type=bool,default=False)

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='../vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    parser.add_argument('--weights2', type=str, default='./82.4_T2T_ViTt_19.pth.tar',
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
