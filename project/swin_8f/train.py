import math
import os
import argparse

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import dataset_config
from dataset import TSNDataSet
from model import swin_base_patch4_window7_224_in22k as create_model
from utils import train_one_epoch, evaluate

os.environ['CUDA_VISIBLE_DEVICES'] ='2'

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
    _, pre_wigth = args.weights.split('/')
    best_path = './weights/{}_lr{}_batch{}_{}'.format(args.dataset_name, args.lr, batch_size, pre_wigth)
    if os.path.exists(best_path) is False:
        os.makedirs(best_path)

    # load weight
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        if 'module' in list(weights_dict.keys())[0]:
            model = model.cuda()
            # model = torch.nn.DataParallel(model, device_ids=[x for x in range(len(args.device))])
            model = torch.nn.DataParallel(model.to(device), args.device)
            model.load_state_dict(weights_dict, False)
        else:
            model.load_state_dict(weights_dict, False)
            model = model.cuda()
            # model = torch.nn.DataParallel(model, device_ids=[x for x in range(len(args.device))])
            model = torch.nn.DataParallel(model.to(device), args.device)



    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    val_acc_best = 0
    _,pre_wigth = args.weights.split('/')
    best_path = './weights/{}_lr{}_batch{}_{}'.format(args.dataset_name,args.lr,batch_size,pre_wigth)
    if os.path.exists(best_path) is False:
        os.makedirs(best_path)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                batch_size=batch_size)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     batch_size=batch_size)

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
            torch.save(model.state_dict(), best_path + '/' + "model-best.pth")
            print('------best model updata ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--device', default=[0], help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--num_segments',type=int,default=8)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--pin_memory',type=bool,default=True)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./swin_base_patch4_window7_224_22k.pth',
                        help='initial weights path')
    # 数据集所在根目录
    parser.add_argument('--dataset_name', default='era', help='create model name')

    parser.add_argument('--modality', type=str, default='RGB')
    parser.add_argument('--non_local',type = bool,default=False)

    opt = parser.parse_args()

    main(opt)
