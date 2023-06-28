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
from MVIT import MViT as create_model
from utils import evaluate,train_one_epoch

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

def main(args):
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset_name,
                                                                                                      args.modality)
    batch_size = args.batch_size
    device = torch.device('cuda:{}'.format(args.device[0]) if torch.cuda.is_available() else "cpu")

    if os.path.exists("../../weights") is False:
        os.makedirs("../../weights")

    tb_writer = SummaryWriter()

    model = create_model(args)

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
    pre_wigth,_ = args.weights.split('.')
    best_path = './weights/{}_lr{}_batch{}_{}'.format(args.dataset_name,args.lr,batch_size,pre_wigth)
    if os.path.exists(best_path) is False:
        os.makedirs(best_path)
    print("lr:",args.lr)
    print("batch_size:", args.batch_size)

    #load weight
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        model_dict = model.state_dict()
        pre_train_dict = weights_dict["model_state"]
        pre_train_dict_match = {
            k: v
            for k, v in pre_train_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        model.load_state_dict(pre_train_dict_match, strict=False)

        model = torch.nn.DataParallel(model.to(device), args.device)

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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--device', default=[0], help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--drop_out_rate', type=int, default=0.5)

    parser.add_argument('--num_segments',type=int,default=8)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--pin_memory',type=bool,default=True)

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='MViTv2_B_in21k.pth',
                        help='initial weights path')
    # 数据集所在根目录
    parser.add_argument('--dataset_name', default='era', help='create model name')

    parser.add_argument('--modality', type=str, default='RGB')
    parser.add_argument('--non_local',type = bool,default=False)
    # ===============MVIT===================
    parser.add_argument('--train_corp_size', type=int, default=224)
    parser.add_argument('--test_corp_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int , default=96)
    parser.add_argument('--num_head', type=int , default=1)
    parser.add_argument('--depth', type=int , default=24)
    parser.add_argument('--cls_embed_on', type=bool, default=False)
    parser.add_argument('--use_abs_pos', type=bool, default=False)
    parser.add_argument('--zero_decay_pos_cls', type=bool, default=False)
    parser.add_argument('--patch_kernel', type=list, default=[7,7])
    parser.add_argument('--patch_stride', type=list, default=[4,4])
    parser.add_argument('--patch_padding', type=list, default=[3,3])
    parser.add_argument('--droppath_rate', type=float, default=0.3)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--act_func', type=str, default="softmax")
    parser.add_argument('--dim_mul', type=list, default=[[2, 2.0], [5, 2.0], [21, 2.0]])
    parser.add_argument('--head_mul', type=list, default=[[2, 2.0], [5, 2.0], [21, 2.0]])
    parser.add_argument('--pool_Q_stride', type=list, default=[[0, 1, 1], [1, 1, 1], [2, 2, 2], [3, 1, 1], [4, 1, 1], [5, 2, 2], [6, 1, 1], [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1], [12, 1, 1], [13, 1, 1], [14, 1, 1], [15, 1, 1], [16, 1, 1], [17, 1, 1], [18, 1, 1], [19, 1, 1], [20, 1, 1], [21, 2, 2], [22, 1, 1], [23, 1, 1]])
    parser.add_argument('--pool_kqv_kernel', type=list, default=[3,3])
    parser.add_argument('--pool_kv_stride_adaptive', type=list, default=[4,4])
    parser.add_argument('--pool_kv_stride', default=None)
    parser.add_argument('--dim_mul_in_att', type=bool, default=True)
    parser.add_argument('--MLP_ratio', type=float, default=4.0)
    parser.add_argument('--kqv_bias', type=bool, default=True)
    parser.add_argument('--mode', type=str, default="conv")
    parser.add_argument('--pool_first', type=bool, default=False)
    parser.add_argument('--rel_pos_spatial', type=bool, default=True)
    parser.add_argument('--rel_pos_zero_init', type=bool, default=False)
    parser.add_argument('--residual_pooling', type=bool, default=False)


    opt = parser.parse_args()

    main(opt)
