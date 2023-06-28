# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import os

def return_era(modality):

    filename_categories = 'categories.txt'
    if modality == 'RGB':
        root_data ='/data/zhoubotong/dataset/ERA_Dataset/'
        filename_imglist_train = 'train_videofolder.txt'
        filename_imglist_val = 'test_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_MOD20(modality):
    filename_categories = 'categories.txt'
    if modality == 'RGB':
        root_data = '/data/zhoubotong/dataset/MOD20_frame_1/'
        filename_imglist_train = 'train_videofolder.txt'
        filename_imglist_val = 'test_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_DA(modality):

    filename_categories = 'categories.txt'
    if modality == 'RGB':
        root_data ='/data/zhoubotong/dataset/Drone-action/'
        filename_imglist_train = 'train_videofolder.txt'
        filename_imglist_val = 'test_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_DA(modality):

    filename_categories = 'categories.txt'
    if modality == 'RGB':
        root_data ='/data/zhoubotong/dataset/Drone-action/'
        filename_imglist_train = 'train_videofolder.txt'
        filename_imglist_val = 'test_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'era': return_era,'MOD20':return_MOD20 ,'DA': return_DA}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(root_data, file_imglist_train)
    file_imglist_val = os.path.join(root_data, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(root_data, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]      #delete str at the end of item (default:' ' \t \n )
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val , root_data , prefix