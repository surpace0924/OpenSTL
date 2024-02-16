import cv2
import gzip
import numpy as np
import os
import random
import pandas as pd
import math

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class JarTest(Dataset):
    def __init__(self, root, is_train=True, data_name='jartest', res_dir='res', ex_name='jartest',
                 n_frames_input=10, n_frames_output=10, image_size=64,
                 num_objects=[2], transform=None, use_augment=False):
        super(JarTest, self).__init__()

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.mean = 0
        self.std = 1
        self.data_name = data_name

        # 水質データの読み込み
        dataset_dir_path = os.path.join(root, 'jartest')
        wq_data_path = os.path.join(dataset_dir_path, 'water_quality.csv')
        df_water = pd.read_csv(wq_data_path)

        # 水質データを最終濁度についてソートし，idのnumpy配列を得る
        df_water = df_water.sort_values('fin_turbidity')
        idx_array = df_water['id'].values

        # idx配列をsplit_num行の2次元配列にする
        # あまりの部分は -1 で埋める
        split_num = 5
        rows = split_num
        cols = math.ceil(len(idx_array)/split_num)
        total_elements = rows * cols
        remainder_array = -1*np.ones(total_elements - len(idx_array))
        idx_array = np.concatenate((idx_array, remainder_array)).astype(np.int32)
        idx_mat = idx_array.reshape(cols, rows).T
        
        # データセットの読み込み
        dataset_path = os.path.join(dataset_dir_path, 'dataset.npy')
        self.dataset = np.load(dataset_path)

        # train test に分割
        idx = 0
        if '0' in data_name:
            idx = 0
        elif '1' in data_name:
            idx = 1
        elif '2' in data_name:
            idx = 2
        elif '3' in data_name:
            idx = 3
        elif '4' in data_name:
            idx = 4

        if is_train:
            # 訓練データはidに対応する行以外を抜き出す
            video_id_array = np.delete(idx_mat, idx, 0).flatten()
        else:
            # 検証データはid に対応する行を抜き出す
            video_id_array = idx_mat[idx]

            # # 使用するインデックスのリストを結果ディレクトリに保存
            # path = os.path.join(res_dir, ex_name, 'idx_list.csv')
            # df = pd.DataFrame(video_id_array)
            # df.to_csv(path, index=False, header=False)
            
        # あまりの -1 は削除
        video_id_array = video_id_array[video_id_array != -1]
        self.dataset = self.dataset[video_id_array]
        self.dataset = self.dataset.astype(np.float32)
    
        # 時間方向に間引く
        self.dataset = self.dataset[:, ::2, ...]
        print(self.dataset.shape)
        print(video_id_array)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        input = data[:self.n_frames_input, ...]
        output = data[self.n_frames_input:self.n_frames_input+self.n_frames_output, ...]
        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        return input, output

    def __len__(self):
        return len(self.dataset)


def load_data(batch_size, val_batch_size, data_root, num_workers=4, res_dir='res', ex_name='jartest', data_name='jartest',
              pre_seq_length=15, aft_seq_length=15, in_shape=[15, 3, 64, 64],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):
    image_size = in_shape[-1] if in_shape is not None else 64
    train_set = JarTest(root=data_root, is_train=True, data_name=data_name, res_dir=res_dir, ex_name=ex_name,
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length, num_objects=[2],
                            image_size=image_size, use_augment=use_augment)
    test_set = JarTest(root=data_root, is_train=False, data_name=data_name, res_dir=res_dir, ex_name=ex_name,
                           n_frames_input=pre_seq_length,
                           n_frames_output=aft_seq_length, num_objects=[2],
                           image_size=image_size, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    from openstl.utils import init_dist
    os.environ['LOCAL_RANK'] = str(0)
    os.environ['RANK'] = str(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist_params = dict(launcher='pytorch', backend='nccl', init_method='env://', world_size=1)
    init_dist(**dist_params)

    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  data_name='jartest',
                  pre_seq_length=10, aft_seq_length=10,
                  distributed=True, use_prefetcher=False)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
