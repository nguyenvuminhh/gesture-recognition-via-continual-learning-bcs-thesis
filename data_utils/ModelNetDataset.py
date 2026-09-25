'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import re

from tqdm import tqdm
from torch.utils.data import Dataset

from data_utils.constants import DATA_SET_NAME_LIST, NOF_CONTEXTS, DataSetNameClass
from data_utils.dataset_name_pattern_mapping import get_dataset_pattern

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataset(Dataset):
    def __init__(self, root, args, split, process_data, dataset_name_index=0):
        self.args = args
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.dataset_name_index = dataset_name_index
        self.continual_learning = args.continual_learning

        assert split in ['train', 'val', 'test', 'val_so_far', 'test_so_far'], f"Invalid split: {split}"
        self.split = split


        # 类别文件
        self.catfile = os.path.join(self.root, 'my_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # 文件列表
        shape_ids = {
            'train': [line.rstrip() for line in open(os.path.join(self.root, 'train_filelist.txt'))],
            'val': [line.rstrip() for line in open(os.path.join(self.root, 'val_filelist.txt'))],
            'test': [line.rstrip() for line in open(os.path.join(self.root, 'test_filelist.txt'))]
        }

        dataset_name = DATA_SET_NAME_LIST[self.dataset_name_index]
        if self.continual_learning and split != 'test':
            if split in ['val_so_far']:
                dataset_name = "upto " + DATA_SET_NAME_LIST[self.dataset_name_index]
                pattern = '|'.join([get_dataset_pattern(DATA_SET_NAME_LIST[i]) for i in range(self.dataset_name_index + 1)])
            else:  # split is 'train', 'val'
                pattern = get_dataset_pattern(dataset_name)
        else:
            pattern = r'.*'

        split = split.replace('_so_far', '') 
        shape_ids[split] = [x for x in shape_ids[split] if re.match(pattern, x)]

        shape_names = [x[:[i for i, c in enumerate(x) if c == '_'][2]] for x in shape_ids[split]]

        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i in range(len(shape_ids[split]))]

        print(f"Split: {self.split}, Dataset {self.dataset_name_index} - {dataset_name}, Length: {len(self.datapath)}")

        # 缓存文件名
        if self.uniform:
            self.save_path = os.path.join(root, f'modelnet{self.num_category}_{split}_{self.npoints}pts_fps.dat')
        else:
            self.save_path = os.path.join(root, f'modelnet{self.num_category}_{split}_{self.npoints}pts.dat')

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(f'Processing data {split}... (only once)')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    filename = self.datapath[index][1]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(self.datapath[index][1], delimiter=',').astype(np.float32)
                    
                    
                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print(f'Load processed data from {self.save_path}...')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(self.datapath[index][1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)

    def next_context(self):
        """
        Increment the dataset_name_index by 1 and return a new instance of ModelNetDataset.
        """
        if self.continual_learning is False:
            raise ValueError("Continual learning is not enabled for this dataset loader.")
        
        new_index = self.dataset_name_index + 1
        if new_index >= NOF_CONTEXTS:
            return None

        new_loader = ModelNetDataset(
            root=self.root,
            args=self.args,
            split=self.split,
            process_data=self.process_data,
            dataset_name_index=new_index,
        )

        return new_loader

if __name__ == '__main__':
    import torch

    data = ModelNetDataset('/data/output_txt/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
