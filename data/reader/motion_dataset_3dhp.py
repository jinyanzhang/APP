import os, random
import copy
import torch
import numpy as np

from torch.utils.data import Dataset

from utils.data import read_pkl

# root joint 14
left_joints, right_joints = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]


def flip_data(data, cropped=False):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    if cropped:
        flipped_data[..., 0] = 288 - flipped_data[..., 0] - 1
    else:
        flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


class MotionDataset3DHP(Dataset):
    def __init__(self, args, subset_list, data_split, rank=None, world_size=None):
        """
        :param args: Arguments from the config file
        :param subset_list: A list of datasets
        :param data_split: Either 'train' or 'test'
        """
        super(MotionDataset3DHP, self).__init__()
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split

        self.flip = args.flip

        self.rank = 0
        self.offset = 0
        self.file_list = self._generate_file_list()
        self.length = len(self.file_list)

        if data_split == 'test':
            self.dist_size = self.prepare_labels(rank, world_size)
            if world_size is not None:
                self.length = self.offset if self.rank != world_size - 1 else self.length - self.offset * (world_size - 1)
            print(f'Dataset: rank {self.rank}, offset {self.offset}, length {self.length}, idx begin with {self.offset * self.rank}')

    def prepare_labels(self, rank, world_size):
        if rank is not None and world_size is not None:
            offset = len(self.file_list) // world_size
            # BUGFIX
            self.rank = rank
            self.offset = offset
            dist_size = [offset if i < world_size - 1 else len(self.file_list) - offset * (world_size - 1) for i in range(world_size)]
            # start = offset * rank
            # end = len(self.file_list) if rank == world_size - 1 else start + offset

            return dist_size
        else:
            return [self.length]

    def _generate_file_list(self):
        file_list = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list.append(os.path.join(data_path, i))
        return file_list

    def __len__(self):
        return self.length

    def __get_train_data(self, motion_file):
        motion_2d = motion_file["joints_2d"]
        motion_3d = motion_file["joints_3d_image_norm"]

        if self.flip and random.random() > 0.5:
            motion_2d = flip_data(motion_2d)
            motion_3d = flip_data(motion_3d)

        motion_2d = torch.FloatTensor(motion_2d)
        motion_3d = torch.FloatTensor(motion_3d)

        return motion_2d, motion_3d

    def __get_test_data(self, motion_file):
        motion_2d = motion_file["joints_2d"]
        motion_3d = motion_file["joints_3d_image"]
        valid_index = motion_file["valid_index"]
        source = int(motion_file['source'][-1])

        motion_2d = torch.FloatTensor(motion_2d)
        motion_3d = torch.FloatTensor(motion_3d)

        return motion_2d, motion_3d, valid_index, source

    def __getitem__(self, idx):
        idx += self.rank * self.offset
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        if self.data_split == 'train':
            return self.__get_train_data(motion_file)
        else:
            return self.__get_test_data(motion_file)


class MotionFeatureDataset3DHP(MotionDataset3DHP):
    def __init__(self, args, subset_list, data_split, rank=None, world_size=None):
        super(MotionFeatureDataset3DHP, self).__init__(args, subset_list, data_split, rank, world_size)
        self.subset = args.subset_list[0]
        
        self.poses_root = args.poses_root
        self.feature_root = args.feature_root
        
    def __get_train_data(self, motion_file, file_path):
        flip_tag = self.flip and random.random() > 0.5
        postfix = 'f' if flip_tag else ''
        
        npz_file = np.load(f"{self.feature_root}/{self.subset}/{self.data_split}/{file_path.split('/')[-1][:-4]}{postfix}.npz")
        feature_maps = [torch.FloatTensor(npz_file[f'f_{i}']) for i in range(4)]
        npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}{postfix}.npz")
        motion_2d = torch.FloatTensor(npz_file['f'])
        
        motion_2d_crop = motion_file["joints_2d_crop"]
        motion_3d = motion_file["joints_3d_image_norm"]
        if flip_tag:
            motion_2d_crop = flip_data(motion_2d_crop, True)
            motion_3d = flip_data(motion_3d)

        motion_2d = torch.FloatTensor(motion_2d)
        motion_2d_crop = torch.FloatTensor(motion_2d_crop)
        motion_3d = torch.FloatTensor(motion_3d)

        return *feature_maps, motion_2d, motion_2d_crop, motion_3d

    def __get_test_data(self, motion_file, file_path):
        npz_file = np.load(f"{self.feature_root}/{self.subset}/{self.data_split}/{file_path.split('/')[-1][:-4]}.npz")
        feature_maps = [torch.FloatTensor(npz_file[f'f_{i}']) for i in range(4)]
        
        npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}.npz")
        motion_2d = torch.FloatTensor(npz_file['f'])
        
        motion_2d_crop = motion_file["joints_2d_crop"]
        motion_3d = motion_file["joints_3d_image"]
        valid_index = motion_file["valid_index"]
        source = int(motion_file['source'][-1])

        motion_3d = torch.FloatTensor(motion_3d)

        return *feature_maps, motion_2d, motion_2d_crop, motion_3d, valid_index, source

    def __getitem__(self, idx):
        idx += self.rank * self.offset
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        if self.data_split == 'train':
            return self.__get_train_data(motion_file, file_path)
        else:
            return self.__get_test_data(motion_file, file_path)


class MotionImageDataset3DHP(MotionDataset3DHP):
    def __init__(self, args, subset_list, data_split, rank=None, world_size=None):
        super(MotionImageDataset3DHP, self).__init__(args, subset_list, data_split, rank, world_size)
        self.images_root = args.images_root
        self.poses_root = args.poses_root
        
    def __get_train_data(self, motion_file, file_path):
        flip_tag = self.flip and random.random() > 0.5
        postfix = 'f' if flip_tag else ''
        
        npz_file = np.load(f"{self.images_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}{postfix}.npz")
        images = torch.FloatTensor(npz_file['f'])
        npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}{postfix}.npz")
        motion_2d = torch.FloatTensor(npz_file['f'])
        
        motion_2d_crop = motion_file["joints_2d_crop"]
        motion_3d = motion_file["joints_3d_image_norm"]
        if flip_tag:
            motion_2d_crop = flip_data(motion_2d_crop, True)
            motion_3d = flip_data(motion_3d)

        images = torch.FloatTensor(images)
        motion_2d = torch.FloatTensor(motion_2d)
        motion_2d_crop = torch.FloatTensor(motion_2d_crop)
        motion_3d = torch.FloatTensor(motion_3d)

        return images, motion_2d, motion_2d_crop, motion_3d

    def __get_test_data(self, motion_file, file_path):
        npz_file = np.load(f"{self.images_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}.npz")
        images = torch.FloatTensor(npz_file['f'])
        npz_file = np.load(f"{self.images_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}f.npz")
        images_flipped = torch.FloatTensor(npz_file['f'])
        
        npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}.npz")
        motion_2d = torch.FloatTensor(npz_file['f'])
        npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}f.npz")
        motion_2d_flipped = torch.FloatTensor(npz_file['f'])
        
        motion_2d_crop = motion_file["joints_2d_crop"]
        motion_3d = motion_file["joints_3d_image"]
        valid_index = motion_file["valid_index"]
        source = int(motion_file['source'][-1])

        motion_3d = torch.FloatTensor(motion_3d)

        return images, images_flipped, motion_2d, motion_2d_flipped, motion_2d_crop, motion_3d, valid_index, source

    def __getitem__(self, idx):
        idx += self.rank * self.offset
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        if self.data_split == 'train':
            return self.__get_train_data(motion_file, file_path)
        else:
            return self.__get_test_data(motion_file, file_path)