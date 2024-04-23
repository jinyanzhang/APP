import os
import cv2
import json
import torch
import random
import numpy as np
from collections import defaultdict

from torch.utils.data import Dataset
from utils.data import read_pkl, flip_data

class MotionDataset3D(Dataset):
    def __init__(self, args, subset_list, data_split, return_stats=False):
        """
        :param args: Arguments from the config file
        :param subset_list: A list of datasets
        :param data_split: Either 'train' or 'test'
        """
        np.random.seed(0)
        self.data_root = args.data_root
        self.add_velocity = args.add_velocity
        self.subset_list = subset_list
        self.data_split = data_split
        self.return_stats = return_stats

        self.flip = args.flip
        self.use_proj_as_2d = args.use_proj_as_2d

        self.file_list = self._generate_file_list()

    def _generate_file_list(self):
        file_list = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list.append(os.path.join(data_path, i))
        return file_list

    @staticmethod
    def _construct_motion2d_by_projection(motion_3d):
        """Constructs 2D pose sequence by projecting the 3D pose orthographically"""
        motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
        motion_2d[:, :, :2] = motion_3d[:, :, :2]  # Get x and y from the 3D pose
        motion_2d[:, :, 2] = 1  # Set confidence score as 1
        return motion_2d

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        motion_2d = motion_file["data_input"]
        motion_3d = motion_file["data_label"]

        if motion_2d is None or self.use_proj_as_2d:
            motion_2d = self._construct_motion2d_by_projection(motion_3d)

        if self.add_velocity:
            motion_2d_coord = motion_2d[..., :2]
            velocity_motion_2d = motion_2d_coord[1:] - motion_2d_coord[:-1]
            motion_2d = motion_2d[:-1]
            motion_2d = np.concatenate((motion_2d, velocity_motion_2d), axis=-1)

            motion_3d = motion_3d[:-1]

        if self.data_split == 'train':
            if self.flip and random.random() > 0.5:
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)

        if self.return_stats:
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), motion_file['mean'], motion_file['std']
        else:
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)


class MultipleMotionDataset3D(MotionDataset3D):
    def __init__(self, args, subset_list, data_split, rank=None, world_size=None):
        super(MultipleMotionDataset3D, self).__init__(args, subset_list, data_split)
        self.rank = 0
        self.offset = 0
        self.length = len(self.file_list)
        self.detector_type = 'sh'
        
        self.subset = args.subset_list[0]
        
        self.n_frames = args.n_frames
        # self.image_root = args.image_root
        self.data_split = data_split
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

    def set_detector_type(self, detector_type='sh'):
        if detector_type not in ['sh', 'hrnet', 'cpn']:
            self.detector_type = ''
        else:
            if detector_type == 'sh':
                self.feature_length = 1
            self.detector_type = '_' + detector_type

    # def get_images(self, image_paths, flipped=False):
    #     images = []
    #     for image_path in image_paths:
    #         image_full_path = f'{self.image_root}/{image_path}'
    #         image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
    #         if flipped:
    #             image = np.flip(image, -1).copy()
    #         image = cv2.resize(image, (192, 256))
    #         image = image.astype(np.float32) / 255.0
    #         images.append(image)
    #     images = np.stack(images, axis=0)
    #     return images

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx += self.offset * self.rank
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        motion_2d = motion_file[f'joints_2d{self.detector_type}']
        # motion_2d_crop = motion_file[f'joints_2d{self.detector_type}_crop']

        motion_3d = motion_file['joints_3d_image_norm']
        # image_path = motion_file['image_path']

        if motion_2d is None or self.use_proj_as_2d:
            motion_2d = self._construct_motion2d_by_projection(motion_3d)

        if self.add_velocity:
            motion_2d_coord = motion_2d[..., :2]
            velocity_motion_2d = motion_2d_coord[1:] - motion_2d_coord[:-1]
            motion_2d = motion_2d[:-1]
            motion_2d = np.concatenate((motion_2d, velocity_motion_2d), axis=-1)

            motion_3d = motion_3d[:-1]

        motion_2d = torch.FloatTensor(motion_2d)
        # motion_2d_crop = torch.FloatTensor(motion_2d_crop)
        motion_3d = torch.FloatTensor(motion_3d)

        if self.data_split == 'train':
            flip_tag = self.flip and random.random() > 0.5
            if flip_tag:
                motion_2d = flip_data(motion_2d)
                # motion_2d_crop = flip_data(motion_2d_crop, cropped=True)
                motion_3d = flip_data(motion_3d)
                # images = self.get_images(image_path, True)
                # images = torch.FloatTensor(images)
            # else:
            #     images = self.get_images(image_path, False)
            #     images = torch.FloatTensor(images)
        # else:
        #     images = self.get_images(image_path, False)
        #     images = torch.FloatTensor(images)

        # return images, motion_2d, motion_2d_crop, motion_3d
        return motion_2d, motion_3d


class MultipleMotionFeatureDataset3D(MotionDataset3D):
    def __init__(self, args, subset_list, data_split, rank=None, world_size=None):
        super(MultipleMotionFeatureDataset3D, self).__init__(args, subset_list, data_split)
        self.rank = 0
        self.offset = 0
        self.length = len(self.file_list)
        self.detector_type = 'hrnet'
        self.feature_type = 'cpn'
        self.motion_2d_key = 'joints_2d_hrnet_crop'
        
        self.subset = args.subset_list[0]
        
        self.n_frames = args.n_frames
        
        self.feature_root = args.feature_root
        self.poses_root = args.poses_root
        self.feature_length = 4
        
        self.data_split = data_split
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

    def set_detector_type(self, detector_type='sh'):
        if detector_type not in ['gt', 'sh', 'cpn', 'hrnet']:
            raise ValueError
        else:
            self.detector_type = detector_type
            if detector_type == 'gt':
                self.motion_2d_key = 'joints_2d_crop'
                self.feature_type = 'gt'
            else:
                self.motion_2d_key = f'joints_2d_{detector_type}_crop'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx += self.offset * self.rank
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        motion_2d_crop = motion_file[self.motion_2d_key]
        motion_3d = motion_file['joints_3d_image_norm']
        
        motion_2d_crop = torch.FloatTensor(motion_2d_crop)
        motion_3d = torch.FloatTensor(motion_3d)

        if self.data_split == 'train':
            flip_tag = self.flip and random.random() > 0.5
            if flip_tag:
                motion_2d_crop = flip_data(motion_2d_crop, cropped=True)
                motion_3d = flip_data(motion_3d)

            postfix = 'f' if flip_tag else ''
            npz_file = np.load(f"{self.feature_root}/{self.subset}_{self.feature_type}/{self.data_split}/{file_path.split('/')[-1][:-4]}{postfix}.npz")
            feature_maps = [torch.FloatTensor(npz_file[f'f_{i}']) for i in range(self.feature_length)]
            
            npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}{postfix}.npz")
            motion_2d = torch.FloatTensor(npz_file['f'])
        else:
            npz_file = np.load(f"{self.feature_root}/{self.subset}_{self.feature_type}/{self.data_split}/{file_path.split('/')[-1][:-4]}.npz")
            npz_file_flipped = np.load(f"{self.feature_root}/{self.subset}_{self.feature_type}/{self.data_split}/{file_path.split('/')[-1][:-4]}f.npz")
            feature_maps = [torch.FloatTensor(npz_file[f'f_{i}']) for i in range(self.feature_length)]
            feature_maps += [torch.FloatTensor(npz_file_flipped[f'f_{i}']) for i in range(self.feature_length)]
            
            npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}.npz")
            motion_2d = torch.FloatTensor(npz_file['f'])
            npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}f.npz")
            motion_2d_flipped = torch.FloatTensor(npz_file['f'])
            return *feature_maps, motion_2d, motion_2d_flipped, motion_2d_crop, motion_3d

        return *feature_maps, motion_2d, motion_2d_crop, motion_3d


class MultipleMotionImageDataset3D(MotionDataset3D):
    def __init__(self, args, subset_list, data_split, rank=None, world_size=None):
        super(MultipleMotionImageDataset3D, self).__init__(args, subset_list, data_split)
        self.rank = 0
        self.offset = 0
        self.length = len(self.file_list)
        self.detector_type = 'hrnet'
        self.motion_2d_key = 'joints_2d_hrnet_crop'
        
        self.subset = args.subset_list[0]
        
        self.n_frames = args.n_frames
        
        self.images_root = args.images_root
        self.poses_root = args.poses_root
        
        self.data_split = data_split
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

    def set_detector_type(self, detector_type='hrnet'):
        if detector_type not in ['gt', 'hrnet']:
            raise ValueError
        else:
            self.detector_type = detector_type
            if detector_type == 'gt':
                self.motion_2d_key = 'joints_2d_crop'
            else:
                self.motion_2d_key = f'joints_2d_{detector_type}_crop'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx += self.offset * self.rank
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        motion_2d_crop = motion_file[self.motion_2d_key]
        motion_3d = motion_file['joints_3d_image_norm']
        
        motion_2d_crop = torch.FloatTensor(motion_2d_crop)
        motion_3d = torch.FloatTensor(motion_3d)

        if self.data_split == 'train':
            flip_tag = self.flip and random.random() > 0.5
            if flip_tag:
                motion_2d_crop = flip_data(motion_2d_crop, cropped=True)
                motion_3d = flip_data(motion_3d)

            postfix = 'f' if flip_tag else ''
            
            npz_file = np.load(f"{self.images_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}{postfix}.npz")
            images = torch.FloatTensor(npz_file['f'])
            npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}{postfix}.npz")
            motion_2d = torch.FloatTensor(npz_file['f'])
        else:
            npz_file = np.load(f"{self.images_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}.npz")
            images = torch.FloatTensor(npz_file['f'])
            npz_file = np.load(f"{self.images_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}f.npz")
            images_flipped = torch.FloatTensor(npz_file['f'])
            
            npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}.npz")
            motion_2d = torch.FloatTensor(npz_file['f'])
            npz_file = np.load(f"{self.poses_root}/{self.data_split}/{file_path.split('/')[-1][:-4]}f.npz")
            motion_2d_flipped = torch.FloatTensor(npz_file['f'])
            return images, images_flipped, motion_2d, motion_2d_flipped, motion_2d_crop, motion_3d

        return images, motion_2d, motion_2d_crop, motion_3d