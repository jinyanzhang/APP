from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import torch
import pickle
import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.data import read_pkl, resample

random.seed(0)

class DataReader3DHP(object):
    def __init__(self, n_frames, data_stride_train, image_stride=9,
                 dt_root='/data1/motion3d', dt_file='3dhp_gt.pkl',
                 dt_name='3DHPImages'):
        self.train_intervals = None
        self.test_intervals = None
        self.dt_dataset = read_pkl('%s/%s' % (dt_root, dt_file))
        self.n_frames = n_frames
        self.data_stride_train = data_stride_train

        self.image_stride = image_stride
        self.dt_root = dt_root
        self.dt_name = dt_name

        self.seq_valid_nums = {f'TS{i+1}': num for i, num in enumerate([603, 540, 506, 558, 276, 392])}
        assert n_frames % image_stride == 0

    def denormalize(self, pred, seq):
        out = pred.cpu().numpy()
        for idx in range(out.shape[0]):
            if f'TS{seq[idx]}' in ['TS5', 'TS6']:
                res_w, res_h = 1920, 1080
            else:
                res_w, res_h = 2048, 2048
            out[idx, ..., :2] = (out[idx, ..., :2] + np.array([1, res_h / res_w])) * res_w / 2
            out[idx, ..., 2:] = out[idx, ..., 2:] * res_w / 2
        out = out - out[..., 0:1, :]
        return torch.tensor(out).cuda()

    def __get_train_data(self):
        train_joints_2d = self.dt_dataset['train']['joints_2d']
        train_joints_2d = np.concatenate([train_joints_2d, np.ones((*train_joints_2d.shape[:-1], 1), dtype=np.float32)], axis=-1)
        train_joints_2d_crop = self.dt_dataset['train']['joints_2d_crop']
        train_joints_3d_image_norm = self.dt_dataset['train']['joints_3d_image_norm']
        train_image = np.array(self.dt_dataset['train']['image'])
        
        frames = self.n_frames
        stride = self.data_stride_train
        image_stride = self.image_stride
        
        low, high, intervals = 0, 1, []
        
        for i in range(1, len(train_image)):
            if  train_image[i][:-15] == train_image[low][:-15]:
                high = i + 1
            else:
                intervals.append([low, high])
                low = i
        intervals.append([low, high])

        train_intervals = []
        train_image_intervals = []
        for interval in intervals:
            low, high = interval
            train_intervals += [np.array(range(i, i + frames)) for i in range(low, high - frames, stride)]
            train_image_intervals += [np.array(range(i + image_stride // 2, i + frames, image_stride)) for i in range(low, high - frames, stride)]
            # keep last
            if high % frames != 0:
                last_interval = np.array(list(range((high // frames) * frames, high)))
                resampled = resample(last_interval.shape[0], frames)
                last_interval = last_interval[resampled]
                last_image_interval = last_interval[image_stride // 2::image_stride]
                train_intervals.append(last_interval)
                train_image_intervals.append(last_image_interval)

        train_joints_2d = np.array(train_joints_2d[train_intervals]).astype(np.float32)
        train_joints_2d_crop = np.array(train_joints_2d_crop[train_intervals]).astype(np.float32)
        train_joints_3d_image_norm = np.array(train_joints_3d_image_norm[train_intervals]).astype(np.float32)
        train_image = np.array(train_image[train_image_intervals])
        
        save_path = os.path.join(self.dt_root, f"{self.dt_name}_{self.n_frames}_{self.n_frames // self.image_stride}", 'train')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        for i in tqdm(range(train_joints_2d.shape[0])):
            data_dict = {}
            data_dict['joints_2d'] = train_joints_2d[i]
            data_dict['joints_2d_crop'] = train_joints_2d_crop[i]
            data_dict['joints_3d_image_norm'] = train_joints_3d_image_norm[i]
            
            data_dict['image'] = train_image[i]
            with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as fp:
                pickle.dump(data_dict, fp)
            fp.close()
    
    def __get_test_data(self):
        test_joints_2d = self.dt_dataset['test']['joints_2d']
        test_joints_2d = np.concatenate([test_joints_2d, np.ones((*test_joints_2d.shape[:-1], 1), dtype=np.float32)], axis=-1)
        test_joints_2d_crop = self.dt_dataset['test']['joints_2d_crop']
        test_joints_3d_image_norm = self.dt_dataset['test']['joints_3d_image_norm']
        test_joints_3d_image = self.dt_dataset['test']['joints_3d_image']
        
        test_image = self.dt_dataset['test']['image']
        test_source = self.dt_dataset['test']['source']
        test_valid = self.dt_dataset['test']['valid']
        
        frames, pad = self.n_frames, self.n_frames // 2
        image_stride = self.image_stride
        
        test_intervals = {f'TS{i}': [] for i in range(1, 7)}
        test_valid_image = deepcopy(test_intervals)
        test_valid_source = deepcopy(test_intervals)
        
        test_valid_joints_2d = deepcopy(test_intervals)
        test_valid_joints_2d_crop = deepcopy(test_intervals)
        test_valid_joints_3d_image = deepcopy(test_intervals)
        test_valid_joints_3d_image_norm = deepcopy(test_intervals)

        test_valid_intervals = deepcopy(test_intervals)
        test_valid_indices = deepcopy(test_intervals)
        # denote how many valid frames before tmp valid frame
        test_valid_counter = deepcopy(test_intervals)

        low, high = 0, 1
        # for every TS seq, it is corresponding to only one interval
        for i in range(1, len(test_source)):
            if  test_source[i] == test_source[low]:
                high = i + 1
            else:
                test_intervals[test_source[low]].append([low, high])
                low = i
        test_intervals[test_source[low]].append([low, high])

        # for every interval of TS seq
        for seq in test_intervals.keys():
            for i, interval in enumerate(test_intervals[seq]):
                low, high = interval
                counter = 0
                # scan every value in [low, high), find the position where the valid equals to 1
                for j in range(low, high):
                    if test_valid[j] < 1:
                        continue
                    test_valid_indices[seq].append(j)
                    test_valid_counter[seq].append(counter)
                    # [-40, 0, 41)
                    test_valid_intervals[seq].append(range(j - pad, j + pad + 1))
                    counter += 1

        zero_joint_2d = np.zeros((1, 17, 2), dtype=np.float32)
        zero_joint_3d = np.zeros((1, 17, 3), dtype=np.float32)
        zero_image = 'pad'

        # for out of indices item, padding them with zeros
        for seq in test_valid_intervals.keys():
            for i, valid_interval in enumerate(test_valid_intervals[seq]):
                # get the lower bound and upper bound
                # [low, high)
                low, high = test_intervals[seq][0]
                # [start, end)
                start = valid_interval.start
                end = valid_interval.stop

                tmp_source = test_source[(end - start) // 2]

                # left and right pad
                left_pad, right_pad = 0, 0
                if start < low:
                    left_pad = low - start
                    start = low
                if end > high:
                    right_pad = end - high
                    end = high

                tmp_joints_2d = test_joints_2d[start: end]
                tmp_joints_2d_crop = test_joints_2d_crop[start: end]
                tmp_joints_3d_image = test_joints_3d_image[start: end]
                tmp_joints_3d_image_norm = test_joints_3d_image_norm[start: end]

                tmp_image = test_image[start: end]

                if left_pad != 0:
                    tmp_joints_2d = np.concatenate((np.tile(zero_joint_3d, [left_pad, 1, 1]), tmp_joints_2d), axis=0)
                    tmp_joints_2d_crop = np.concatenate((np.tile(zero_joint_2d, [left_pad, 1, 1]), tmp_joints_2d_crop), axis=0)
                    tmp_joints_3d_image = np.concatenate((np.tile(zero_joint_3d, [left_pad, 1, 1]), tmp_joints_3d_image), axis=0)
                    tmp_joints_3d_image_norm = np.concatenate((np.tile(zero_joint_3d, [left_pad, 1, 1]), tmp_joints_3d_image_norm), axis=0)

                    tmp_image = [zero_image] * left_pad + tmp_image
                if right_pad != 0:
                    tmp_joints_2d = np.concatenate((tmp_joints_2d, np.tile(zero_joint_3d, [right_pad, 1, 1])), axis=0)
                    tmp_joints_2d_crop = np.concatenate((tmp_joints_2d_crop, np.tile(zero_joint_2d, [right_pad, 1, 1])), axis=0)
                    tmp_joints_3d_image = np.concatenate((tmp_joints_3d_image, np.tile(zero_joint_3d, [right_pad, 1, 1])), axis=0)
                    tmp_joints_3d_image_norm = np.concatenate((tmp_joints_3d_image_norm, np.tile(zero_joint_3d, [right_pad, 1, 1])), axis=0)

                    tmp_image = tmp_image + [zero_image] * right_pad

                test_valid_joints_2d[seq].append(tmp_joints_2d)
                test_valid_joints_2d_crop[seq].append(tmp_joints_2d_crop)
                test_valid_joints_3d_image[seq].append(tmp_joints_3d_image)
                test_valid_joints_3d_image_norm[seq].append(tmp_joints_3d_image_norm)

                test_valid_source[seq].append(tmp_source)
                test_valid_image[seq].append(tmp_image[image_stride // 2::image_stride])

        counter = 0
        save_path = os.path.join(self.dt_root, f"{self.dt_name}_{frames}_{frames // image_stride}", 'test')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        for seq in tqdm([f'TS{i}' for i in range(1, 7)]):
            assert self.seq_valid_nums[seq] == len(test_valid_source[seq])
            for j in range(self.seq_valid_nums[seq]):
                data_dict = {}
                data_dict['joints_2d'] = test_valid_joints_2d[seq][j]
                data_dict['joints_2d_crop'] = test_valid_joints_2d_crop[seq][j]
                data_dict['joints_3d_image'] = test_valid_joints_3d_image[seq][j]
                data_dict['joints_3d_image_norm'] = test_valid_joints_3d_image_norm[seq][j]

                data_dict['source'] = test_valid_source[seq][j]
                data_dict['valid_interval'] = test_valid_intervals[seq][j]
                # indicate the valid frame index in the whole test dataset
                data_dict['valid_index'] = test_valid_indices[seq][j]
                # indicate how many valid frames before tmp valid frame in the sequence such as TS1
                data_dict['valid_counter'] = test_valid_counter[seq][j]

                data_dict['image'] = test_valid_image[seq][j]
                with open(os.path.join(save_path, "%08d.pkl" % counter), "wb") as fp:
                    pickle.dump(data_dict, fp)
                fp.close()
                counter += 1

    def get_sliced_data(self):
        self.__get_train_data()
        self.__get_test_data()


if __name__ == '__main__':
    n_frames = 81
    image_stride = 9

    datareader = DataReader3DHP(n_frames=n_frames, image_stride=image_stride, data_stride_train=n_frames // 9)
    datareader.get_sliced_data()
