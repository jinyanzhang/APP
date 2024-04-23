from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import pickle
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tqdm import tqdm
from utils.data import read_pkl, split_clips

random.seed(0)

class DataReaderH36M(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True,
                 dt_root='data/motion3d', dt_file='h36m_cpn_cam_source.pkl'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_dataset = read_pkl('%s/%s' % (dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence

    def read_2d(self):
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)
                if len(train_confidence.shape) == 2:  # (1559752, 17)
                    train_confidence = train_confidence[:, :, None]
                    test_confidence = test_confidence[:, :, None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:, :, 0:1]
                test_confidence = np.ones(testset.shape)[:, :, 0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset

    def read_3d(self):
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(
            np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(
            np.float32)  # [N, 17, 3]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            train_labels[idx, :, :2] = train_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[idx, :, 2:] = train_labels[idx, :, 2:] / res_w * 2

        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2

        return train_labels, test_labels

    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw

    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]  # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]  # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test

    def turn_into_test_clips(self, data):
        """Converts (total_frames, ...) tensor to (n_clips, n_frames, ...) based on split_id_test"""
        split_id_train, split_id_test = self.get_split_id()
        data = data[split_id_test]
        return data

    def get_hw(self):
        #       Only Testset HW is needed for denormalization
        test_hw = self.read_hw()  # train_data (1559752, 2) test_data (566920, 2)
        test_hw = self.turn_into_test_clips(test_hw)[:, 0, :]  # (N, 2)
        return test_hw

    def get_sliced_data(self):
        train_data, test_data = self.read_2d()  # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d()  # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]  # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]  # (N, 27, 17, 3)
        # ipdb.set_trace()
        return train_data, test_data, train_labels, test_labels

    def denormalize(self, test_data, all_sequence=False):
        #       data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)
        if all_sequence:
            test_data = self.turn_into_test_clips(test_data)

        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        assert len(data) == len(test_hw), f"Data n_clips is {len(data)} while test_hw size is {len(test_hw)}"
        # denormalize (x,y,z) coordinates for results
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data  # [n_clips, -1, 17, 3]


class MultipleDataReaderH36M(DataReaderH36M):
    """
    This class allow you to use multiple 2d detector (CPN/SH/HRNet) extracted 2d poses from our made datasets
    """
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, image_stride=27, read_confidence=True,
                 dt_root='/home/xxxxxx/MotionAGFormer/data1/motion3d', dt_file='h36m_sh_cpn_hrnet_yolo_misc.pkl',
                 dt_name='MultipleDetectedImages'):
        super(MultipleDataReaderH36M, self).__init__(n_frames, sample_stride, data_stride_train, data_stride_test,
                                                     read_confidence, dt_root, dt_file)

        self.image_stride = image_stride
        self.dt_root = dt_root
        self.dt_name = dt_name
        assert n_frames % image_stride == 0
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def read_2d(self, detector_type='hrnet'):
        if len(detector_type):
            detector_type = '_' + detector_type
        trainset = self.dt_dataset['train'][f'joints_2d{detector_type}'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test'][f'joints_2d{detector_type}'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]

        if self.read_confidence:
            if f'confidence{detector_type}' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train'][f'confidence{detector_type}'][::self.sample_stride].astype(np.float32)
                test_confidence = self.dt_dataset['test'][f'confidence{detector_type}'][::self.sample_stride].astype(np.float32)
                if len(train_confidence.shape) == 2:  # (1559752, 17)
                    train_confidence = train_confidence[:, :, None]
                    test_confidence = test_confidence[:, :, None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:, :, 0:1]
                test_confidence = np.ones(testset.shape)[:, :, 0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset

    def get_sliced_data(self):
        # 2d poses in normalized image pixel coordinate system
        # train_joints_2d_hrnet (1559752, 17, 3)
        # test_joints_2d_hrnet (566920, 17, 3)
        train_joints_2d_cpn, test_joints_2d_cpn = self.read_2d('cpn')
        train_joints_2d_sh, test_joints_2d_sh = self.read_2d('sh')
        train_joints_2d_hrnet, test_joints_2d_hrnet = self.read_2d('hrnet')
        train_joints_2d, test_joints_2d = self.read_2d('')
        # 2d poses in image pixel coordinate system
        train_joints_2d_cpn_crop = self.dt_dataset['train']['joints_2d_cpn_crop'][::self.sample_stride].astype(np.float32)
        test_joints_2d_cpn_crop = self.dt_dataset['test']['joints_2d_cpn_crop'][::self.sample_stride].astype(np.float32)

        train_joints_2d_sh_crop = self.dt_dataset['train']['joints_2d_sh_crop'][::self.sample_stride].astype(np.float32)
        test_joints_2d_sh_crop = self.dt_dataset['test']['joints_2d_sh_crop'][::self.sample_stride].astype(np.float32)

        train_joints_2d_hrnet_crop = self.dt_dataset['train']['joints_2d_hrnet_crop'][::self.sample_stride].astype(np.float32)
        test_joints_2d_hrnet_crop = self.dt_dataset['test']['joints_2d_hrnet_crop'][::self.sample_stride].astype(np.float32)

        train_joints_2d_crop = self.dt_dataset['train']['joints_2d_crop'][::self.sample_stride].astype(np.float32)
        test_joints_2d_crop = self.dt_dataset['test']['joints_2d_crop'][::self.sample_stride].astype(np.float32)
        
        # 3d poses in normalized image pixel coordinate system
        # train_joints_3d_image_norm (1559752, 17, 3)
        # test_joints_3d_image_norm (566920, 17, 3)
        train_joints_3d_image_norm = self.dt_dataset['train']['joints_3d_image_norm'][::self.sample_stride, :, :3].astype(np.float32)
        test_joints_3d_image_norm = self.dt_dataset['test']['joints_3d_image_norm'][::self.sample_stride, :, :3].astype(np.float32)
        # transformation and inverse transformation while cropping image
        train_crop_trans = self.dt_dataset['train']['crop_trans'][::self.sample_stride]
        test_crop_trans = self.dt_dataset['test']['crop_trans'][::self.sample_stride]
        train_crop_inv_trans = self.dt_dataset['train']['crop_inv_trans'][::self.sample_stride]
        test_crop_inv_trans = self.dt_dataset['test']['crop_inv_trans'][::self.sample_stride]
        # image paths
        train_images = np.array(self.dt_dataset['train']['image'][::self.sample_stride])
        test_images = np.array(self.dt_dataset['test']['image'][::self.sample_stride])
        # 2.5d something
        test_joints_2_5d_image = self.dt_dataset['test']['joints_2.5d_image'][::self.sample_stride]
        test_2_5d_factor = self.dt_dataset['test']['2.5d_factor'][::self.sample_stride]

        split_id_train, split_id_test = self.get_split_id()
        train_joints_2d_cpn_crop, test_joints_2d_cpn_crop = train_joints_2d_cpn_crop[split_id_train], test_joints_2d_cpn_crop[split_id_test]
        train_joints_2d_cpn, test_joints_2d_cpn = train_joints_2d_cpn[split_id_train], test_joints_2d_cpn[split_id_test]  # (N, 27, 17, 3)

        train_joints_2d_sh_crop, test_joints_2d_sh_crop = train_joints_2d_sh_crop[split_id_train], test_joints_2d_sh_crop[split_id_test]
        train_joints_2d_sh, test_joints_2d_sh = train_joints_2d_sh[split_id_train], test_joints_2d_sh[split_id_test]  # (N, 27, 17, 3)

        train_joints_2d_hrnet_crop, test_joints_2d_hrnet_crop = train_joints_2d_hrnet_crop[split_id_train], test_joints_2d_hrnet_crop[split_id_test]
        train_joints_2d_hrnet, test_joints_2d_hrnet = train_joints_2d_hrnet[split_id_train], test_joints_2d_hrnet[split_id_test]  # (N, 27, 17, 3)

        train_joints_2d_crop, test_joints_2d_crop = train_joints_2d_crop[split_id_train], test_joints_2d_crop[split_id_test]
        train_joints_2d, test_joints_2d = train_joints_2d[split_id_train], test_joints_2d[split_id_test]  # (N, 27, 17, 3)

        train_joints_3d_image_norm, test_joints_3d_image_norm = train_joints_3d_image_norm[split_id_train], test_joints_3d_image_norm[split_id_test]  # (N, 27, 17, 3)

        test_joints_2_5d_image = test_joints_2_5d_image[split_id_test]
        test_2_5d_factor = test_2_5d_factor[split_id_test]

        # sample again
        sampled_split_id_train = [range(elem.start + self.image_stride // 2, elem.stop, self.image_stride) for elem in split_id_train]
        sampled_split_id_test = [range(elem.start + self.image_stride // 2, elem.stop, self.image_stride) for elem in split_id_test]
        train_crop_trans, test_crop_trans = train_crop_trans[sampled_split_id_train], test_crop_trans[sampled_split_id_test]
        train_crop_inv_trans, test_crop_inv_trans = train_crop_inv_trans[sampled_split_id_train], test_crop_inv_trans[sampled_split_id_test]
        train_images, test_images = train_images[sampled_split_id_train], test_images[sampled_split_id_test]

        for phase in ['train', 'test']:
            save_path = os.path.join(self.dt_root, f"{self.dt_name}_{self.n_frames}_{self.n_frames // self.image_stride}", phase)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            phase_length = eval(f"len({phase}_joints_2d_hrnet)")
            phase_iters = np.arange(phase_length)
            for i in tqdm(phase_iters):
                data_dict = {}
                data_dict['joints_2d_cpn'] = eval(f"{phase}_joints_2d_cpn[i]")
                data_dict['joints_2d_cpn_crop'] = eval(f"{phase}_joints_2d_cpn_crop[i]")

                data_dict['joints_2d_sh'] = eval(f"{phase}_joints_2d_sh[i]")
                data_dict['joints_2d_sh_crop'] = eval(f"{phase}_joints_2d_sh_crop[i]")

                data_dict['joints_2d_hrnet'] = eval(f"{phase}_joints_2d_hrnet[i]")
                data_dict['joints_2d_hrnet_crop'] = eval(f"{phase}_joints_2d_hrnet_crop[i]")

                data_dict['joints_2d'] = eval(f"{phase}_joints_2d[i]")
                data_dict['joints_2d_crop'] = eval(f"{phase}_joints_2d_crop[i]")

                data_dict['joints_3d_image_norm'] = eval(f"{phase}_joints_3d_image_norm[i]")
                if phase == 'test':
                    data_dict['joints_2.5d_image'] = test_joints_2_5d_image[i]
                    data_dict['2.5d_factor'] = test_2_5d_factor[i]

                data_dict['trans'] = eval(f"{phase}_crop_trans[i]")
                data_dict['inv_trans'] = eval(f"{phase}_crop_inv_trans[i]")
                data_dict['image_path'] = eval(f"{phase}_images[i]")

                with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as fp:
                    pickle.dump(data_dict, fp)


if __name__ == '__main__':
    n_frames = 243
    image_stride = 9

    datareader = MultipleDataReaderH36M(n_frames=n_frames, image_stride=image_stride, sample_stride=1, data_stride_train=n_frames // 3, data_stride_test=n_frames)
    datareader.get_sliced_data()
