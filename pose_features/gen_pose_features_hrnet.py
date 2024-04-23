import torch, faulthandler
import os, pickle, sys
sys.path.append('/home/xxxxxx/MotionAGFormer')
import numpy as np

from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model.MotionAGFormer import MotionAGFormer
from einops import rearrange
from utils.data import flip_data

import multiprocessing
import threading


def write_feature(queue, write_dir):
    while True:
        poses, pkl_file, flipped = queue.get()
        if poses is None:
            break
        np.savez_compressed(f'{write_dir}/{pkl_file}{flipped}.npz', f=poses)
            


class GenSequenceDataset(Dataset):
    def __init__(self,
                 data_dir='/home/xxxxxx/MotionAGFormer/data/motion3d/3DHPImages_81_81',
                 detector_type='hrnet'):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        if detector_type == 'gt':
            detector_type = ''
        self.detector_type = detector_type
        
        self.pkl_files = []
        # [/home/xxxxxx/MotionAGFormer/data/motion3d/3DHPImages_81_81/train/*.pkl]
        for data_split in ['train', 'test']:
            data_path = f'{self.data_dir}/{data_split}'
            split_pkl_files = os.listdir(data_path)
            split_pkl_files = [data_split + '/' + elem for elem in split_pkl_files]
            self.pkl_files += split_pkl_files
        

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, index):
        pkl_file = self.pkl_files[index]
        read_path = f'{self.data_dir}/{pkl_file}'
        
        # get motion 2d
        with open(read_path, 'rb') as file:
            motion_2d = pickle.load(file)[f'joints_2d_{self.detector_type}']
        file.close()
        f_motion_2d = flip_data(motion_2d)
        motion_2d = torch.FloatTensor(motion_2d)
        f_motion_2d = torch.FloatTensor(f_motion_2d)
        return motion_2d, f_motion_2d, index


if __name__ == '__main__':
    faulthandler.enable()
    num_threads = 16
    
    detector_type = 'hrnet'
    write_dir='/data1'
    pretrained_dir='/data-home/pretrained'
    data_dir='/data1/motion3d/MultipleDetectedImages_243_27'
    checkpoint_path = f'/data-home/checkpoints/motionagformer-b-h36m-{detector_type}.pth.tr'
    write_dir = f"{write_dir}/{checkpoint_path.split('/')[-1].split('.')[0]}"
    checkpoint = torch.load(checkpoint_path)
    
    os.makedirs(write_dir + '/train', exist_ok=True)
    os.makedirs(write_dir + '/test', exist_ok=True)
    
    model = MotionAGFormer(
            n_layers=16, dim_in=3, dim_feat=128, dim_rep=512,
            dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
            drop=0., drop_path=0., use_layer_scale=True,
            layer_scale_init_value=1e-5, use_adaptive_fusion=True,
            num_heads=8, qkv_bias=False, qkv_scale=None, hierarchical=False,
            use_temporal_similarity=True, temporal_connection_len=1,
            use_tcn=False, graph_only=False, neighbour_num=2, return_mid=True,
            num_joints=17, n_frames=243)
    state_dict = {}
    for key in checkpoint['model'].keys():
        state_dict[key.replace('module.', '')] = checkpoint['model'][key].clone()
    ret = model.load_state_dict(state_dict, strict=True)
    print(ret)
    
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f'Model parameter: {model_params/1000000:.2f}M')
    
    model = model.cuda()
    model.eval()
    
    dataset = GenSequenceDataset(
        data_dir=data_dir,
        detector_type=detector_type)
    
    pkl_files = dataset.pkl_files
    
    # create a queue for feature
    queue = multiprocessing.Queue()
    
    save_threads = []

    for _ in range(num_threads):
        thread = threading.Thread(target=write_feature, args=(queue, write_dir, ))
        save_threads.append(thread)
        thread.start()
    
    dataloader = DataLoader(dataset, num_workers=8, persistent_workers=2, shuffle=False, batch_size=1)
    for batch in tqdm(dataloader):
        poses, f_poses, index = batch
        with torch.no_grad():
            poses = poses.cuda()
            f_poses = f_poses.cuda()
            h_poses = model(poses).squeeze(0).detach().cpu().numpy()
            f_h_poses = model(f_poses).squeeze(0).detach().cpu().numpy()
        
            pkl_file = pkl_files[index][:-4]
            queue.put([h_poses, pkl_file, ''])
            queue.put([f_h_poses, pkl_file, 'f'])
    
    for _ in range(num_threads):
        queue.put([None, None, None])
    
    for thread in save_threads:
        thread.join()

