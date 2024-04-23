import torch, faulthandler
import os, pickle, cv2, sys
sys.path.append('/home/xxxxxx/MotionAGFormer')
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model.backbones import hrnet, cpn, sh
from einops import rearrange

import multiprocessing
import threading


def write_feature(queue, write_dir):
    while True:
        f_maps, pkl_file, flipped = queue.get()
        if f_maps is None:
            break
        tmp_f_maps = {
            'f_0': f_maps[0].copy(),
            'f_1': f_maps[1].copy(),
            'f_2': f_maps[2].copy(),
            'f_3': f_maps[3].copy(),
        }
        np.savez_compressed(f'{write_dir}/{pkl_file}{flipped}.npz', **tmp_f_maps)
            


class GenSequenceDataset(Dataset):
    def __init__(self,
                 data_dir='/home/xxxxxx/MotionAGFormer/data/motion3d/3DHPImages_81_81',
                 images_dir='/home/xxxxxx/data-home/MPI_INF_3DHP/images_384_288',
                 detector_type='hrnet'):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.detector_type = detector_type
        
        # mean stds
        self.hrnet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.hrnet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.sh_mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float32)
        self.sh_std = np.array([0.2458, 0.2410, 0.2468], dtype=np.float32)
        
        self.pkl_files = []
        # [/home/xxxxxx/MotionAGFormer/data/motion3d/3DHPImages_81_81/train/*.pkl]
        for data_split in ['train', 'test']:
            data_path = f'{self.data_dir}/{data_split}'
            split_pkl_files = os.listdir(data_path)
            split_pkl_files = [data_split + '/' + elem for elem in split_pkl_files]
            self.pkl_files += split_pkl_files
        # self.pkl_files = self.pkl_files[11000+2337:]
        

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, index):
        pkl_file = self.pkl_files[index]
        read_path = f'{self.data_dir}/{pkl_file}'
        
        # get image path
        with open(read_path, 'rb') as file:
            image_paths = pickle.load(file)['image_path']
        file.close()
        
        # read images
        images = np.zeros((len(image_paths), 3, 256, 192), dtype=np.float32)
        f_images = np.zeros((len(image_paths), 3, 256, 192), dtype=np.float32)
        for i, image_path in enumerate(image_paths):
            image_full_path = os.path.join(self.images_dir, image_path)
            image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (192, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            if self.detector_type == 'hrnet':
                image = (image - self.hrnet_mean) / self.hrnet_std
            elif self.detector_type == 'sh':
                image = (image - self.sh_mean) / self.sh_std
            image = image.transpose((2, 0, 1))
            images[i] = image
            f_images[i] = np.flip(image, -1).copy()

        return images, f_images, index


if __name__ == '__main__':
    faulthandler.enable()
    num_threads = 16
    
    detector_type = 'cpn'
    write_dir='/data1'
    pretrained_dir='/data-home/pretrained'
    data_dir='/data1/motion3d/MultipleDetectedImages_243_27'
    # /data1/images_256_192_hrnet_False_243
    write_dir = f"{write_dir}/{data_dir.split('/')[-1]}_{detector_type}"
    
    os.makedirs(write_dir + '/train', exist_ok=True)
    os.makedirs(write_dir + '/test', exist_ok=True)
    
    # backbone models
    if detector_type == 'hrnet':
        model = hrnet.build_model('hrnet_w32_256_192')
        model = hrnet.load_model(model, f'{pretrained_dir}/coco/pose_hrnet_w32_256x192.pth')
    elif detector_type == 'cpn':
        model = cpn.build_model('cpn_50_256_192')
        model = cpn.load_model(model, f'{pretrained_dir}/coco/CPN_256x192.pth.tar')
    else:
        raise ValueError
    model = model.cuda()
    model.eval()
    
    dataset = GenSequenceDataset(
        data_dir=data_dir,
        images_dir='/data-home/Human3.6M/images_384_288',
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
        images, f_images, index = batch
        with torch.no_grad():
            images = images.squeeze(0).cuda()
            f_images = f_images.squeeze(0).cuda()
            f_maps = model(images)
            f_maps = [f.detach().cpu().numpy() for f in f_maps]

            f_f_maps = model(f_images)
            f_f_maps = [f.detach().cpu().numpy() for f in f_f_maps]
        
            pkl_file = pkl_files[index][:-4]
            queue.put([f_maps, pkl_file, ''])
            queue.put([f_f_maps, pkl_file, 'f'])
    
    for _ in range(num_threads):
        queue.put([None, None, None])
    
    for thread in save_threads:
        thread.join()

