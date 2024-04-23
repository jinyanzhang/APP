import os, pickle, cv2, sys
sys.path.append('/home/xxxxxx/MotionAGFormer')
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class GenSequenceDataset(Dataset):
    def __init__(self,
                 data_dir='/home/xxxxxx/MotionAGFormer/data/motion3d/3DHPImages_81_81',
                 images_dir='/home/xxxxxx/data-home/MPI_INF_3DHP/images_384_288'):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.images_dir = images_dir
        
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
            image = (image - self.hrnet_mean) / self.hrnet_std
            image = image.transpose((2, 0, 1))
            images[i] = image
            f_images[i] = np.flip(image, -1).copy()
            
        pkl_file = self.pkl_files[index][:-4]
        np.savez_compressed(f'{write_dir}/{pkl_file}.npz', f=images)
        np.savez_compressed(f'{write_dir}/{pkl_file}f.npz', f=f_images)

        return 0


if __name__ == '__main__':
    write_dir='/data1'
    pretrained_dir='/data-home/pretrained'
    data_dir='/data1/motion3d/MultipleDetectedImages_243_27'
    # /data1/images_256_192_hrnet_False_243
    write_dir = f"{write_dir}/{data_dir.split('/')[-1]}_images_256_192"
    
    os.makedirs(write_dir + '/train', exist_ok=True)
    os.makedirs(write_dir + '/test', exist_ok=True)
    
    dataset = GenSequenceDataset(
        data_dir=data_dir,
        images_dir='/data1/Human3.6M/images_384_288')
    
    dataloader = DataLoader(dataset, num_workers=14, persistent_workers=True, prefetch_factor=2, shuffle=False, batch_size=1)
    for _ in tqdm(dataloader):
        pass