import os
import numpy as np
import pickle


# if __name__ == '__main__':
#     npz_file = np.load('/data1/3DHPImages_81_9/test/00000000.npz')
#     f_maps = [npz_file[f'f_{i}'] for i in range(4)]
#     zero_f_maps = {f'f_{i}': f_maps[i][0].copy() for i in range(4)}
#     np.savez_compressed('zero_featuremaps.npz', **zero_f_maps)

def read_pkl(path):
    with open(path, 'rb') as file:
        result = pickle.load(file)
    file.close()
    return result


if __name__ == '__main__':
    indices = []
    pkl_files = os.listdir('/data1/motion3d/3DHPImages_81_9/test')
    counter = 0
    for i, pkl_file in enumerate(pkl_files):
        read_path = os.path.join('/data1/motion3d/3DHPImages_81_9/test', pkl_file)
        obj = read_pkl(read_path)['image']
        if 'pad' in obj:
            counter += 1
            indices.append(i)
    print(counter, len(pkl_files), counter / len(pkl_files))
    np.save('indices.npy', indices)