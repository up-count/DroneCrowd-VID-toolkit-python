import os
import numpy as np
from scipy import io
from tqdm import tqdm
from glob import glob
import warnings


def save_anno_res(gt_path, det_path, list_path, dataset):
    # Process the annotations and groundtruth
    seq_list = sorted(np.loadtxt(list_path))

    print('Loading annotations and detections...')

    all_gt = {}
    all_det = {}

    for seq_id in tqdm(seq_list):
        seq_id = int(seq_id)
        
        if dataset == 'dronecrowd':
            seq_name = f'{seq_id:05d}'
        elif dataset == 'upcount':
            seq_name = f'{seq_id:04d}'
        
        anno = io.loadmat(os.path.join(gt_path, f'{seq_name}.mat'))['anno']

        if dataset == 'dronecrowd':
            gt_ = np.column_stack([anno[:, 0] + 1, anno[:, 1:7], np.zeros((anno.shape[0], 1))])
        elif dataset == 'upcount':
            gt_ = np.column_stack([anno[:, 0], anno[:, 1:7], np.zeros((anno.shape[0], 1))])

        det = []
        gt = []

        for k in range(int(gt_[:, 0].min()), int(gt_[:, 0].max()) + 1):
            if dataset == 'dronecrowd':
                cur_det = np.loadtxt(os.path.join(det_path, f'img{seq_id:03d}{k:03d}_loc.txt'))
            elif dataset == 'upcount':
                path_ = glob(os.path.join(det_path, f'{seq_id:04d}__{k:04d}__*_loc.txt'))[0]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cur_det = np.loadtxt(path_)
                    
            if cur_det.shape[0] == 0:
                cur_det = np.zeros((0, 3))

            cur_det = cur_det.reshape(-1, 3)
                
            if cur_det.shape[1] != 3:
                raise ValueError(
                    f'Wrong format of detection results: {cur_det.shape}')

            idx = gt_[:, 0] == k

            gt.append(np.array(gt_[idx]))

            num_det = cur_det.shape[0]

            if num_det > 0:
                curdet = [
                    np.tile([k, -1], (num_det, 1)),
                    np.reshape(cur_det[:, 0] - 10, (-1, 1)),
                    np.reshape(cur_det[:, 1] - 10, (-1, 1)),
                    np.tile([20, 20], (num_det, 1)),
                    np.reshape(cur_det[:, 2], (-1, 1)),
                ]

                det.append(np.concatenate(curdet, axis=1))

        all_gt[seq_id] = np.concatenate(gt, axis=0)
        all_det[seq_id] = np.concatenate(det, axis=0)

    return all_gt, all_det
