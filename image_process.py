import numpy as np
import torch


def preprocess(voxel):
    nonzero = voxel[voxel>0.0] # 平均と標準偏差の計算に輝度0は含めない
    voxel = np.clip(voxel, 0, np.mean(nonzero) + 2.0*np.std(nonzero))
    voxel = normalize(voxel, np.min(voxel), np.max(voxel))
    voxel = voxel[np.newaxis, ]
    return voxel.astype('f')

def normalize(voxel: np.ndarray, floor: int, ceil: int) -> np.ndarray:
    return (voxel - floor) / (ceil - floor)
