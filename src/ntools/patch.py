import numpy as np


def get_patch_coords(roi,block_size):
    volume_size = roi[3:6]
    origin = roi[0:3]
    grid_count = [i//block_size if i%block_size==0 else i//block_size+1 for i in volume_size]
    hist = np.zeros(grid_count, np.uint16)
    indices = np.where(hist==0)
    indices = np.array(indices).transpose()*block_size
    indices = indices[indices[:,2].argsort()]
    return indices


def get_patch_rois(roi,block_size):
    volume_size = roi[3:6]
    origin = roi[0:3]
    upper_bound = [i+j for i,j in zip(origin,volume_size)]
    block_coords = get_patch_coords(roi,block_size)
    rois = []
    for coord in block_coords:
        c1 = [i+j for i,j in zip(coord,origin)]
        c2 = [i+block_size if i+block_size<j else j for i,j in zip(c1,upper_bound)]
        size = [j-i for i,j in zip(c1,c2)]
        rois.append(c1+size)
    return rois

