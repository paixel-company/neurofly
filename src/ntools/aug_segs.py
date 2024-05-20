import os
import napari
import random
import numpy as np
import argparse
from tifffile import imread, imwrite
from skimage.exposure import match_histograms
from tqdm import tqdm


def generate_random_combinations(numbers, N):
    for _ in range(N):
        combination = random.sample(numbers, 3)
        yield combination


def get_patch_coords(roi,block_size):
    volume_size = roi[3:6]
    origin = roi[0:3]
    grid_count = [i//block_size for i in volume_size]
    hist = np.zeros(grid_count, np.uint16)
    indices = np.where(hist==0)
    indices = np.array(indices).transpose()*block_size
    # indices = indices[indices[:,2].argsort()]
    return indices


def gen_dataset(source_dir, out_dir, N=512):
    skel_source_dir = os.path.join(source_dir, 'skels')
    bg_source_dir = os.path.join(source_dir, 'bg')

    os.mkdir(out_dir)
    img_dir = os.path.join(out_dir, 'img')
    mask_dir = os.path.join(out_dir, 'mask')
    bg_dir = os.path.join(out_dir, 'img_bg')

    os.mkdir(img_dir)
    os.mkdir(mask_dir)
    os.mkdir(bg_dir)

    image_paths = [os.path.join(skel_source_dir, filename) for filename in os.listdir(skel_source_dir) if 'img' in filename and 'json' not in filename]

    mask_paths = [filename.replace('img', 'mask') for filename in image_paths]

    numbers = list(range(0, len(image_paths)))
    combinations = list(generate_random_combinations(numbers, N))

    for i,[p1,p2,p3] in tqdm(enumerate(combinations)):
        img_path1 = image_paths[p1]
        img_path2 = image_paths[p2]
        mask_path1 = mask_paths[p1]
        mask_path2 = mask_paths[p2]

        img1 = imread(img_path1)
        img2 = imread(img_path2)
        mask1 = imread(mask_path1)
        mask2 = imread(mask_path2)
        
        # in case of empty dimension
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        mask1 = mask1.squeeze()
        mask2 = mask2.squeeze()

        reference = imread(image_paths[p3])
        reference = reference.squeeze()

        mask = img1 > img2
        overlapped = np.where(mask, img1, img2)
        matched = match_histograms(overlapped, reference)
        matched = matched.astype(np.uint16)

        mask = np.clip(mask1+mask2,0,1)

        image_path = os.path.join(img_dir, 'img_'+str(i+1)+'.tif')
        mask_path = os.path.join(mask_dir, 'mask_'+str(i+1)+'.tif')

        imwrite(image_path,matched,dtype=np.uint16)
        imwrite(mask_path,mask,dtype=np.uint8)


    image_paths = [os.path.join(bg_source_dir, filename) for filename in os.listdir(bg_source_dir) if '.tif' in filename]

    num = 1
    for image_path in image_paths:
        image = imread(image_path)
        size = list(image.shape)
        block_size = 128
        patch_coords = get_patch_coords([0,0,0]+size,block_size)
        for [x,y,z] in patch_coords:
            block = image[x:x+block_size,y:y+block_size,z:z+block_size]
            image_path = os.path.join(bg_dir, 'img_'+str(num)+'.tif')
            imwrite(image_path,block,dtype=np.uint16)
            num+=1
    
    print('finished')



def command_line_interface():
    parser = argparse.ArgumentParser(description="args for seger")
    parser.add_argument('-source', type=str, default=None, help="dir of labeled images")
    parser.add_argument('-out', type=str, help="dir of output augmented images")
    parser.add_argument('-n', type=int, help="number of images")
    args = parser.parse_args()

    print(f"input dir: {args.source}")
    print(f"out dir: {args.out}")
    print(f"number of images: {args.n}")
    gen_dataset(args.labeled,args.out,N=args.n)



if __name__ == '__main__':
    source_dir = '/home/bean/workspace/data/seg_datasets/rm009_sr_labeled/'
    out_dir = '/home/bean/workspace/data/seg_datasets/rm009_sr_aug/'
    gen_dataset(source_dir,out_dir,N=2048)
