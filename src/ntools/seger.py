import os
import torch
import time
import functools
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist
from skimage.morphology import ball, binary_dilation, binary_erosion
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from tqdm import tqdm
from ntools.patch import patchify_without_splices, get_patch_rois
from ntools.dbio import segs2db
from ntools.image_reader import wrap_image
from ntools.vis import show_segs_as_instances


# normaliz layer
def get_norm_layer(norm_type='instance', dim=2):
    if dim == 2:
        BatchNorm = nn.BatchNorm2d
        InstanceNorm = nn.InstanceNorm2d
    elif dim == 3:
        BatchNorm = nn.BatchNorm3d
        InstanceNorm = nn.InstanceNorm3d
    else:
        raise Exception('Invalid dim.')
    
    if norm_type == 'batch':
        norm_layer = functools.partial(BatchNorm, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm, affine=False, track_running_stats=False)
    elif norm_type == 'identity':
        def norm_layer(x):
            return lambda t:t
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# Conv block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_type='batch', dim=2):
        super(DoubleConv, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')
        
        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        self.conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# Unet
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], *, norm_type='batch', dim=2):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        if dim == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            self.MaxPool = nn.MaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            self.MaxPool = nn.MaxPool3d
        else:
            raise Exception('Invalid dim.')

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, norm_type=norm_type, dim=dim))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.ups.append(
                ConvTranspose(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature, norm_type=norm_type, dim=dim))

        self.final_conv = nn.Sequential(
            Conv(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = self.MaxPool(kernel_size=2)(x)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            if x.shape != skip.shape:
                x = nn.functional.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)

        x = self.final_conv(x)
        return x



class Seger():
    def __init__(self,ckpt_path,bg_thres,l_clip=None,device=None):
        if 'tiny' in ckpt_path:
            model_dims = [32,64,128]
        elif 'medium' in ckpt_path:
            model_dims = [32,64,128,256]
        elif 'dumpy' in ckpt_path:
            model_dims = [64,128,256]
        model = UNet(1, 1, model_dims, norm_type='batch', dim=3)
        if device==None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
        model.to(self.device)
        model.eval()
        self.model = model
        self.bw = 14 #border width (128-100)//2
        self.bg_thres = bg_thres
        self.l_clip = l_clip


    def preprocess(self,img,percentiles=[0.1,1.0]):
        # input img nparray [0,65535]
        # output img tensor [0,1]
        flattened_arr = np.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))-1
        if flattened_arr[clip_high]<self.bg_thres:
            return None
        clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])
        if self.l_clip is not None:
            clipped_arr = np.clip(clipped_arr,self.l_clip,65535)
        min_value = np.min(clipped_arr)
        max_value = np.max(clipped_arr)
        filtered = clipped_arr
        img = (filtered-min_value)/(max_value-min_value)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0)
        return img


    def auto_contrast(self,img):
        pass


    def postprocess(self,mask,min_size=50):
        labeled_mask, num_features = label(mask,return_num=True)
        region_sizes = np.bincount(labeled_mask.ravel())
        small_regions = np.where(region_sizes < min_size)[0]
        for region in small_regions:
            mask[labeled_mask == region] = 0
        return mask
    

    def get_mask(self,img,thres=None):
        img_in = self.preprocess(img)
        if img_in != None:
            with torch.no_grad():
                tensor_out = self.model(img_in.to(self.device)).cpu()
            prob = tensor_out.squeeze(0).squeeze(0)
            if thres==None:
                return prob.detach().numpy()
            else:
                prob[prob>=thres]=1
                prob[prob<thres]=0
                return prob.detach().numpy()
        else:
            return np.zeros_like(img)


    def get_large_mask(self,img):
        '''
        process one large cube (D,W,H>100) with border (default 14), return mask
        '''
        block_size = 100
        border_size = self.bw
        bordered_size = img.shape
        actual_size = [i-border_size*2 for i in bordered_size]
        block_rois = get_patch_rois([border_size,border_size,border_size]+actual_size,block_size)
        large_mask = np.zeros(img.shape,dtype=np.uint8)
        for roi in block_rois:
            tg_size = self.bw # (128-100)//2
            # add border if possible
            x1,x2,y1,y2,z1,z2 = roi[0],roi[0]+roi[3],roi[1],roi[1]+roi[4],roi[2],roi[2]+roi[5]
            x1 = max(0,x1-tg_size)
            y1 = max(0,y1-tg_size)
            z1 = max(0,z1-tg_size)
            x2 = min(img.shape[0],x2+tg_size)
            y2 = min(img.shape[1],y2+tg_size)
            z2 = min(img.shape[2],z2+tg_size)

            block = img[x1:x2,y1:y2,z1:z2]

            x1_pad = roi[0]-x1
            y1_pad = roi[1]-y1
            z1_pad = roi[2]-z1
            x2_pad = x2-roi[0]-roi[3]
            y2_pad = y2-roi[1]-roi[4]
            z2_pad = z2-roi[2]-roi[5]

            pad_widths = [
                (tg_size-x1_pad, tg_size-x2_pad),
                (tg_size-y1_pad, tg_size-y2_pad),
                (tg_size-z1_pad, tg_size-z2_pad)
            ]
            
            # if img.shape%block_size != 0, pad to target size

            ap = [] # additional padding
            for i, (p1,p2) in enumerate(pad_widths):
                res = block_size+tg_size*2 - (block.shape[i]+p1+p2)
                ap.append(res)
                if res!=0:
                    pad_widths[i] = (p1,p2+res)

            padded_block = np.pad(block, pad_widths, mode='reflect')

            mask = self.get_mask(padded_block,thres=0.5)
            mask = mask.astype(np.uint8)
            mask = mask[tg_size:-tg_size-ap[0],tg_size:-tg_size-ap[1],tg_size:-tg_size-ap[2]]
            large_mask[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[4],roi[2]:roi[2]+roi[5]] = mask
        processed_mask = self.postprocess(large_mask)
        # processed_mask = large_mask
        final_mask = processed_mask[border_size:-border_size,border_size:-border_size,border_size:-border_size]

        return final_mask



    def mask_to_segs(self, mask, offset=[0,0,0]):
        '''
        segment:
        {
            sid: int,
            points: [head,...,tail],
            sampled_points: points[::interval]
            nbrs = [[index_of_point,sid],...],
        }
        '''

        interval = 3
        # remove border
        # border_size = [0,0,3]
        # if border_size>0:
        #     mask[:border_size[0], :, :] = 0
        #     mask[-border_size[0]:, :, :] = 0

        #     mask[:, :border_size[1], :] = 0
        #     mask[:, -border_size[1]:, :] = 0

        #     mask[:, :, :border_size[2]] = 0
        #     mask[:, :, -border_size[2]:] = 0

        x_border = 3
        y_border = 3
        z_border = 3


        skel = skeletonize(mask)
        skel[:, :, :x_border] = 0
        skel[:, :, -x_border:] = 0
        skel[:, :, :y_border] = 0
        skel[:, :, -y_border:] = 0
        skel[:, :, :z_border] = 0
        skel[:, :, -z_border:] = 0

        labels = label(skel, connectivity=3)
        regions = regionprops(labels)

        segments = []
        for region in regions:
            points = region.coords
            distances = cdist(points, points)
            adjacency_matrix = distances <= 1.8 # sqrt(3)
            np.fill_diagonal(adjacency_matrix, 0)
            graph = nx.from_numpy_array(adjacency_matrix.astype(np.uint8))
            # keep only DFS tree
            spanning_tree = nx.minimum_spanning_tree(graph, algorithm='kruskal', weight=None)
            graph.remove_edges_from(set(graph.edges) - set(spanning_tree.edges))
            branch_nodes = [node for node, degree in graph.degree() if degree >= 3]
            branch_nbrs = []
            for node in branch_nodes:
                branch_nbrs += list(graph.neighbors(node))

            for bn in branch_nodes:
                if len(list(graph.neighbors(node)))==3:
                    segments.append(
                        {
                            'sid' : None,
                            'points' : [[i+j for i,j in zip(points[bn],offset)]],
                            'sampled_points' : [[i+j for i,j in zip(points[bn],offset)]]
                        }
                    )

            graph.remove_nodes_from(branch_nbrs)
            graph.remove_nodes_from(branch_nodes)

            connected_components = list(nx.connected_components(graph))

            for nodes in connected_components:
                if len(nodes)<=interval*2:
                    continue
                subgraph = graph.subgraph(nodes).copy()
                end_nodes = [node for node, degree in subgraph.degree() if degree == 1]
                if (len(end_nodes)!=2):
                    continue
                path = nx.shortest_path(subgraph, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
                # path to segment
                seg_points = np.array([points[i].tolist() for i in path])
                seg_points = seg_points + np.array(offset)
                seg_points = seg_points.tolist()
                sampled_points = seg_points[:-(interval-1):interval]
                sampled_points.append(seg_points[-1])
                segments.append(
                    {
                        'sid' : None,
                        'points' : seg_points,
                        'sampled_points' : sampled_points
                    }
                )
        return skel, segments



    def process_whole(self,image_path,roi=None,dec=None):
        '''
        cut whole brain image to [300,300,300] cubes without splices (z coordinates % 300 == 0)
        '''
        image = wrap_image(image_path)
        if roi==None:
            image_roi = image.roi
        else:
            image_roi = roi
        offset = image_roi[0:3]
        bounds = [i+j for i,j in zip(offset,image_roi[3:])]
        rois = patchify_without_splices(image_roi,[300,300,300])
        # pad rois
        segs = []
        for roi in tqdm(rois):
            roi[:3] = [i-self.bw for i in roi[:3]]
            roi[3:] = [i+self.bw*2 for i in roi[3:]]
            padded_block = image.from_roi(roi,padding='reflect')
            if dec!=None:
                padded_block = dec.process_img(padded_block)
            mask = self.get_large_mask(padded_block)
            _, segs_in_block = self.mask_to_segs(mask,offset=[i+self.bw for i in roi[:3]])
            segs+=segs_in_block
        
        for i,seg in enumerate(segs):
            seg['sid'] = i

        return segs


def command_line_interface():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="args for seger")
    parser.add_argument('-weight_path', type=str, default=None, help="path to weight of the segmentation model")
    parser.add_argument('-image_path', type=str, help="path to the input image, only zarr, ims, tif are currently supported")
    parser.add_argument('-db_path', type=str, default=None, help="path to the output database file")
    parser.add_argument('-roi', type=int, nargs='+', default=None, help="image roi, if kept None, process the whole image")
    parser.add_argument('-l_clip', type=int, default=None, help="value to clip voxel values, use this to repress bright noise")
    parser.add_argument('-vis', action='store_true', default=False, help="whether to visualize result after segmentation")
    args = parser.parse_args()
    if args.weight_path is None:
        args.weight_path = os.path.join(package_dir,'universal_tiny.pth')

    print(f"Using weight: {args.weight_path}")
    print(f"Processing image: {args.image_path}, roi: {args.roi}")

    seger = Seger(args.weight_path,l_clip=args.l_clip,bg_thres=150) # bg_thres is used to filter out empty image like image borders
    segs = seger.process_whole(args.image_path, roi=args.roi)

    if args.db_path is not None:
        print(f"Saving {len(segs)} segs to {args.db_path}")
        segs2db(segs,args.db_path)


    if args.vis:
        import napari
        viewer = napari.Viewer(ndisplay=3)
        image = wrap_image(args.image_path)
        if args.roi is None:
            args.roi = image.roi
        if (np.array(args.roi[3:])<np.array([1024,1024,1024])).all():
            img = image.from_roi(args.roi)
            image_layer = viewer.add_image(img)
            image_layer.translate = args.roi[0:3]
        else:
            print(f"image size {args.roi[3:]} is too large to render")
        seg_points = []
        for seg in segs:
            seg_points.append(seg['sampled_points'])
        show_segs_as_instances(seg_points, viewer, size=2)
        napari.run()



if __name__ == '__main__':
    from ntools.vis import show_segs_as_instances

    # seger = Seger('src/weights/rm009_tiny.pth',bg_thres=150)
    # seger = Seger('src/weights/universal_dumpy.pth',bg_thres=150)
    seger = Seger('src/weights/universal_medium.pth',bg_thres=150)
    # seger = Seger('src/weights/z002_tiny.pth',bg_thres=150)
    # seger = Seger('src/weights/lzh_tiny.pth',bg_thres=150)


    # '''
    image_path = '/home/bean/workspace/data/z002.ims'
    # roi = [5280,4000,8480,500,500,500] # cell body
    # roi = [3500,6200,7400,500,500,500] # cells 200
    # roi = [3800,5300,11000,500,500,500] # cells 300
    # roi = [7000,6689,4500,500,500,500] # vessel with bright noise
    # roi = [6200,6300,9200,500,500,500] # vessel
    # roi = [6500,7300,10500,500,500,500] # vessel
    # roi = [2800,3100,10400,500,500,500] # cortex
    # roi = [3800,3200,7200,500,500,300] # sparse axon
    # roi = [3200,4000,7700,500,500,500] # cortex
    # roi = [7000,5200,7600,500,500,500] # weak signal
    # roi = [2300,3600,10000,500,500,500] # weak cortex
    # roi = [3050,4300,8400,128,128,128]
    # roi = [3000,4200,8000,300,300,300] # missed segment
    roi = [4400,5900,7200,500,500,500] # close axons
    # roi = [3700,4300,7800,300,300,300] # axons
    # roi = [5900,2000,6000,500,500,500] # weak cortex signal
    # '''


    '''
    image_path = '/home/bean/workspace/data/mouse_lzh.ims'
    # roi = [6200,3600,9000,500,500,500]
    # roi = [7200,3600,11100,500,500,500]
    # roi = [7400,3000,11000,500,500,500] # sparse cortex
    # roi = [4000,6100,5000,500,500,500] # more sparse cortex
    '''


    '''
    image_path = '/home/bean/workspace/data/ROI1.ims'
    offset = [51500,22500,59400]
    # roi = [52400,24300,64000,500,500,500]
    roi = [54700,23600,60000,500,500,500]
    roi[0:3] = [i-j for i,j in zip(roi[0:3],offset)]
    '''

    
    '''
    image_path = '/home/bean/workspace/data/mouse_3.ims'
    roi = [7000,5600,10000,300,300,300]
    '''

    # image_path = '/home/bean/workspace/data/seg_datasets/c002_labeled/skels/img_1.tif'
    
    image = wrap_image(image_path)
    segs = seger.process_whole(image_path, roi=roi)
    img = image.from_roi(roi)

    import napari
    viewer = napari.Viewer(ndisplay=3)
    image_layer = viewer.add_image(img)
    image_layer.translate = roi[0:3]
    seg_points = []
    for seg in segs:
        seg_points.append(seg['sampled_points'])
    show_segs_as_instances(seg_points,viewer,size=2)
    napari.run()

