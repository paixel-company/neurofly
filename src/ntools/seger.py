import torch
import torch.nn as nn
import functools
import numpy as np
import networkx as nx
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist
from scipy.ndimage import label
from skimage.morphology import ball
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from tqdm import tqdm
from ntools.patch import patchify_without_splices, get_patch_rois
from ntools.read_zarr import Image
from empatches import EMPatches
from ntools.dbio import segs2db
import time

# ==========
# normaliz layer
# ==========
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

# ==========
# UNet
# ==========
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
    def __init__(self,ckpt_path,device=None):
        model = UNet(1, 1, [64,128,256,512], norm_type='batch', dim=3) 
        if device==None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
        model.to(self.device)
        model.eval()
        self.model = model
        self.bw = 14 #border width


    def preprocess(self,img,percentiles=[0.01,0.9999],radius=None):
        # input img [0,65535]
        # output img [0,1]
        # median filter can increase accuracy a little bit with 3-4 times time consumption
        flattened_arr = np.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))
        if flattened_arr[clip_high]<1500:
            return None
        clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])
        min_value = np.min(clipped_arr)
        max_value = np.max(clipped_arr)
        if radius != None and max_value<5000:
            filtered = median_filter(clipped_arr,footprint=ball(radius),mode='reflect')
        else:
            filtered = clipped_arr 
        max_value = max(max_value,2500)
        img = (filtered-min_value)/(max_value-min_value)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0)
        return img


    def postprocess(self,mask,min_size=50):
        labeled_mask, num_features = label(mask,return_num=True)
        region_sizes = np.bincount(labeled_mask.ravel())
        small_regions = np.where(region_sizes < min_size)[0]
        for region in small_regions:
            mask[labeled_mask == region] = 0
        return mask
    

    def get_mask(self,img,thres=None):
        img_in = self.preprocess(img,radius=None)
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
        border_size = 1

        mask[:border_size, :, :] = 0
        mask[-border_size:, :, :] = 0

        mask[:, :border_size, :] = 0
        mask[:, -border_size:, :] = 0

        mask[:, :, :border_size] = 0
        mask[:, :, -border_size:] = 0


        skel = skeletonize(mask)
        labels = label(skel, connectivity=3)
        regions = regionprops(labels)

        segments = []
        for region in regions:
            points = region.coords
            distances = cdist(points, points)
            adjacency_matrix = distances <= 1.8
            np.fill_diagonal(adjacency_matrix, 0)
            graph = nx.from_numpy_matrix(adjacency_matrix.astype(np.uint8))
            # keep only DFS tree
            spanning_tree = nx.minimum_spanning_tree(graph, algorithm='kruskal', weight=None)
            graph.remove_edges_from(set(graph.edges) - set(spanning_tree.edges))

            branch_nodes = [node for node, degree in graph.degree() if degree >= 3]

            # branch_nbrs = []
            # for node in branch_nodes:
            #     branch_nbrs += list(graph.neighbors(node))
            # graph.remove_nodes_from(branch_nodes)

            graph.remove_nodes_from(branch_nodes)
            connected_components = list(nx.connected_components(graph))

            for nodes in connected_components:
                if len(nodes)<=interval:
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
                sampled_points = seg_points[:-(interval-2):interval]
                sampled_points.append(seg_points[-1])
                segments.append(
                    {
                        'sid' : None,
                        'points' : seg_points,
                        'sampled_points' : sampled_points, 
                        'nbrs' : [],
                    }
                )
        return skel, segments


    def process_whole(self,zarr_path,roi=None,dec=None):
        '''
        cut whole brain image to [300,300,300] cubes without splices (z coordinates % 300 == 0)
        '''
        image = Image(zarr_path)
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
            tg_size = self.bw # (128-100)//2
            # add border if possible
            x1,x2,y1,y2,z1,z2 = roi[0],roi[0]+roi[3],roi[1],roi[1]+roi[4],roi[2],roi[2]+roi[5]
            x1 = max(offset[0],x1-tg_size)
            y1 = max(offset[1],y1-tg_size)
            z1 = max(offset[2],z1-tg_size)
            x2 = min(bounds[0],x2+tg_size)
            y2 = min(bounds[1],y2+tg_size)
            z2 = min(bounds[2],z2+tg_size)

            block = image[x1:x2,y1:y2,z1:z2]

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
                res = roi[i+3]+tg_size*2 - (block.shape[i]+p1+p2)
                ap.append(res)
                if res!=0:
                    pad_widths[i] = (p1,p2+res)
            padded_block = np.pad(block, pad_widths, mode='reflect')
            if dec!=None:
                padded_block = dec.process_img(padded_block)
            mask = self.get_large_mask(padded_block)
            _, segs_in_block = self.mask_to_segs(mask,offset=roi[0:3])
            segs+=segs_in_block
        
        for i,seg in enumerate(segs):
            seg['sid'] = i

        return segs



if __name__ == '__main__':
    import numpy as np
    from ntools.vis import show_segs_as_instances
    from tifffile import imread
    seger = Seger('src/weights/unet_seg2.pth')

    # image_path = '/home/bean/workspace/data/axon.zarr'
    # roi = [47800, 34200, 39600] + [328,328,328]

    # image_path = '/home/bean/workspace/data/axon2.zarr'
    # roi = [53000, 23500, 56500] + [500,500,300]

    # image_path = '/home/bean/workspace/data/roi_dense1.zarr'
    # roi = [52500, 22000, 60500] + [300,300,300]
    '''

    image_path = '/home/bean/workspace/data/test.zarr'
    image = Image(image_path)
    roi = image.roi[0:3] + [328,328,328]

    image = Image(image_path)
    img = image.from_roi(roi)

    # dimg = dec.process_img(img)
    dimg = img

    # img = imread('/home/bean/workspace/data/deconvolved.tif')
    # size = 200
    # img = img[:size,:size,:size]
    


    # mask = seger.get_large_mask(img)
    dmask = seger.get_large_mask(dimg)

    # skel,segments = seger.mask_to_segs(mask)
    skel,dsegments = seger.mask_to_segs(dmask)

    # segs = []
    # for seg in segments:
    #     segs.append(seg['points'])
        
    dsegs = []
    for seg in dsegments:
        dsegs.append(seg['points'])


    import napari
    viewer = napari.Viewer(ndisplay=3)
    # show_segs_as_instances(segs,viewer)
    show_segs_as_instances(dsegs,viewer)
    viewer.add_image(img)
    viewer.add_image(dimg)
    viewer.add_image(skel)
    napari.run()
    '''

    from ntools.neuron import save_segs
    # image_path = '/home/bean/workspace/data/ROI1_whole.zarr'
    image_path = '/home/bean/workspace/data/test.zarr'
    roi = [38300,20600,67100,300,300,300] 
    segs = seger.process_whole(image_path, roi=roi)
    image = Image(image_path)
    img = image.from_roi(roi)
    # segs = seger.process_whole(image_path)

    # save_segs(segs,'tests/test.json')
    # segs2db(segs,'tests/test.db')

    import napari
    viewer = napari.Viewer(ndisplay=3)
    image_layer = viewer.add_image(img)
    image_layer.translate = roi[0:3]
    # show_segs_as_instances(segs,viewer)
    seg_points = []
    for seg in segs:
        seg_points.append(seg['points'])
    show_segs_as_instances(seg_points,viewer)
    napari.run()

