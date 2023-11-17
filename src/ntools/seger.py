import torch
import torch.nn as nn
import functools
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist
from skimage.morphology import ball
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops


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


    def preprocess(self,img,percentiles=[0.01,0.9999],radius=2):
        # input img [0,65535]
        # output img [0,1]
        flattened_arr = np.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))
        clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])
        filtered = median_filter(clipped_arr,footprint=ball(radius),mode='reflect')
        min_value = np.min(filtered)
        max_value = np.max(filtered)
        img = (filtered-min_value)/(max_value-min_value)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0)
        return img
    
    def get_mask(self,img,thres=None):
        img_in = self.preprocess(img).to(self.device)
        tensor_out = self.model(img_in).cpu()
        prob = tensor_out.squeeze(0).squeeze(0)
        if thres==None:
            return prob.detach().numpy()
        else:
            prob[prob>=thres]=1
            prob[prob<thres]=0
            return prob.detach().numpy()



def get_points(image,seger,roi):
    mask = seger.get_mask(image,thres=0.8)
    skel = skeletonize(mask)
    labels = label(skel, connectivity=3)
    print(labels.shape)
    regions = regionprops(labels)
    for props in regions:
        print(props.centroid)
        print(props.bbox)
        print(props.area)


    # offset = roi[0:3]
    # points = np.array([[coord[0]+offset[0],coord[1]+offset[1],coord[2]+offset[2]] for coord in points])
    return skel


if __name__ == '__main__':
    import numpy as np
    from ntools.read_zarr import Image
    image_path = '/Users/bean/workspace/data/axon.zarr'
    roi = [47690, 34158, 39542] + [64,64,64]
    image = Image(image_path)
    img = image.from_roi(roi)
    seger = Seger('src/weights/unet_seg.pth')
    skel = get_points(img,seger,roi=[0,0,0,0,0,0])
    import napari
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img)
    viewer.add_image(skel)
    napari.run()

