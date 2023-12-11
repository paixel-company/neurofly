import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import numpy as np

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

        # self.final_conv = nn.Sequential(
        #     Conv(features[0], out_channels, kernel_size=1),
        #     nn.ReLU()
        # )
        self.final_conv = Conv(features[0], out_channels, kernel_size=1)


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



def define_G(netG, in_channels, out_channels, features, norm_type='batch', *, dim=2):
    if netG =='unet':
        net=UNet(in_channels, out_channels, features, norm_type=norm_type, dim=dim)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return net





class Deconver():
    def __init__(self,weight_path,device=None):
        if device==None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        model = define_G('unet', 1, 1, [32, 64, 128], dim=3)
        ckpt = torch.load('src/weights/self_net_3d.pkl', map_location=self.device)
        model.load_state_dict(ckpt)
        model.to(self.device)
        model.eval() 
        self.model = model
    
    def preprocess(self,img):
        # input img [0,65535]
        # output img [0,1]
        percentiles = [0.01,0.9999]
        flattened_arr = np.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))
        clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])

        min_value = np.min(clipped_arr)
        max_value = np.max(clipped_arr) 
        img = (clipped_arr-min_value)/(max_value-min_value)
        return img
    
    def process(self,img):
        lr = self.preprocess(img)
        lr = torch.from_numpy(lr.astype(np.float32)).to(self.device)[None,None,...]
        out = self.model(lr)
        out = out.squeeze(0).squeeze(0).detach().cpu().numpy() 
        return out


if __name__ == '__main__':
    from tifffile import imread
    import napari

    viewer = napari.Viewer()

    img_path = '/home/bean/workspace/data/dirty.tif'
    img = imread(img_path)
    img = img[0:128,0:256,0:256]


    dec = Deconver('src/weights/self_net_3d.pkl')
    # device = torch.device('cuda:0')
    # model = define_G('unet', 1, 1, [32, 64, 128], dim=3)
    # ckpt = torch.load('src/weights/self_net_3d.pkl', map_location=device)
    # model.load_state_dict(ckpt)
    # model.to(device)
    # model.eval()

    # lr = preprocess(img)
    # lr = torch.from_numpy(lr.astype(np.float32)).to(device)[None,None,...]
    # out = model(lr)
    # out = out.squeeze(0).squeeze(0).detach().cpu().numpy()
    out = dec.process(img)

    viewer.add_image(img)
    viewer.add_image(out)
    napari.run()

