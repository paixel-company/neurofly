import torch
import torch.nn as nn
from torch.nn import init
import functools
import math
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate
from skimage.exposure import match_histograms

class Deblur_Net(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert(n_blocks >= 0)
        super(Deblur_Net, self).__init__()
        use_bias = True

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),nn.LeakyReLU(0.2,True)]

        n_1 = 2
        for i in range(n_1):  # add layers
            model += [nn.Conv2d(ngf , ngf , kernel_size=3, stride=1, padding=1, bias=use_bias),nn.LeakyReLU(0.2,True)]

        for i in range(n_blocks):       # add ResNet blocks

            model += [Res_Block(ngf, padding_type=padding_type, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_1):  # add layers
            model += [nn.Conv2d(ngf ,ngf,kernel_size=3, stride=1,padding=1,bias=use_bias),nn.LeakyReLU(0.2,True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)



class Res_Block(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(Res_Block, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.LeakyReLU(0.2,True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



class DeConver():
    def __init__(self,weight_path='src/weights/deblur_net.pkl',device=None):
        model = Deblur_Net(input_nc=1,output_nc=1,ngf=64,use_dropout=True)
        weight_path = 'src/weights/deblur_net.pkl'
        ckpt = torch.load(weight_path, map_location='cpu')
        model.load_state_dict({k.replace('module.',''):v for k,v in ckpt.items()},strict=False)
        if device==None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        model.to(self.device)
        model.eval()
        self.model = model

    def process_img(self, img, BS=64):
        # check if it is needed
        self.classify_img(img)

        # pad to cube
        (inx, iny, inz) = img.shape
        tx = ty = tz = max(img.shape)
        px = tx - inx
        py = ty - iny
        pz = tz - inz
        img = np.pad(img, ((0, px), (0, py), (0, pz)), 'constant')

        rotated_img = rotate(img, 45, axes=(0, 2), reshape=True) # [high_res,high_res,low_res] [1.41, 1, 1.41]

        # preprocess
        # min_v = np.min(rotated_img)
        # max_v = np.max(rotated_img)
        min_v = 0
        # max_v = 16384
        # max_v = 21846
        max_v = 32768
        input_img = (rotated_img.astype(np.float32) - min_v) / (max_v - min_v)
        input_img = np.clip(input_img,0,1)
        # ----------
        input_img = np.expand_dims(input_img, axis=1)

        input_tensor = torch.from_numpy(input_img).to(self.device)

        num_b = input_tensor.shape[0]//BS
        res = input_tensor.shape[0]%BS

        out = []
        with torch.no_grad():
            for i in range(num_b):
                out.append(self.model(input_tensor[BS*i:BS*(i+1),...]))
            if res>0:
                out.append(self.model(input_tensor[-res:,...]))

        out = torch.cat(out, 0)
        out = torch.squeeze(out,dim=1)
        out = out.cpu().numpy()

        out = rotate(out, -45, axes=(0, 2), reshape=True)
        os = img.shape[1]//2
        rs = out.shape[0]//2-1
        out = out[rs-os:rs+os,:,rs-os:rs+os]
        out = out[:tx-px,:ty-py,:tz-pz]

        out = np.clip(out,0,65535/max_v)
        out = out*(max_v-min_v) + min_v
        out = match_histograms(out,img)
        out = np.clip(out,0,65535)
        out = out.astype(np.uint16)
        return out


    def classify_img(self,img):
        pass


if __name__ == '__main__':

    from tifffile import imread
    import napari

    viewer = napari.Viewer()

    img_path = '/home/bean/workspace/data/dirty.tif'
    img = imread(img_path)
    img = img[0:128,0:256,0:256]

    deconver = DeConver()
    out = deconver.process_img(img)
    print(out.shape)

    viewer.add_image(img)
    viewer.add_image(out)
    napari.run()

