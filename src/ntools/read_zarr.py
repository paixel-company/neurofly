import zarr
import numpy as np

class Image():
    '''
    zarr.attrs['roi'] = [x_offset,y_offset,z_offset,x_size,y_size,z_size]
    To load image directly from global coordinates, wrap .zarr object in this class.
    '''
    def __init__(self,zarr_path):
        self.image = zarr.open(zarr_path, mode='r') 
        self.roi = self.image.attrs['roi']
        print(f'Image ROI: {self.roi[0:3]} to {[i+j for i,j in zip(self.roi[0:3],self.roi[3:6])]}')
        self.shape = self.roi[3:6]
    
    def __getitem__(self, indices):
        x_min, x_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        z_min, z_max = indices[2].start, indices[2].stop
        x_slice = slice(x_min-self.roi[0],x_max-self.roi[0])
        y_slice = slice(y_min-self.roi[1],y_max-self.roi[1])
        z_slice = slice(z_min-self.roi[2],z_max-self.roi[2])
        return self.image[x_slice,y_slice,z_slice]

    def from_roi(self, coords):
        # coords: [x_offset,y_offset,z_offset,x_size,y_size,z_size]
        x_min, x_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        z_min, z_max = coords[2], coords[5]+coords[2]
        # add padding
        [xlb,ylb,zlb] = self.roi[0:3] 
        [xhb,yhb,zhb] = [i+j for i,j in zip(self.roi[:3],self.roi[3:])]

        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)

        x_slice = slice(x_min-self.roi[0]+xlp,x_max-self.roi[0]-xhp)
        y_slice = slice(y_min-self.roi[1]+ylp,y_max-self.roi[1]-yhp)
        z_slice = slice(z_min-self.roi[2]+zlp,z_max-self.roi[2]-zhp) 
        img = self.image[x_slice,y_slice,z_slice]

        padded = np.pad(img, ((xlp, xhp), (ylp, yhp), (zlp, zhp)), 'constant')

        return padded

    def from_local(self, coords):
        # coords: [x_offset,y_offset,z_offset,x_size,y_size,z_size]
        x_min, x_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        z_min, z_max = coords[2], coords[5]+coords[2]
        x_slice = slice(x_min,x_max)
        y_slice = slice(y_min,y_max)
        z_slice = slice(z_min,z_max) 
        return self.image[x_slice,y_slice,z_slice]



# import napari
# viewer = napari.Viewer()
# image_path = '/home/bean/workspace/data/test.zarr'
# image = Image(image_path)
# center = [i+j for i,j in zip(image.roi[:3],image.roi[3:])]
# SIZE = 32
# roi = [i-SIZE for i in center] + [SIZE*2,SIZE*2,SIZE*2]
# img = image.from_roi(roi)
# viewer.add_image(img)
# napari.run()