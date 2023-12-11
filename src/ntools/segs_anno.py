from ntools.dbio import get_one_point, read_segs, segs_from_sids, get_random_point
from ntools.segs import SegsTree, PointsTree
from ntools.read_zarr import Image
from magicgui import magicgui, widgets
import numpy as np
import napari
import pathlib



class Annotator:
    def __init__(self, db_path, image_path, deconver):
        self.size = 32
        self.db_path = db_path
        self.image = Image(image_path)
        self.viewer = napari.Viewer(ndisplay=3)
        self.image_layer = self.viewer.add_image(np.zeros((64, 64, 64), dtype=np.uint16))
        self.dec_layer = self.viewer.add_image(np.zeros((64, 64, 64), dtype=np.uint16))
        self.point_layer = self.viewer.add_points(data=None,ndim=3,size=0.8,edge_color='black',shading='spherical',properties=None,face_colormap='turbo')
        self.add_control()
        self.deconver = deconver
        napari.run()


    def add_control(self):
        self.viewer.bind_key('r', self.refresh)
        self.point_layer.mouse_drag_callbacks.append(self.get_point_under_cursor)

        self.a = widgets.SpinBox(value=10, label="a")
        self.b = widgets.Slider(value=20, min=0, max=100, label="b")
        self.result = widgets.Label(value=self.a.value * self.b.value, label="result")
        self.button = widgets.PushButton(text="refresh")
        self.button.clicked.connect(self.deconvolve)
        self.model_path = widgets.FileEdit(label="model_path")

        self.container = widgets.Container(widgets=[self.a, self.b, self.result, self.button, self.model_path])
        self.viewer.window.add_dock_widget(self.container, area='right')


    def deconvolve(self):
        self.result.value = self.a.value * self.b.value
        self.refresh(self.viewer)
        img = self.image_layer.data
        t = self.image_layer.translate
        deconvolved = self.deconver.process(img)
        self.dec_layer.data = deconvolved
        self.dec_layer.translate = t
        self.dec_layer.reset_contrast_limits()
        self.viewer.reset_view()


    def get_point_under_cursor(self, layer, event):
        if event.button == 2:
            # remove all connected points
            index = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                layer.selected_data = [index]
                print(layer.data[index])
                print(layer.properties['sids'][index])
                print(layer.properties['cids'][index])
            else:
                layer.selected_data = []


        if event.button == 1:
            # remove nearby points
            index = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                layer.selected_data = [index]
                print(layer.data[index])
                print(layer.properties['sids'][index])
                print(layer.properties['cids'][index])
            else:
                layer.selected_data = []
    

    def refresh(self, viewer):
        point = get_random_point(self.db_path)
        nbr_sids = stree.get_nbr_segs(point['coord'],dis=self.size)
        segs = segs_from_sids(db_path,nbr_sids)
        center = point['coord']
        roi = [i-self.size for i in center] + [self.size*2,self.size*2,self.size*2]
        img = self.image.from_roi(roi)

        [x1,y1,z1] = roi[:3]
        [x2,y2,z2] = [i+j for i,j in zip(roi[:3],roi[3:])]
        points = []
        sids = []
        cids = []
        colors = []
        interval = 3
        scolors = [i/len(segs) for i in list(range(len(segs)+1))]

        for i, seg in enumerate(segs):
            seg_color = scolors[i]
            seg_points = []
            seg_sids = []
            seg_cids = []
            seg_colors = []
            for i, point in enumerate(seg['points']):
                [x,y,z] = point
                if x<=x1 or x>=x2 or y<=y1 or y>=y2 or z<=z1 or z>=z2:
                    continue
                seg_points.append(point)
                seg_sids.append(seg['sid'])
                seg_cids.append(i)
                seg_colors.append(seg_color)
            if len(seg_points)==0:
                continue

            sampled_points = seg_points[:-(interval-2):interval]
            sampled_points.append(seg_points[-1])
            sampled_sids = seg_sids[:-(interval-2):interval]
            sampled_sids.append(seg_sids[-1])
            sampled_cids = seg_cids[:-(interval-2):interval]
            sampled_cids.append(seg_cids[-1])
            sampled_colors = seg_colors[:-(interval-2):interval]
            sampled_colors.append(seg_colors[-1])

            points += sampled_points
            sids += sampled_sids
            cids += sampled_cids
            colors += sampled_colors

        properties = {
            'colors': np.array(colors),
            'sids': np.array(sids),
            'cids': np.array(cids)
        }

        viewer.layers['Points'].data = np.array(points)
        viewer.layers['Points'].properties = properties
        viewer.layers['Points'].face_colormap = 'turbo'
        viewer.layers['Points'].face_color = 'colors'
        viewer.layers['Points'].selected_data = []

        viewer.layers['Image'].data = img
        viewer.layers['Image'].translate = roi[:3]
        viewer.layers['Image'].reset_contrast_limits()



        viewer.reset_view()



if __name__ == '__main__':
    # SIZE = 32
    db_path = 'tests/test.db'
    image_path = '/home/bean/workspace/data/test.zarr'
    model_path = 'src/weights/self_net_3d.pkl'
    # image = Image(image_path)
    segs = read_segs(db_path)
    stree = SegsTree(segs)
    # point = get_one_point(db_path)
    # nbr_sids = stree.get_nbr_segs(point['coord'],dis=SIZE)
    # roi_segs = segs_from_sids(db_path,nbr_sids)
    # center = point['coord']
    # roi = [i-SIZE for i in center] + [SIZE*2,SIZE*2,SIZE*2]
    # img = image.from_roi(roi)

    # show_segs_and_image(roi_segs,img,roi)
    from ntools.deconv import Deconver
    deconver = Deconver(model_path)
    anno = Annotator(db_path,image_path,deconver)
