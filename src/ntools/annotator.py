import numpy as np
import os
from brightest_path_lib.algorithm import NBAStarSearch
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from skimage.morphology import skeletonize
from scipy.ndimage import zoom
from tkinter import filedialog
from tifffile import imwrite
from skimage import morphology, measure
from skimage.morphology import ball


class VolumeAnnotator:
    def __init__(self, image, skels = None, viewer = None, save_dir = None):
        if viewer is None:
            raise TypeError('You need to pass a napari viewer for the myviewer argument')
        else:
            self.viewer = viewer
            self.image_layer = self.viewer.add_image(image)
            self.start_layer = self.viewer.add_points(ndim=3,face_color='cyan',size=2,name='start point',edge_color='black',shading='spherical')
            self.goal_layer = self.viewer.add_points(ndim=3,face_color='red',size=2,name='goal point',edge_color='black',shading='spherical')
            self.path_layer = self.viewer.add_points(ndim=3,face_color='green',size=1,name='path point',edge_color='black',shading='spherical')
            self.save_dir = save_dir
        #establish key bindings
        if skels is not None:
            self.path_layer.data = skels
        self.add_callback()


    def add_callback(self):
        self.viewer.bind_key('f', self.find_path)
        self.viewer.bind_key('r', self.step_forward)
        self.viewer.bind_key('s', self.save_result)
        self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)
        self.path_layer.mouse_drag_callbacks.append(self.get_point_under_cursor)

    def find_path(self,viewer):
        sa = NBAStarSearch(self.image_layer.data, start_point=self.start_layer.data[0], goal_point=self.goal_layer.data[0])
        path = sa.search()
        if len(self.path_layer.data)!=0:
            total_path = np.concatenate((self.path_layer.data, np.array(path)), axis=0)
        else:
            total_path = path

        self.path_layer.data = total_path


    def step_forward(self,viewer):
        self.start_layer.data = self.goal_layer.data
        self.goal_layer.data = []


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
                points = layer.data
                filtered = self.remove_connected_points(points,1.8,index)
                layer.data = filtered
        if event.button == 1:
            # remove nearby points
            index = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                points = layer.data
                filtered = self.remove_nearby_points(points,10,index)
                layer.data = filtered


    def on_double_click(self,layer,event):
        #based on ray casting
        near_point, far_point = layer.get_ray_intersections(
            event.position,
            event.view_direction,
            event.dims_displayed
        )
        sample_ray = far_point - near_point
        length_sample_vector = np.linalg.norm(sample_ray)
        increment_vector = sample_ray / (2 * length_sample_vector)
        n_iterations = int(2 * length_sample_vector)
        bbox = np.array([
            [0, layer.data.shape[0]-1],
            [0, layer.data.shape[1]-1],
            [0, layer.data.shape[2]-1]
        ])
        sample_points = []
        values = []
        for i in range(n_iterations):
            sample_point = np.asarray(near_point + i * increment_vector, dtype=int)
            sample_point = self.clamp_point_to_bbox(sample_point, bbox)
            value = layer.data[sample_point[0], sample_point[1], sample_point[2]]
            sample_points.append(sample_point)
            values.append(value)
        max_point_index = values.index(max(values))
        max_point = sample_points[max_point_index]
        print('Put point at: ', max_point)
        if(event.button==2):
            self.start_layer.data = max_point
        if(event.button==1):
            self.goal_layer.data = max_point


    def clamp_point_to_bbox(self,point: np.ndarray, bbox: np.ndarray):
        clamped_point = np.clip(point, bbox[:, 0], bbox[:, 1])
        return clamped_point
    

    def remove_connected_points(self, points, threshold, point_index):
        distances = cdist(points, points)
        adjacency_matrix = distances <= threshold
        num_components, labels = connected_components(adjacency_matrix, directed=False)
        label_to_remove = labels[point_index]
        filtered_points = points[labels != label_to_remove]
        return filtered_points

    
    def remove_nearby_points(self, points, dis, point_index):
        c_point = points[point_index]
        distances = np.linalg.norm(points-c_point, axis=1)
        # indices = np.argsort(distances)
        indices_to_keep = distances > dis
        filtered_points = points[indices_to_keep]
        return filtered_points
    

    def save_result(self,viewer):
        # image_name = filedialog.asksaveasfilename()
        all_files = os.listdir(self.save_dir)

        tif_files = [file for file in all_files if file.endswith('.tif')]
        next_image_number = len(tif_files)//2 + 1
        image_name = f'img_{next_image_number}.tif'
        image_name = os.path.join(self.save_dir, image_name)

        image = self.image_layer.data
        imwrite(image_name,image)

        coordinates = self.path_layer.data
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1

        # mask = zoom(mask,(2,6,2),mode='nearest').astype(np.uint8)
        # mask = morphology.binary_dilation(mask, footprint=ball(1))
        # mask = skeletonize(mask)
        # mask = morphology.binary_dilation(mask, footprint=ball(1))

        directory, filename = os.path.split(image_name)

        mask_name = filename.replace('img', 'mask')
        mask_path = os.path.join(directory, mask_name)

        imwrite(mask_path,mask)

        print(image_name+' saved')
        print(mask_path+' saved')

