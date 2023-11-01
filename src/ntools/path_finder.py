import numpy as np
from brightest_path_lib.algorithm import NBAStarSearch
import dijkstra3d as djk

class PathFinder:
    def __init__(self, image, viewer = None):
        if viewer is None:
            raise TypeError('You need to pass a napari viewer for the myviewer argument')
        else:
            self.viewer = viewer
            self.image_layer = self.viewer.add_image(image)
            self.start_layer = self.viewer.add_points(ndim=3,face_color='cyan',size=2,name='start point',edge_color='black',shading='spherical')
            self.goal_layer = self.viewer.add_points(ndim=3,face_color='red',size=2,name='goal point',edge_color='black',shading='spherical')
            self.path_layer = self.viewer.add_points(ndim=3,face_color='green',size=1,name='goal point',edge_color='black',shading='spherical')
        #establish key bindings
        self.add_callback()


    def add_callback(self):
        self.viewer.bind_key('r', self.find_path)
        self.viewer.bind_key('f', self.find_simple_path)
        self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)


    def find_path(self,viewer):
        sa = NBAStarSearch(self.image_layer.data, start_point=self.start_layer.data[0], goal_point=self.goal_layer.data[0])
        path = sa.search()
        self.path_layer.data = path



    def find_simple_path(self,viewer):
        field = self.image_layer.data 
        field = (field.max()-field)/(field.max()-field.min())
        path = djk.dijkstra(field, source=self.start_layer.data[0], target=self.goal_layer.data[0], bidirectional=True,connectivity=26).tolist()
        self.path_layer.data = path



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

