from ntools.dbio import read_edges, read_nodes
from magicgui import magicgui, widgets
from ntools.read_ims import Image as Image_ims
from ntools.read_zarr import Image as Image_zarr
from scipy.spatial import KDTree
import numpy as np
import networkx as nx
import napari
import pathlib
import random



class Annotator:
    def __init__(self):
        # --------- GUI ---------
        self.viewer = napari.Viewer(ndisplay=3)
        # panorama mode
        self.panorama_layer = self.viewer.add_points(None,ndim=3,size=None,shading='spherical',edge_width=0,properties=None,face_colormap='hsl',name='panorama view',blending='additive',visible=True)
        # labeling mode
        self.image_layer = self.viewer.add_image(np.ones((64, 64, 64), dtype=np.uint16),name='image',visible=False)
        self.point_layer = self.viewer.add_points(None,ndim=3,size=None,shading='spherical',edge_width=0,properties=None,face_colormap='hsl',name='points',visible=False)
        self.edge_layer = self.viewer.add_vectors(None,ndim=3,name='existing edges',visible=False)
        self.ex_edge_layer = self.viewer.add_vectors(None,ndim=3,name='added edges',visible=False,edge_color='orange')
        # ------------------------

        self.add_control() # control panel

        # --------- data structure ---------
        self.image = None
        self.G = None # networkx graph
        self.kdtree = None
        self.coord_ids = None # kdtree indices to node indices
        self.connected_nodes = []
        self.delected_nodes = {
            'nodes': [],
            'edges': [] 
        }
        self.added_nodes = []
        # ----------------------------------

        napari.run()


    def add_control(self):
        # self.viewer.bind_key('r', self.refresh,overwrite=True)
        self.viewer.bind_key('r', self.switch_layer,overwrite=True)

        self.panorama_layer.mouse_drag_callbacks.append(self.node_selection)
        self.point_layer.mouse_drag_callbacks.append(self.add_edge)
        self.image_layer.mouse_drag_callbacks.append(self.put_point)


        self.image_path = widgets.FileEdit(label="image path")
        self.db_path = widgets.FileEdit(label="database path")
        self.refresh_panorama_button = widgets.PushButton(text="refresh panorama")
        self.refresh_panorama_button.clicked.connect(self.refresh_panorama)


        self.len_thres = widgets.Slider(label="min length", value=10, min=0, max=200)
        self.min_size = widgets.Slider(label="min size", value=3, min=0, max=10)
        self.max_size = widgets.Slider(label="max size", value=10, min=1, max=20)
        self.mode_switch = widgets.PushButton(text="switch mode")
        self.mode_switch.clicked.connect(self.switch_mode)

        self.selected_node = widgets.LineEdit(label="node selection", value=0)

        self.image_size = widgets.Slider(label="block size", value=64, min=64, max=1024)
        self.image_size.changed.connect(self.clip_value)
        self.refresh_button = widgets.PushButton(text="refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.recover_button = widgets.PushButton(text="recover")
        self.recover_button.clicked.connect(self.recover)
        # next task is just ask for new task without submitting
        self.submit_button = widgets.PushButton(text="submit")
        self.submit_button.clicked.connect(self.submit_result)

        self.container = widgets.Container(widgets=[
            self.image_path,
            self.db_path,
            self.len_thres,
            self.min_size,
            self.max_size,
            self.refresh_panorama_button,
            self.mode_switch,
            self.selected_node,
            self.image_size,
            self.refresh_button,
            self.recover_button,
            self.submit_button
            ])
        self.viewer.window.add_dock_widget(self.container, area='right')
    

    def clip_value(self):
        self.image_size.value = (self.image_size.value//64)*64


    def switch_mode(self,viewer):
        mode = 'panorama' if self.panorama_layer.visible == True else 'labeling'
        if mode == 'panorama':
            self.panorama_layer.visible = False
            self.point_layer.visible = True
            self.image_layer.visible = True
            self.edge_layer.visible = True
            self.ex_edge_layer.visible = True
            self.viewer.camera.zoom = 5
            self.refresh()
        else:
            self.refresh_panorama()


    def switch_layer(self,viewer):
        if self.viewer.layers.selection.active == self.point_layer:
            self.viewer.layers.selection.active = self.image_layer
        elif self.viewer.layers.selection.active == self.image_layer:
            self.viewer.layers.selection.active = self.point_layer


    def refresh(self):
        # update canvas according to center and size
        # it only needs one node id to generate one task
        # 1. choose one unchecked node from CC as center node
        # 2. query neighbour nodes from kdtree
        # 3. assign properties for nodes to identify different segments and center point, add existing edges to vector layer
        # 4. load image

        connected_component = nx.node_connected_component(self.G, int(self.selected_node.value))
        unchecked_nodes = []
        for node in connected_component:
            if self.G.degree(node) == 1 and self.G.nodes[node]['checked'] == 0:
                unchecked_nodes.append(node)

        selection = unchecked_nodes[-1]
        self.selected_node.value = str(selection)

        c_coord = self.G.nodes[selection]['coord']
        nbrs = self.kdtree.query_ball_point(c_coord, self.image_size.value//2, p=float(np.inf))
        nbrs = [self.coord_ids[i] for i in nbrs]
        nbrs = nbrs + self.added_nodes
        sub_g = self.G.subgraph(nbrs)

        connected_components = list(nx.connected_components(sub_g))
        coords = []
        sizes = []
        colors = []
        nids = []
        edges = []

        for cc in connected_components:
            color = random.random()
            nodes = [self.G.nodes[i] for i in cc if self.G.has_node(i)]
            for node in nodes:
                coords.append(node['coord'])
                nids.append(node['nid'])
                colors.append(color)
                if node['nid']!=selection:
                    sizes.append(1)
                else:
                    sizes.append(2)


        for c_node in nbrs:
            if not self.G.has_node(c_node):
                continue
            p1 = self.G.nodes[c_node]['coord']
            for pid in list(self.G.neighbors(c_node)):
                p2 = self.G.nodes[pid]['coord']
                v = [j-i for i,j in zip(p1,p2)]
                edges.append([p1,v])


        colors = np.array(colors)
        colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
        colors = np.nan_to_num(colors, nan=0.5)

        properties = {
            'colors': colors,
            'nids': np.array(nids)
        }


        image = self.image.from_roi([i-self.image_size.value//2 for i in c_coord]+[self.image_size.value,self.image_size.value,self.image_size.value])
        min_i = np.min(image)
        max_i = np.max(image)
        max_i -= (max_i-min_i)/2

        self.point_layer.data = np.array(coords)
        self.point_layer.properties = properties
        self.point_layer.face_colormap = 'hsl'
        self.point_layer.face_color = 'colors'
        self.point_layer.size = sizes
        self.point_layer.selected_data = []
        self.ex_edge_layer.data = np.array(edges)
        
        self.image_layer.data = image
        self.image_layer.reset_contrast_limits()
        self.image_layer.contrast_limits = [min_i,max_i]
        self.image_layer.translate = [i-self.image_size.value//2 for i in c_coord]
        self.viewer.camera.center = c_coord
        self.viewer.layers.selection.active = self.point_layer


    def recover(self):
        # recover the preserved delected nodes if exists
        for node in self.delected_nodes['nodes']:
            self.G.add_node(node['nid'], nid = node['nid'],coord = node['coord'], nbrs = node['nbrs'], checked = node['checked'])
        for edge in self.delected_nodes['edges']:
            self.G.add_edge(edge[0],edge[1])

        self.delected_nodes = {
            'nodes': [],
            'edges': []
        }

        if len(self.added_nodes)!=0:
            self.G.remove_nodes_from(self.added_nodes)
        self.added_nodes = []
        
        self.refresh()


    def submit_result(self):
        # label the center node of current task as checked in self.G
        # add new edge to the self.G and database
        # if there's new nodes, update kdtree
        # run refresh to updata canvas

        self.delected_nodes = {
            'nodes': [],
            'edges': []
        }

        coords = []
        coord_ids = []
        if len(self.added_nodes)!=0:
            for i,node in enumerate(list(self.G)):
                coords.append(self.G.nodes[node]['coord'])
                coord_ids.append(self.G.nodes[node]['nid'])
            self.kdtree = KDTree(np.array(coords))
            self.coord_ids = coord_ids

        self.added_nodes = []


        for node in self.connected_nodes:
            self.G.add_edge(int(self.selected_node.value),node)
        self.G.nodes[int(self.selected_node.value)]['checked']+=1

        self.connected_nodes = []
        self.edge_layer.data = None
        self.refresh()


    def refresh_panorama(self):
        if self.G is None:
            # load graph and kdtree from database
            nodes = read_nodes(self.db_path.value)
            edges = read_edges(self.db_path.value)
            self.G = nx.Graph()
            for node in nodes:
                self.G.add_node(node['nid'], nid = node['nid'],coord = node['coord'], nbrs = node['nbrs'], checked = node['checked'])

            for edge in edges:
                self.G.add_edge(edge['src'],edge['des'],creator = edge['creator'])

            coords = []
            coord_ids = []
            for i,node in enumerate(nodes):
                coords.append(node['coord'])
                coord_ids.append(node['nid'])
            self.kdtree = KDTree(np.array(coords))
            self.coord_ids = coord_ids
            # read image
            if 'ims' in str(self.image_path.value):
                self.image = Image_ims(self.image_path.value)
            elif 'zarr' in str(self.image_path.value):
                self.image = Image_zarr(self.image_path.value)
            else:
               raise Exception("image type not supported yet") 


        connected_components = list(nx.connected_components(self.G))

        coords = []
        sizes = []
        colors = []
        nids = []


        for cc in connected_components:
            if len(cc)<int(self.len_thres.value):
                continue
            color = random.random()
            nodes = [self.G.nodes[i] for i in cc]
            for node in nodes:
                coords.append(node['coord'])
                nids.append(node['nid'])
                colors.append(color)
                sizes.append(len(nodes))


        colors = np.array(colors)
        colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
        colors = np.nan_to_num(colors, nan=0.5)
        sizes = np.array(sizes)
        sizes = (sizes-np.min(sizes))/(np.max(sizes)-np.min(sizes))*10
        sizes = np.nan_to_num(sizes, nan=10)
        sizes = np.clip(sizes,int(self.min_size.value),int(self.max_size.value))


        properties = {
            'colors': colors,
            'nids': np.array(nids)
        }

        self.panorama_layer.data = np.array(coords)
        self.panorama_layer.properties = properties
        self.panorama_layer.face_colormap = 'hsl'
        self.panorama_layer.face_color = 'colors'
        self.panorama_layer.size = sizes
        self.panorama_layer.selected_data = []

        self.panorama_layer.visible = True
        self.point_layer.visible = False 
        self.image_layer.visible = False
        self.edge_layer.visible = False
        self.ex_edge_layer.visible = False
        self.viewer.reset_view()
        self.viewer.layers.selection.active = self.panorama_layer


    def node_selection(self, layer, event):
        if event.button == 2:
            # remove all connected points
            index = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                self.selected_node.value = str(layer.properties['nids'][index])
            else:
                self.selected_node.value = None
    

    def add_edge(self, layer, event):
        if event.button == 1:
            # add connection to center node
            index = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                # add vector between selected node and center node
                node_id = self.point_layer.properties['nids'][index]
                if node_id not in self.connected_nodes:
                    self.connected_nodes.append(node_id)
                elif node_id in self.connected_nodes:
                    self.connected_nodes.remove(node_id)

                # refresh edge layer
                vectors = []
                p1 = self.G.nodes[int(self.selected_node.value)]['coord']
                for pid in self.connected_nodes:
                    p2 = self.G.nodes[pid]['coord']
                    v = [j-i for i,j in zip(p1,p2)]
                    vectors.append([p1,v])
                self.edge_layer.data = np.array(vectors)


        if event.button == 2:
            # remove connection to center node
            index = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                # remove node and its edges
                node_id = self.point_layer.properties['nids'][index]
                if node_id != int(self.selected_node.value):
                    # preserve the delected node, until next submit
                    self.delected_nodes['nodes'].append(self.G.nodes[node_id])
                    for nbr in self.G.neighbors(node_id):
                        self.delected_nodes['edges'].append([node_id,nbr])
                    self.G.remove_node(int(node_id))
                    self.refresh()


    def put_point(self,layer,event):
        # based on ray casting
        if(event.button==2):
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
                sample_point = np.clip(sample_point, bbox[:, 0], bbox[:, 1])
                value = layer.data[sample_point[0], sample_point[1], sample_point[2]]
                sample_points.append(sample_point)
                values.append(value)
            max_point_index = values.index(max(values))
            max_point = sample_points[max_point_index]
            max_point = [i+int(j) for i,j in zip(max_point,self.image_layer.translate)]

            # get new node id
            new_id = len(self.G)
            while self.G.has_node(new_id):
                new_id+=1
            
            self.G.add_node(new_id, nid = new_id, coord = max_point, nbrs = [], checked = 0)
            self.added_nodes.append(new_id)

            self.refresh()
        

if __name__ == '__main__':
    anno = Annotator()
