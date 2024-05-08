from ntools.dbio import read_edges, read_nodes, delete_nodes, add_nodes, add_edges, check_node, uncheck_nodes, augment_nodes, change_type
from magicgui import magicgui, widgets
from ntools.image_reader import wrap_image
from scipy.spatial import KDTree
from napari.utils.notifications import show_info
import numpy as np
import networkx as nx
import napari
import random


# use PushButton itself as a recorder
class PushButton(widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Annotator:
    def __init__(self):
        # --------- GUI ---------
        self.viewer = napari.Viewer(ndisplay=3)
        # panorama mode
        self.panorama_image = self.viewer.add_image(np.ones((64, 64, 64), dtype=np.uint16), name='panorama image',visible=False)
        self.panorama_points = self.viewer.add_points(None,ndim=3,size=None,shading='spherical',edge_width=0,properties=None,face_colormap='hsl',name='panorama view',blending='additive',visible=True)
        # labeling mode
        self.image_layer = self.viewer.add_image(np.ones((64, 64, 64), dtype=np.uint16),name='image',visible=False)
        self.point_layer = self.viewer.add_points(None,ndim=3,size=None,shading='spherical',edge_width=0,properties=None,face_colormap='hsl',name='points',visible=False)
        self.edge_layer = self.viewer.add_vectors(None,ndim=3,name='added edges',visible=False)
        self.ex_edge_layer = self.viewer.add_vectors(None,ndim=3,name='existing edges',visible=False,edge_color='orange')
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
        # ----- napari bindings -----
        self.viewer.bind_key('q', self.switch_mode,overwrite=True)
        self.viewer.bind_key('r', self.recover,overwrite=True)
        self.viewer.bind_key('g', self.switch_layer,overwrite=True)
        self.viewer.bind_key('d', self.refresh,overwrite=True)
        self.viewer.bind_key('f', self.submit_result,overwrite=True)
        self.viewer.bind_key('w', self.connect_one_nearest,overwrite=True)
        self.viewer.bind_key('e', self.connect_two_nearest,overwrite=True)
        self.viewer.bind_key('b', self.last_task,overwrite=True)
        self.viewer.bind_key('s', self.label_soma,overwrite=True)
        self.viewer.bind_key('a', self.label_ambiguous,overwrite=True)

        self.panorama_points.mouse_drag_callbacks.append(self.node_selection)
        self.point_layer.mouse_drag_callbacks.append(self.node_operations)
        self.image_layer.mouse_drag_callbacks.append(self.put_point)
        # ---------------------------

        # ----- widgets -----
        self.user_name = widgets.LineEdit(label="user name", value='tester')
        self.image_path = widgets.FileEdit(label="image path")
        self.db_path = widgets.FileEdit(label="database path")
        self.image_switch = widgets.CheckBox(value=False,text='show panorama image')
        self.segs_switch = widgets.CheckBox(value=True,text='show/hide long segments')
        self.refresh_panorama_button = widgets.PushButton(text="refresh panorama")

        self.min_length = widgets.Slider(label="min_length", value=10, min=0, max=200)
        self.len_thres = widgets.Slider(label="length thres", value=20, min=0, max=9999)
        self.min_size = widgets.Slider(label="min size", value=3, min=0, max=10)
        self.max_size = widgets.Slider(label="max size", value=10, min=1, max=20)
        self.mode_switch = PushButton(text="switch mode (q)")
        self.mode_switch.mode = 'panorama'

        self.selected_node = widgets.LineEdit(label="node selection", value=0)
        self.total_length = widgets.LineEdit(label="total length", value=0)
        self.nodes_left = widgets.LineEdit(label="nodes left", value=0)
        self.image_size = widgets.Slider(label="block size", value=64, min=64, max=1024)
        self.refresh_button = widgets.PushButton(text="refresh (d)")
        self.return_button = widgets.PushButton(text="shit I misclicked (b)")
        self.recover_button = widgets.PushButton(text="recover (r)")
        # next task is just ask for new task without submitting
        self.submit_button = PushButton(text="submit (f)")
        self.submit_button.history = []
        self.proofreading_switch = widgets.CheckBox(value=False,text='Proofreading')
        self.soma_buttom = widgets.PushButton(text="label/unlabel soma (s)")
        self.ambiguous_button = widgets.PushButton(text="label/unlabel ambiguous (a)")
        # ---------------------------

        # ----- widgets bindings -----
        self.submit_button.clicked.connect(self.submit_result)
        self.refresh_panorama_button.clicked.connect(self.refresh_panorama)
        self.mode_switch.clicked.connect(self.switch_mode)
        self.image_size.changed.connect(self.clip_value)
        self.refresh_button.clicked.connect(self.refresh)
        self.recover_button.clicked.connect(self.recover)
        self.return_button.clicked.connect(self.last_task)
        self.soma_buttom.clicked.connect(self.label_soma)
        self.ambiguous_button.clicked.connect(self.label_ambiguous)
        # ---------------------------

        self.container = widgets.Container(widgets=[
            self.user_name,
            self.image_path,
            self.db_path,
            self.image_switch,
            self.segs_switch,
            self.min_length,
            self.len_thres,
            self.min_size,
            self.max_size,
            self.refresh_panorama_button,
            self.mode_switch,
            self.selected_node,
            self.total_length,
            self.nodes_left,
            self.image_size,
            self.proofreading_switch,
            self.refresh_button,
            self.return_button,
            self.recover_button,
            self.soma_buttom,
            self.ambiguous_button,
            self.submit_button
            ])

        self.viewer.window.add_dock_widget(self.container, area='right')



    def label_soma(self,viewer):
        node_id = int(self.selected_node.value)
        if self.G.nodes[node_id]['type'] == 0:
            change_type(str(self.db_path.value),node_id,1)
            self.G.nodes[node_id]['type'] = 1
        elif self.G.nodes[node_id]['type'] == 1:
            change_type(str(self.db_path.value),node_id,0)
            self.G.nodes[node_id]['type'] = 0
        self.refresh(self.viewer)
        

    def label_ambiguous(self,viewer):
        node_id = int(self.selected_node.value)
        if self.G.nodes[node_id]['type'] == 0:
            change_type(str(self.db_path.value),node_id,8)
            self.G.nodes[node_id]['type'] = 8
            show_info("{node_id} labeled as ambiguous")
        elif self.G.nodes[node_id]['type'] == 8:
            change_type(str(self.db_path.value),node_id,0)
            self.G.nodes[node_id]['type'] = 0
            show_info("{node_id} labeled as normal")

    def connect_one_nearest(self,viewer):
        # find one closest neighbour point, add it to self.connected_nodes
        if len(self.delected_nodes['nodes'])>0 or len(self.added_nodes)>0:
            show_info('Do not use this if you have added or deleted nodes')
            return
        selection = int(self.selected_node.value)
        c_coord = self.G.nodes[selection]['coord']
        d, nbrs = self.kdtree.query(c_coord, self.image_size.value//2, p=float(np.inf))
        nbrs = [self.coord_ids[i] for i in nbrs]
        cc = list(nx.node_connected_component(self.G,selection))
        nbrs = [i for i in nbrs if i not in cc]
        if len(nbrs)>0:
            self.connected_nodes = [nbrs[0]]
        self.refresh_edge_layer()
        self.refresh(self.viewer)


    def connect_two_nearest(self,viewer):
        # find one closest neighbour point, add it to self.connected_nodes
        if len(self.delected_nodes['nodes'])>0 or len(self.added_nodes)>0:
            show_info('Do not use this if you have added or deleted nodes')
            return
        selection = int(self.selected_node.value)
        c_coord = self.G.nodes[selection]['coord']
        d, nbrs = self.kdtree.query(c_coord, self.image_size.value//2, p=float(np.inf))
        nbrs = [self.coord_ids[i] for i in nbrs]
        cc = nx.node_connected_component(self.G,selection)
        nbrs = [i for i in nbrs if i not in cc]
        if len(nbrs)>1:
            self.connected_nodes = nbrs[:2]
        self.refresh_edge_layer()
        self.refresh(self.viewer)


    def clip_value(self):
        # image size should be integer multiple of 64
        self.image_size.value = (self.image_size.value//64)*64


    def switch_mode(self,viewer):
        if self.mode_switch.mode == 'panorama':
            self.mode_switch.mode = 'labeling'
            self.panorama_points.visible = False
            self.panorama_image.visible = False
            self.point_layer.visible = True
            self.image_layer.visible = True
            self.edge_layer.visible = True
            self.ex_edge_layer.visible = True
            self.viewer.camera.zoom = 5
            self.refresh(self.viewer)
        else:
            # remove current recordings
            self.submit_result(self.viewer)
            self.mode_switch.mode = 'panorama'
            self.connected_nodes = []
            self.delected_nodes = {
                'nodes': [],
                'edges': []
            }
            self.added_nodes = []
            self.refresh_panorama()


    def switch_layer(self,viewer):
        if self.viewer.layers.selection.active == self.point_layer:
            self.viewer.layers.selection.active = self.image_layer
        elif self.viewer.layers.selection.active == self.image_layer:
            self.viewer.layers.selection.active = self.point_layer
    

    def last_task(self,viewer):
        if len(self.submit_button.history)>0:
            last_node = self.submit_button.history[-1]
            self.G.nodes[last_node]['checked'] = -1
            self.selected_node.value = str(last_node)
            self.connected_nodes = []
            self.submit_button.history.remove(last_node)
            self.refresh_edge_layer()
            self.refresh(self.viewer)
        else:
            show_info("No history recorded")


    def refresh(self, viewer):
        # update canvas according to center and size
        # it only needs one node id to generate one task
        # 1. choose one unchecked node from CC as center node
        # 2. query neighbour nodes from kdtree
        # 3. assign properties for nodes to identify different segments and center point, add existing edges to vector layer
        # 4. load image

        connected_component = nx.node_connected_component(self.G, int(self.selected_node.value))
        unchecked_nodes = []
        for node in connected_component:
            if (self.G.degree(node) == 1 and self.G.nodes[node]['checked'] == 0) or self.G.nodes[node]['checked'] == -1:
                unchecked_nodes.append(node)
        # sort unchecked nodes according to distance between selected node
        dis = []
        for nid in unchecked_nodes:
            dis.append(nx.shortest_path_length(self.G, source=int(self.selected_node.value), target=nid))
        unchecked_nodes = [x for _,x in sorted(zip(dis,unchecked_nodes))]
        
        self.update_meter(len(connected_component),len(unchecked_nodes))

        if len(unchecked_nodes)==0 and self.proofreading_switch.value == False:
            show_info('all nodes checked, run proofreading')
            unchecked_nodes.append(int(self.selected_node.value))

        if self.proofreading_switch.value == False:
            selection = unchecked_nodes[0]
            self.selected_node.value = str(selection)
            c_coord = self.G.nodes[selection]['coord']
            nbrs = self.kdtree.query_ball_point(c_coord, self.image_size.value//2, p=float(np.inf))
            nbrs = [self.coord_ids[i] for i in nbrs]
            nbrs = nbrs + self.added_nodes
            sub_g = self.G.subgraph(nbrs)
            connected_components = list(nx.connected_components(sub_g))
        else:
            selection = int(self.selected_node.value)
            c_coord = self.G.nodes[selection]['coord']
            self.selected_node.value = str(selection)
            connected_components = [list(nx.node_connected_component(self.G, selection))]
            nbrs = connected_components[0]


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
                if node['checked']==-1 and self.proofreading_switch.value==True:
                    colors.append(0)
                else:
                    colors.append(color)
                if node['nid']==selection:
                    sizes.append(2)
                else:
                    sizes.append(1)
                if node['type']==1:
                    sizes.pop(-1)
                    sizes.append(4)

        for c_node in nbrs:
            if not self.G.has_node(c_node):
                continue
            p1 = self.G.nodes[c_node]['coord']
            for pid in list(self.G.neighbors(c_node)):
                p2 = self.G.nodes[pid]['coord']
                v = [j-i for i,j in zip(p1,p2)]
                edges.append([p1,v])


        colors = np.array(colors)
        if np.max(colors) != np.min(colors):
            colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))

        properties = {
            'colors': colors,
            'nids': np.array(nids)
        }


        image = self.image.from_roi([i-self.image_size.value//2 for i in c_coord]+[self.image_size.value,self.image_size.value,self.image_size.value])

        translate = [i-self.image_size.value//2 for i in c_coord]

        local_coords = np.array(coords) - np.array(translate)

        mask = np.all((local_coords>= np.array([0,0,0])) & (local_coords < np.array([self.image_size.value,self.image_size.value,self.image_size.value])), axis=1)
        local_coords = local_coords[mask]

        intensities = image[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
        mean_value = np.mean(intensities)
        std_value = np.std(intensities)
        min_value = np.min(intensities)
        max_value = np.max(intensities)


        self.point_layer.data = np.array(coords)
        self.point_layer.properties = properties
        self.point_layer.face_colormap = 'hsl'
        self.point_layer.face_color = 'colors'
        self.point_layer.size = sizes
        self.point_layer.selected_data = []
        self.ex_edge_layer.data = np.array(edges)
        
        self.image_layer.data = image
        self.image_layer.reset_contrast_limits()
        self.image_layer.contrast_limits = [mean_value//2,mean_value+std_value]
        self.image_layer.translate = translate
        self.viewer.camera.center = c_coord
        self.viewer.layers.selection.active = self.point_layer


    def recover(self, viewer):
        # recover the preserved delected nodes if exists
        for node in self.delected_nodes['nodes']:
            self.G.add_node(node['nid'], nid = node['nid'],coord = node['coord'], type = node['type'], checked = 0)
        for edge in self.delected_nodes['edges']:
            self.G.add_edge(edge[0],edge[1])

        self.delected_nodes = {
            'nodes': [],
            'edges': []
        }

        if len(self.added_nodes)!=0:
            self.G.remove_nodes_from(self.added_nodes)
        self.added_nodes = []
        self.connected_nodes = []
        self.refresh_edge_layer()
        self.refresh(self.viewer)


    def submit_result(self,viewer):
        self.submit_button.history.append(int(self.selected_node.value))
        self.update_database()
        self.update_local()


    def update_local(self):
        # label the center node of current task as checked in self.G
        # update canvas and local graph
        # if there's new nodes, update kdtree
        # run refresh to updata canvas
        self.delected_nodes = {
            'nodes': [],
            'edges': []
        }

        coords = []
        coord_ids = []
        if len(self.added_nodes)!=0:
            # added_nodes already in self.G, not in self.kdtree
            # TODO fix bug that node has no 'coord'
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
        self.refresh(self.viewer)


    def update_database(self):
        # update database according to 'delected_nodes', 'added_nodes', 'connected_nodes'
        # deleted_nodes['nodes']: [{'nid','coord','type','checked'}]
        # deleted_nodes['edges']: [[src,tar]]
        # added_nodes: [{'nid','coord','type','checked'}]
        # connected_nodes: [nid,nid,...]
        # 1. remove nodes and edges in delected_nodes
        # 2. add new nodes to database
        # 3. add new edges to database
        path = str(self.db_path.value)
        if self.proofreading_switch.value == True:
            # uncheck nodes from connected components
            nids = []
            for nid in nx.node_connected_component(self.G,int(self.selected_node.value)):
                node = self.G.nodes[nid]
                if node['checked'] == -1:
                    nids.append(node['nid'])
            uncheck_nodes(path,nids)
        else:
            deleted_nodes = []
            for node in self.delected_nodes['nodes']:
                deleted_nodes.append(node['nid'])

            if len(deleted_nodes)>0:
                delete_nodes(path,deleted_nodes)

            added_nodes = []
            for nid in self.added_nodes:
                added_nodes.append(self.G.nodes[nid])

            if len(added_nodes)>0:
                add_nodes(path,added_nodes)

            added_edges = []
            for node in self.connected_nodes:
                added_edges.append([int(self.selected_node.value),int(node)])

            if len(added_edges)>0:
                add_edges(path,added_edges,self.user_name.value)

            check_node(path,int(self.selected_node.value))


    def update_meter(self,total_len,n_nodes):
        self.total_length.value = int(total_len)
        self.nodes_left.value = int(n_nodes)


    def refresh_panorama(self):
        if self.mode_switch.mode == 'labeling':
            show_info('switch to panorama mode first')
            return
        if self.G is None:
            # check if nodes table has type column
            augment_nodes(self.db_path.value)
            # load graph and kdtree from database
            nodes = read_nodes(self.db_path.value)
            edges = read_edges(self.db_path.value)
            self.G = nx.Graph()
            for node in nodes:
                self.G.add_node(node['nid'], nid = node['nid'],coord = node['coord'], type = node['type'], checked = node['checked'])

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
            self.image = wrap_image(str(self.image_path.value))
        
        # load low reslotion image if exists ('ims' file format)
        if 'ims' in str(self.image_path.value) and (self.panorama_image.scale == np.array([1,1,1])).all() and self.image_switch.value == True:
            # iterate levels, find one that has proper size
            level = 0
            for i, roi in enumerate(self.image.rois):
                if (np.array(roi[3:])<np.array([1000,1000,1000])).all():
                    level = i
                    break
            # update panorama image layer
            roi = self.image.rois[i]
            spacing = self.image.info[i]['spacing']
            image = self.image.from_roi(roi,level=level)
            self.panorama_image.data = image
            self.panorama_image.scale = spacing
            self.panorama_image.visible = True
            self.panorama_image.reset_contrast_limits()


        connected_components = list(nx.connected_components(self.G))

        coords = []
        sizes = []
        colors = []
        nids = []


        for cc in connected_components:
            if (len(cc)<int(self.len_thres.value) and self.segs_switch.value == True) or len(cc) <= self.min_length.value:
                continue
            if (len(cc)>=int(self.len_thres.value) and self.segs_switch.value == False) or len(cc) <= self.min_length.value:
                continue
            color = random.random()
            # check empty nodes
            nodes = [self.G.nodes[i] for i in cc]
            for nid,node in zip(list(cc),nodes):
                if node == {}:
                    delete_nodes(str(self.db_path.value),[nid])
                    self.G.remove_node(nid)
                    continue
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
        
        camera_center  = [i + j//2 for i,j in zip(self.image.roi[0:3],self.image.roi[3:])]


        self.panorama_points.visible = True
        self.panorama_points.data = np.array(coords)
        self.panorama_points.properties = properties
        self.panorama_points.face_colormap = 'hsl'
        self.panorama_points.face_color = 'colors'
        self.panorama_points.size = sizes
        self.panorama_points.selected_data = []


        if self.image_switch.value == True:
            self.panorama_image.visible = True
        else:
            self.panorama_image.visible = False

        self.point_layer.visible = False 
        self.image_layer.visible = False
        self.edge_layer.visible = False
        self.ex_edge_layer.visible = False

        self.viewer.reset_view()
        self.viewer.camera.center = camera_center
        self.viewer.layers.selection.active = self.panorama_points


    def node_selection(self, layer, event):
        # this is appended to panorama_points layer
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


    def refresh_edge_layer(self):
        vectors = []
        p1 = self.G.nodes[int(self.selected_node.value)]['coord']
        for pid in self.connected_nodes:
            p2 = self.G.nodes[pid]['coord']
            v = [j-i for i,j in zip(p1,p2)]
            vectors.append([p1,v])
        self.edge_layer.data = np.array(vectors)


    def node_operations(self, layer, event):
        '''
        this is appended to point_layer layer
        node operations:
            In proofreading:
                mouse 1: switch center node
                mouse 2: label node as unchecked
            In labeling mode:
                mouse 1: add/remove edge
                mouse 2: remove node and its edges
                shift + mouse1: switch center node
        
        One operation contains (click type, mode, modifier)
        '''
        index = layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if index is None:
            return
        else:
            node_id = self.point_layer.properties['nids'][index]
            # some ugly code
            if 'Shift' in event.modifiers:
                modifier = 'Shift'
            else:
                modifier = None

            mode = 'proofreading' if self.proofreading_switch.value == True else 'labeling'
        
        operation = (event.button, mode, modifier)
        
        match operation:
            case (1, 'proofreading', None): # switch center node
                self.selected_node.value = str(node_id)
                self.refresh(self.viewer)

            case (2, 'proofreading', None): # label node as unchecked
                if self.G.nodes[node_id]['checked'] == -1:
                    self.G.nodes[node_id]['checked'] = 0
                else:
                    self.G.nodes[node_id]['checked'] = -1
                    self.refresh(self.viewer)

            case (1, 'labeling', None): # add/remove edge
                if node_id not in self.connected_nodes:
                    self.connected_nodes.append(node_id)
                elif node_id in self.connected_nodes:
                    self.connected_nodes.remove(node_id)
                # refresh edge layer
                self.refresh_edge_layer()

            case (2, 'labeling', None): # remove node and its edges
                current_cc = nx.node_connected_component(self.G, int(self.selected_node.value))
                if len(current_cc)==1:
                    show_info("Can't remove the last node, switch center node before doing so.")
                    return
                if node_id not in current_cc:
                    # preserve the delected node, until next submit
                    if node_id in self.added_nodes:
                        self.added_nodes.remove(node_id)
                    else:
                        self.delected_nodes['nodes'].append(self.G.nodes[node_id])
                    for nbr in self.G.neighbors(node_id):
                        self.delected_nodes['edges'].append([node_id,nbr])
                        # after removing, label its neighbors as unchecked
                        self.G.nodes[nbr]['checked'] = 0

                    self.G.remove_node(node_id)

                    if node_id in self.connected_nodes:
                        self.connected_nodes.remove(node_id)
                    
                    self.refresh_edge_layer()
                    self.refresh(self.viewer)

                else:
                    # cut current_cc, select the largest subgraph
                    self.delected_nodes['nodes'].append(self.G.nodes[node_id])
                    # center node is not removed, keep it unchecked
                    self.G.nodes[node_id]['checked']-=1
                    nbrs = list(self.G.neighbors(node_id))
                    for nbr in nbrs:
                        self.delected_nodes['edges'].append([node_id,nbr])
                        self.G.nodes[nbr]['checked'] = 0
                    self.G.remove_node(node_id)
                    if node_id in self.connected_nodes:
                        self.connected_nodes.remove(node_id)

                    l_size = 0
                    for nbr in nbrs:
                        length = len(nx.node_connected_component(self.G,nbr))
                        if length>l_size:
                            self.selected_node.value = str(nbr)
                            l_size = length

                    self.refresh_edge_layer()
                    self.refresh(self.viewer)

            case (1, 'labeling', 'Shift'): # switch center node
                self.connected_nodes = []
                self.selected_node.value = str(node_id)
                if self.G.nodes[node_id]['checked'] >= 0:
                    self.G.nodes[node_id]['checked'] = -1
                self.refresh_edge_layer()
                self.refresh(self.viewer)

            case _ :
                show_info("operation not supported")


    def put_point(self,layer,event):
        # add new node to self.G and self.add_nodes
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
            
            self.G.add_node(new_id, nid = new_id, coord = max_point, type = 0, checked = 0)
            self.added_nodes.append(new_id)

            self.refresh(self.viewer)
            self.viewer.layers.selection.active = self.image_layer



    def save_lyp(self):
        '''
        load lyp template
        get all large connected components, save each as a single .lyp file
        save undirected graph, i.e. add all neighbours to 'parent_ids' except soma node
        for soma node, add "#Soma" to 'message'
        node template:
        {
            "creator_id": 0,
            "group_id": "1",
            "id": "5030239",
            "message": "", # if soma node, "#Soma" else ""
            "parent_ids": "5030238",
            "position": "49722.937 32652.702 44400.602",
            "timestamp": "1693814131" #unix timestamp
        }
        '''


def main():
    anno = Annotator()


if __name__ == '__main__':
    main()
