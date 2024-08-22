import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from ntools.dbio import read_edges, read_nodes
from ntools.image_reader import wrap_image
from rtree import index
from tqdm import tqdm
from brightest_path_lib.algorithm import AStarSearch

class Neurites():
    '''
    Neurites class represents neurites with nodes and associated links between them. It integrates KDTree for efficient spatial querying and NetworkX graph for exploring and retrieving based on graph structure. This allows both proximity-based searches and graph-based operations.
    '''
    def __init__(self,db_path,image_path=None,require_rtree=True):
        if image_path != None:
            self.image = wrap_image(image_path)
        else:
            self.image = None

        nodes = read_nodes(db_path)
        edges = read_edges(db_path)
        
        self.G = nx.Graph() # networkx graph
        for node in nodes:
            self.G.add_node(node['nid'], nid = node['nid'], coord = node['coord'], type = node['type'], checked = node['checked'])

        for edge in edges:
            self.G.add_edge(edge['src'],edge['des'],creator = edge['creator'])


        p = index.Property(dimension=3)
        rtree_idx = index.Index(properties=p)

        coords = []
        coord_ids = []
        print("loading nodes")
        for node in tqdm(nodes):
            coords.append(node['coord'])
            coord_ids.append(node['nid'])
            if require_rtree:
                rtree_idx.insert(node['nid'], tuple(node['coord']+node['coord']), obj=node)
        self.kdtree = KDTree(np.array(coords))
        self.coord_ids = coord_ids
        self.rtree = rtree_idx

    
    def get_pn_links(self,k,dis_thres):
        # augment graph by adding knn edges
        # to be finished
        for node in self.G.nodes:
            coord = self.G.nodes[node]['coord']
            d, nbrs = self.kdtree.query(coord, k, p=2)
            nbrs = [self.coord_ids[i] for i in nbrs]
            print(d,nbrs,sep='\n')
        return
    

    def get_skeleton(self):
        # get all segments 
        # then interp the sparse points to get dense skeleton
        segs, _ = self.get_segs_within(self.image.roi)
        img = self.image.from_roi(self.image.roi)
        skel = []
        for seg in tqdm(segs):
            for src, tar in zip(seg[:-1],seg[1:]):
                sa = AStarSearch(img,src,tar)
                path = None
                try:
                    path = sa.search()
                except:
                    print(f"Can't solve path from {src} to {tar}")
                    path = [src, tar]
                skel += path
        skel = np.array(skel)
        mask = np.zeros_like(img,dtype=np.uint16)
        mask[skel[:, 0], skel[:, 1], skel[:, 2]] = 1
        return mask


    def get_segs_within(self,roi):
        # get segs within roi, return a list of lists of nodes
        # 1. query nodes within roi
        # 2. generate subgraph
        # 3. remove branches, then traverse every connected components
        # 4. filter out short paths

        nbrs = list(self.rtree.intersection(tuple(roi[0:3]+[i+j for i,j in zip(roi[:3],roi[3:])]), objects=False))
        sub_g = self.G.subgraph(nbrs).copy()
        branch_nodes = [node for node, degree in sub_g.degree() if degree >= 3]
        sub_g.remove_nodes_from(branch_nodes) 
        connected_components = list(nx.connected_components(sub_g))
        segs = []
        for nodes in connected_components:
            end_nodes = [node for node in nodes if sub_g.degree[node] == 1]
            if (len(end_nodes)!=2):
                continue
            path = nx.shortest_path(sub_g, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
            path = [[i-j for i,j in zip(self.G.nodes[node]['coord'],roi[0:3])] for node in path]
            segs.append(path)
        # get intensity value

        intens = []
        if self.image is not None:
            img = self.image.from_roi(roi)
            for seg in segs:
                seg_intens = []
                for coord in seg:
                    seg_intens.append(img[coord[0],coord[1],coord[2]])
                intens.append(seg_intens)

        return segs, intens


    def get_segs_by(self,creator,len_thres=5):
        edges_to_include = [(u, v) for u, v, attr in self.G.edges(data=True) if attr.get('creator') == creator]
        sub_g = self.G.edge_subgraph(edges_to_include).copy()
        branch_points = [node for node, degree in sub_g.degree() if degree >= 3]
        sub_g.remove_nodes_from(branch_points)
        connected_components = list(nx.connected_components(sub_g))
        segs = []
        for nodes in connected_components:
            if len(nodes)<=len_thres:
                continue
            subgraph = sub_g.subgraph(nodes).copy()
            end_nodes = [node for node, degree in subgraph.degree() if degree == 1]
            if (len(end_nodes)!=2):
                continue
            path = nx.shortest_path(subgraph, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
            # path to segment
            segs.append([subgraph.nodes[i]['coord'] for i in path])
        return segs


if __name__ == '__main__':
    # visualize segs within given roi
    '''
    db_path = '/Users/bean/workspace/data/RM009_axons_1.db'
    image_path = '/Users/bean/workspace/data/RM009_axons_1.tif'
    neurites = Neurites(db_path,image_path=image_path)
    roi = [0,0,0,300,300,300]
    segs,intens = neurites.get_segs_within(roi)
    print(len(segs),len(intens))
    import napari
    viewer = napari.Viewer(ndisplay=3)
    img = neurites.image.from_roi(roi)
    viewer.add_image(img)
    segs = sum(segs,[])
    viewer.add_points(np.array(segs),size=2)
    napari.run()
    '''
    db_path = '/Users/bean/workspace/data/labeled_blocks/fmost_test.db'
    image_path = '/Users/bean/workspace/data/labeled_blocks/fmost_test.tif'
    neurites = Neurites(db_path,image_path=image_path)
    mask = neurites.get_skeleton()
    import napari
    viewer = napari.Viewer(ndisplay=3)
    img = neurites.image.from_roi(neurites.image.roi)
    viewer.add_image(img)
    viewer.add_image(mask)
    napari.run()
