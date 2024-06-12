import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from ntools.dbio import read_edges, read_nodes, delete_nodes, add_nodes, add_edges, check_node, uncheck_nodes, change_type
from magicgui import magicgui, widgets
from ntools.image_reader import wrap_image


class Neurites():
    '''
    Neurites class represents neurites with nodes and associated links between them. It integrates KDTree for efficient spatial querying and NetworkX graph for exploring and retrieving based on graph structure. This allows both proximity-based searches and graph-based operations.
    '''
    def __init__(self,db_path,image_path=None):
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

        coords = []
        coord_ids = []
        for node in nodes:
            coords.append(node['coord'])
            coord_ids.append(node['nid'])
        self.kdtree = KDTree(np.array(coords))
        self.coord_ids = coord_ids

    
    def get_pn_links(self,k,dis_thres):
        # augment graph by adding knn edges
        for node in self.G.nodes:
            coord = self.G.nodes[node]['coord']
            d, nbrs = self.kdtree.query(coord, k, p=2)
            nbrs = [self.coord_ids[i] for i in nbrs]
            print(d,nbrs,sep='\n')
        return


if __name__ == '__main__':
    db_path = '/Users/bean/workspace/data/RM009_arbor_1.db' 
    image_path = '/Users/bean/workspace/data/RM009_arbor_1.tif' 
    neurites = Neurites(db_path,image_path=image_path)
    neurites.get_pn_links(k=5,dis_thres=20)
