import numpy as np
import networkx as nx
import treelib
from treelib import Node, Tree
from scipy.spatial import KDTree
from tqdm import tqdm
import json
import glob
import random


class Neuron():
    '''
    From .swc or .lym to Tree and Graph stucture.
    For visualization, proofreading, data cleaning, etc.
    '''

    def __init__(self,neuron_dir,scale=1):
        self.root = None
        self.nodes, self.edges = self._read_neuron(neuron_dir,scale)
        self.graph = self.gen_graph()
        self.tree = self.gen_tree()
        self.bfs_tree = list(self.tree.expand_tree(mode=Tree.WIDTH))


    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        # make neuron tree iteable, return tree node containing branch point -> end
        if self.i < len(self.bfs_tree):
            node_data = self.tree.get_node(self.bfs_tree[self.i])
            tid = node_data.identifier
            points = node_data.data
            nodes = [self.nodes[i] for i in points] 
            nodes_by_id = dict((d['nid'], dict(d)) for (index, d) in enumerate(nodes))
            self.i += 1
            return tid, nodes_by_id
        raise StopIteration
    

    def get_branch_nodes(self):
        # return branch nodes
        b_nodes = {}
        for i,(nid,node) in enumerate(self.nodes.items()):
            if node['type'] == 'branch':
                b_nodes[nid] = node
        return b_nodes


    def get_end_nodes(self):
        # return branch nodes
        e_nodes = {}
        for i,(nid,node) in enumerate(self.nodes.items()):
            if node['type'] == 'end':
                e_nodes[nid] = node
        return e_nodes
    

    def get_nbrs(self,nid,n_nbr=3):
        neighbors = list(self.graph.neighbors(nid))[:n_nbr]
        return neighbors
    

    def _read_neuron(self,neuron_dir,scale=1):
        nodes = []
        edges = []
        # TODO: remove circles from lyp files if exist
        if neuron_dir.endswith('.lyp'):
            with open(neuron_dir) as neuron_file:
                data = json.load(neuron_file)
                for node in data['nodes']:
                    type = 'route'
                    nid = int(node['id'])
                    if 'parent_ids' in node.keys():
                        # only one parent allowed
                        pid = int(node['parent_ids'].split(' ')[0])
                    else:
                        pid = -1
                        type = 'branch'
                        self.root = nid
                    pos = node['position'].split(' ')
                    pos = [int(float(i)) for i in pos]
                    nodes.append(
                        {
                            'nid':nid,
                            'pid':pid,
                            'pos':pos,
                            'type':type
                        }
                    )
                    if(pid!=-1):
                        edges.append([pid,nid])
            neuron_file.close()


        elif neuron_dir.endswith('.swc'):
            scale=scale
            d = np.loadtxt(neuron_dir)
            tr = np.int32(d[:,np.array([0,6,1])]).tolist()#[nid, pid, type]
            pos = np.int32(d[:, 2:5]*scale).tolist()#[x,y,z]
            for tr,pos in zip(tr,pos):
                pid = tr[1]
                if pid==-1:
                    self.root = tr[0]
                    type = 'root'
                else:
                    type = 'route'
                    edges.append([tr[1],tr[0]])
                nodes.append(
                    {
                        'nid':tr[0], #node id 
                        'pid':tr[1], #parent id
                        'pos':pos,
                        'type':type
                    }
                )

        nodes_by_id = dict((d['nid'], dict(d)) for (index, d) in enumerate(nodes))

        return nodes_by_id,edges


    def gen_graph(self):
        G = nx.Graph()
        for i,(nid,node) in enumerate(self.nodes.items()):
            G.add_node(node['nid'],pos=node['pos'],type=node['type'],pid=node['pid'])
        for edge in self.edges:
            G.add_edge(edge[0],edge[1])
        degrees = np.array(G.degree())
        branch_points = degrees[np.where(degrees[:,1]>2)][:,0]
        end_points = degrees[np.where(degrees[:,1]==1)][:,0]

        for p in branch_points:
            G.nodes[p]['type']='branch'
            self.nodes[p]['type']='branch'

        for p in end_points:
            G.nodes[p]['type']='end'
            self.nodes[p]['type']='end'

        if self.root is None:
            self.root = end_points[0] 
        
        G.nodes[self.root]['type'] = 'root'
        self.nodes[self.root]['type'] = 'root'

        return G


    def gen_tree(self):
        '''
        Structure of neuron tree:
        root: [identifier: soma id (by default 0)]
        node: [
            identifier: id of the last point in this branch, can be an end of branch,
            data: [head_id, ... , tail_id]
            # head can be root or branch point
            # tail can be end or branch point
        ]

        '''
        tree = Tree()
        if nx.is_tree(self.graph):
            dfs_tree = nx.dfs_tree(self.graph,source=self.root)
            tree.create_node(tag='root',identifier=self.root,data=[self.root])
            branch = []
            for node in list(dfs_tree)[1:]:
                t = self.nodes[node]['type'] 
                if t=='route':
                    branch.append(node)
                    continue
                if t=='branch':
                    branch.append(node)
                    #find parent tree node of this branch
                    pid = self.nodes[branch[0]]['pid']
                    if tree.get_node(pid) is not None:
                        parent = pid
                        branch.insert(0,pid)
                    else:
                        parent = self.root
                    #-------------------
                    tree.create_node(tag=f'{branch[0]}->{branch[-1]}',identifier=node,data=branch,parent=parent)
                    branch = []
                    continue
                if t=='end':
                    branch.append(node)
                    pid = self.nodes[branch[0]]['pid']
                    if tree.get_node(pid) is not None:
                        parent = pid
                        branch.insert(0,pid)
                    else:
                        parent = self.root
                    tree.create_node(tag=f'{branch[0]}->{branch[-1]}',identifier=node,data=branch,parent=parent)
                    branch = []
                    continue
            return tree
        else:
            print('No tree structure found in this file, failed to load neuron.')
            return None


    def _get_ones_streaks(self,L,min_size=3):
        breaks = [i for i,(a,b) in enumerate(zip(L,L[1:]),1) if a!=b]
        ran = [[s,e-1] for s,e in zip([0]+breaks,breaks+[len(L)])
                            if e-s>=min_size and L[s]]
        if(len(ran))>=1:
            return [np.arange(s,e+1).tolist() for (s,e) in ran]
        else:
            return None


    def gen_segs(self,roi,min_len):
        segs = []
        [ox,oy,oz]=roi[0:3]
        [ex,ey,ez]=[i+j for i,j in zip(roi[0:3],roi[3:6])]
        if self.tree is not None:
            for segment in self.tree.all_nodes_itr():
                points = segment.data
                nodes = [self.nodes[i] for i in points] 
                coords = np.array([self.nodes[i]['pos'] for i in points])
                # some ugly code
                x_in = np.logical_and(coords[:,0]>ox,coords[:,0]<ex)
                y_in = np.logical_and(coords[:,1]>oy,coords[:,1]<ey)
                z_in = np.logical_and(coords[:,2]>oz,coords[:,2]<ez)
                c_in = np.logical_and(x_in,y_in)
                c_in = np.logical_and(c_in,z_in).tolist()
                # -----------
                segments_within_roi = self._get_ones_streaks(c_in,min_size=min_len)
                if segments_within_roi is not None:
                    for seg in segments_within_roi:
                        nodes_segment = list(map(lambda i: nodes[i], seg))
                        segs.append(nodes_segment)
        return segs


    def replace_branch(self,new_branch,tid):
        '''
        assign new ids for refined points on current branch (given tree node id)
        delete old ids in self.nodes, then add new ones
        in principle, branch points are regarded precise and kept
        '''
        if len(new_branch)<2:
            return 0
        max_ind = max(self.nodes.keys())+1
        # assign new ids
        for i,nnode in enumerate(new_branch[1:-1]):
            nnode['nid'] = i+max_ind 
            nnode['pid'] = i+max_ind-1
        new_branch[1]['pid'] = new_branch[0]['nid']
        new_branch[-1]['pid'] = new_branch[-2]['nid']
        for node in new_branch:
            self.nodes[node['nid']]=node
        self.tree[tid].data = [d['nid'] for (index, d) in enumerate(new_branch)]


    def save_neuron(self,path):
        '''
        save neuron tree as swc file
        swc format: [node_id,type,x,y,z,diameter,parent_id]
        types: 1 <-> root(soma)
               0 <-> route/branch point
               8 <-> end point

        Types are not necessary, for all these information hide in the graph structure. By default, node 0 also represents soma which have a p_id of -1.

        Before finding a way to calculate it, the value of diameter is set to 1.

        '''
        type_dict = {
            'root': 1,
            'branch': 0,
            'route': 0,
            'end': 8
        }

        nodes = []
        for tn in list(self.tree.expand_tree(mode=Tree.DEPTH)):
            nodes += self.tree[tn].data
        nodes = list(set(nodes))

        n_dict = {}
        for i,node in enumerate(nodes):
            n_dict[node] = i # assign consecutive ids
        n_dict[-1] = -1

        with open(path, 'w') as f:
            f.write('# refined swc by Rubin Zhao'+'\n')
            for n in nodes:
                nid = n_dict[n]
                [x,y,z] = self.nodes[n]['pos']
                pid = n_dict[self.nodes[n]['pid']]
                t = type_dict[self.nodes[n]['type']]
                dia = 1
                line = [str(i) for i in [nid,t,x,y,z,dia,pid]]
                f.write(' '.join(line)+'\n')
        f.close()



def show_neurons(neurons,viewer=None):
    from tqdm import tqdm
    if viewer == None:
        import napari
        viewer = napari.Viewer(ndisplay=3,title='neurons')
    point_roi = [0,0,0,70000,70000,70000]
    v_colors = []
    vectors = []
    p_colors = []
    points = []
    for neuron in tqdm(neurons):
        neuron_color = random.random()
        segs = neuron.gen_segs(roi=point_roi,min_len=0) 
        for seg in segs:
            for n1,n2 in zip(seg[:-1],seg[1:]):
                p1=n1['pos']
                p2=n2['pos']
                v = [j-i for i,j in zip(p1,p2)]
                p = [i-j for i,j in zip(p1,point_roi[0:3])] 
                vectors.append([p,v]) 
                v_colors.append(neuron_color)
            for node in seg:
                points.append([i-j for i,j in zip(node['pos'],point_roi[0:3])])
                t = node['type']
                if t == 'branch':
                    p_colors.append(0.3)
                elif t == 'route':
                    p_colors.append(0.4)
                else:
                    p_colors.append(0.7)

    vector_properties = {
        'color': np.array(v_colors),
    }

    point_properties = {
        'color': np.array(p_colors)
    }

    vectors = np.array(vectors)
    points = np.array(points)

    vector_layer = viewer.add_vectors(vectors,properties=vector_properties,name='connections',edge_colormap='hsv',edge_color='color',edge_width=0.5,opacity=1)
    point_layer = viewer.add_points(points,properties=point_properties,size=1,name='points',face_colormap='plasma',face_color='color')



def show_neuron_segs(neurons,viewer=None):
    from tqdm import tqdm
    if viewer==None:
        import napari
        viewer = napari.Viewer(ndisplay=3,title='segs')
    point_roi = [0,0,0,70000,70000,70000]
    for neuron in tqdm(neurons):
        segs = neuron.gen_segs(roi=point_roi,min_len=0) 
        v_colors = []
        vectors = []
        p_colors = []
        points = []

        for seg in segs:
            seg_color = random.random()
            for n1,n2 in zip(seg[:-1],seg[1:]):
                p1=n1['pos']
                p2=n2['pos']
                v = [j-i for i,j in zip(p1,p2)]
                p = [i-j for i,j in zip(p1,point_roi[0:3])] 
                vectors.append([p,v]) 
                v_colors.append(seg_color)
            for node in seg:
                points.append([i-j for i,j in zip(node['pos'],point_roi[0:3])])
                t = node['type']
                if t == 'branch':
                    p_colors.append(0.3)
                elif t == 'route':
                    p_colors.append(0.4)
                else:
                    p_colors.append(0.7)

        vector_properties = {
            'color': np.array(v_colors),
        }

        point_properties = {
            'color': np.array(p_colors)
        }

        vectors = np.array(vectors)
        points = np.array(points)

        vector_layer = viewer.add_vectors(vectors,properties=vector_properties,name='connections',edge_colormap='hsv',edge_color='color',edge_width=0.5,opacity=1)
        point_layer = viewer.add_points(points,properties=point_properties,size=1,name='points',face_colormap='plasma',face_color='color')



def get_branches(neuron_files,roi,ids=[],scale=1):
    branches=[]
    if len(ids)==0:
        for i, neuron in enumerate(tqdm(neuron_files)):
            n = Neuron(neuron,scale=scale)
            segs = n.gen_segs(roi,min_len=0) 
            branches.append(segs)
            if(len(segs)>0):
                print(i,neuron)
    else:
        neuron_files = [neuron_files[i] for i in ids]
        for i, neuron in enumerate(neuron_files):
            # print(neuron)
            n = Neuron(neuron,scale=scale)
            segs = n.gen_segs(roi,min_len=0) 
            branches.append(segs)

    return branches



def get_points_and_vecs(segs,point_roi):
    vectors = []
    points = []
    segs = sum(segs,[])
    for seg in segs:
        for n1,n2 in zip(seg[:-1],seg[1:]):
            p1=n1['pos']
            p2=n2['pos']
            v = [j-i for i,j in zip(p1,p2)]
            p = [i-j for i,j in zip(p1,point_roi[0:3])]
            vectors.append([p,v])
        for node in seg:
            points.append([i-j for i,j in zip(node['pos'],point_roi[0:3])])

    return np.array(vectors),np.array(points)


def save_segs(segs,path):
    with open(path, "w") as json_file:
        json.dump(segs, json_file, indent=4)


def read_segs(path):
    with open(path, "r") as json_file:
        segs = json.load(json_file)
    return segs

