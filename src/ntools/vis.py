import napari
import numpy as np
import random
from ntools.dbio import read_nodes,read_edges
from scipy.spatial import KDTree

def show_segs_as_instances(segs,viewer,size=0.8):
    '''
    segs: [
        [[x,y,z],[x,y,z],...],
        ...
    ]
    '''
    points = []
    colors = []
    num_segs = 0
    num_branches = 0
    for seg in segs:
        seg_color = random.random()
        points+=seg
        colors+=[seg_color for _ in seg]
        if len(seg)>=2:
            num_segs+=1
        if len(seg)==1:
            num_branches+=1

    colors = np.array(colors)
    colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
    properties = {
        'colors': colors
    }

    print(f'num of segs (length >= 2): {num_segs}')
    print(f'num of branch points: {num_branches}')
    print(f'num of points: {len(points)}')
    point_layer = viewer.add_points(np.array(points),ndim=3,face_color='colors',size=size,edge_color='colors',shading='spherical',edge_width=0,properties=properties,face_colormap='turbo')



def vis_edges_by_creator(viewer,db_path,color_dict):
    '''
    visualize edges by there creators
    color_dict: {
        'creator1': color1,
        'creator2': color2,
        'default': default_color
        ...
    }
    '''
    # find all edges labeled manually
    nodes = read_nodes(db_path)
    edges = read_edges(db_path)
    nodes = {n['nid']: n for n in nodes}
    edges = [[e['src'],e['des'],e['creator']] for e in edges]
    edges = [edge for edge in edges if edge[0]<edge[1]]

    vectors = []
    v_colors = []
    for edge in edges:
        [src,tar,creator] = [nodes[edge[0]]['coord'],nodes[edge[1]]['coord'],edge[2]]
        v = [j-i for i,j in zip(src,tar)]
        p = src
        vectors.append([p,v])
        if creator in color_dict.keys():
            v_colors.append(color_dict[creator])
        else:
            v_colors.append(color_dict['default'])
    
    viewer.add_vectors(vectors,edge_color=v_colors,edge_width=2,vector_style='line')


def compare(gt,pred1,pred2=None):
    gt_nodes = read_nodes(gt)
    gt_coords = [n['coord'] for n in gt_nodes]
    gt_nodes = {n['nid']: n for n in gt_nodes}
    gt_nids = gt_nodes.keys()
    gt_tree = KDTree(np.array(gt_coords))

    pred1_nodes = read_nodes(pred1)
    pred1_coords = [n['coord'] for n in pred1_nodes]
    pred1_nodes = {n['nid']: n for n in pred1_nodes}
    pred1_nids = list(pred1_nodes.keys())
    pred1_tree = KDTree(pred1_coords)

    pred2_nodes = read_nodes(pred2)
    pred2_coords = [n['coord'] for n in pred2_nodes]
    pred2_nodes = {n['nid']: n for n in pred2_nodes}
    pred2_nids = list(pred2_nodes.keys())
    pred2_tree  = KDTree(np.array(pred2_coords))
    print(f"Ground truth length: {len(gt_nodes)}")


    distances, _ = pred1_tree.query(np.array(gt_coords))
    recalled = np.where(distances <= 4)[0]
    false_negative = np.where(distances > 4)[0]
    print(f"SR recall: {len(recalled)/len(gt_coords)}")


    distances, _ = gt_tree.query(np.array(pred1_coords))
    true_pos = np.where(distances <= 4)[0]
    false_pos = np.where(distances > 4)[0]
    true_pos = [pred1_nids[i] for i in true_pos]
    false_pos = [pred1_nids[i] for i in false_pos]
    print(f"SR precision: {len(true_pos)/len(pred1_coords)}")

    pred2_nodes = read_nodes(pred2)
    pred2_coords = [n['coord'] for n in pred2_nodes]
    pred2_nodes = {n['nid']: n for n in pred2_nodes}
    pred2_nids = list(pred2_nodes.keys())
    pred2_tree  = KDTree(np.array(pred2_coords))

    distances, _ = pred2_tree.query(np.array(gt_coords))
    recalled = np.where(distances <= 4)[0]
    false_negative = np.where(distances > 4)[0]
    print(f"Baseline recall: {len(recalled)/len(gt_coords)}")


    distances, _ = gt_tree.query(np.array(pred2_coords))
    true_pos = np.where(distances <= 4)[0]
    false_pos = np.where(distances > 4)[0]
    true_pos = [pred2_nids[i] for i in true_pos]
    false_pos = [pred2_nids[i] for i in false_pos]
    print(f"Baseline precision: {len(true_pos)/len(pred2_coords)}")


    # nodes = read_nodes(pred1)
    # nodes = {n['nid']: n for n in nodes}
    # edges = [[e['src'],e['des'],e['creator']] for e in pred1_edges]
    # edges = [edge for edge in edges if edge[0]<edge[1]]
    # vectors = []
    # v_colors = []

    # for edge in edges:
    #     [src,tar,creator] = [nodes[edge[0]]['coord'],nodes[edge[1]]['coord'],edge[2]]
    #     v = [j-i for i,j in zip(src,tar)]
    #     p = src
    #     vectors.append([p,v])
    #     if edge[0] in margin or edge[1] in margin:
    #         v_colors.append('red')
    #     else:
    #         v_colors.append('orange')
    

    # viewer = napari.Viewer(ndisplay=3)
    # viewer.add_vectors(vectors,edge_color=v_colors,edge_width=2,vector_style='line')
    # napari.run()



def segs_as_paths(db_path):
    '''
    Analyse connectivity, extract connected components
    Extract branch points, visualize them as points
    Extract paths between branch points, visualize them as shape (paths)
    '''
    pass


if __name__ == '__main__':
    # colorize edges by creators
    '''
    from ntools.image_reader import wrap_image
    db_path = '/home/bean/workspace/data/RM009_arbor_1.db'
    image_path = '/home/bean/workspace/data/RM009_arbor_1.tif'
    color_dict = {
        'tester': 'red',
        'seger': 'yellow',
        'astar': 'red',
        'default': 'white'
    }
    viewer = napari.Viewer(ndisplay=3)
    image = wrap_image(image_path)
    img = image.from_roi(image.roi)
    viewer.add_image(img)
    vis_edges_by_creator(viewer,db_path,color_dict)
    napari.run()
    '''

    '''
    gt_path = '/home/bean/workspace/data/RM009_axons_2.db'
    pred1_path = '/home/bean/workspace/data/RM009_axons_2_sr.db'
    pred2_path = '/home/bean/workspace/data/RM009_axons_2_baseline.db'
    compare(gt_path,pred1_path,pred2_path)
    '''
