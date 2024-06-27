import napari
import numpy as np
import random
from ntools.dbio import read_nodes,read_edges


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




if __name__ == '__main__':
    from ntools.image_reader import wrap_image
    db_path = '/Users/bean/workspace/data/RM009_arbor_1.db'
    image_path = '/Users/bean/workspace/data/RM009_arbor_1.tif'
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

