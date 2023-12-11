import napari
import numpy as np
import random


def show_segs_and_image(img,points,vectors):
    viewer = napari.Viewer(ndisplay=3)
    image_layer = viewer.add_image(img)
    point_layer = viewer.add_points(points,ndim=3,face_color='cyan',size=1,name='start point',edge_color='black',shading='spherical')
    vector_layer = viewer.add_vectors(vectors,edge_color='orange',edge_width=1)
    napari.run()


def show_skels_and_image(img,points):
    viewer = napari.Viewer(ndisplay=3)
    image_layer = viewer.add_image(img)
    point_layer = viewer.add_points(points,ndim=3,face_color='cyan',size=1,name='start point',edge_color='black',shading='spherical')
    napari.run()



def show_segs_as_instances(segs,viewer):
    print(f'num of segs: {len(segs)}')
    points = []
    colors = []
    for seg in segs:
        seg_color = random.random()
        points+=seg
        colors+=[seg_color for _ in seg]

    properties = {
        'colors': np.array(colors)
    }
    print(f'num of points: {len(points)}')
    point_layer = viewer.add_points(np.array(points),ndim=3,face_color='colors',size=0.8,edge_color='black',shading='spherical',properties=properties,face_colormap='hsv')



def show_connected_segs(segs,viewer):
    '''
    segment:
        {
            sid: int,
            head: coord,
            tail: coord,
            points: [head,...,tail],
            sampled_points: points[::interval]
            head_n = [ids],
            tail_n = [ids],
            side_n = [[id,index_of_point],...],
            checked = int
        }
    '''

    pass
