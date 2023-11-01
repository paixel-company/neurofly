import napari
import numpy as np


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

