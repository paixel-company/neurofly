import numpy as np
from scipy.interpolate import splprep, splev

def FSM(
    ctrl_p: np.ndarray,
    degree: int=4,
    sample_num: int=20
):
    '''
        Frenet–Serret Frame
    '''
    x = ctrl_p[:,0]
    y = ctrl_p[:,1]
    z = ctrl_p[:,2]
    tck, u = splprep([x,y,z], s=0, k=degree)
    new_u = np.linspace(0, 1, sample_num)
    P = np.asarray(splev(new_u, tck=tck)).transpose()

    # Tangent
    T = np.asarray(splev(new_u, tck, der=1)).transpose()
    T = T/np.linalg.norm(T, axis=1, keepdims=True)

    # Norm
    C = np.asarray(splev(new_u, tck, der=2)).transpose()
    C = C/np.linalg.norm(C, axis=1, keepdims=True)
    projection = np.sum(C * T, axis=1, keepdims=True) * T
    N = C - projection
    # N = N/np.linalg.norm(N, axis=1, keepdims=True)

    # Binorm
    B = np.cross(T, N)
    B = B/np.linalg.norm(B, axis=1, keepdims=True)
    
    return P,T,N,B


def double_reflection_method(P, T, N_0):
    def normalize(v):
        return v / np.linalg.norm(v)
    
    n = len(P) - 1
    frames = []

    N_0 = normalize(N_0)
    B_0 = np.cross(T[0], N_0)
    frames.append((T[0], N_0, B_0))

    for i in range(n):
        v_R1 = P[i + 1] - P[i]
        c_R1 = np.dot(v_R1, v_R1)
        N_perp = frames[i][1] - (2 / c_R1) * np.dot(v_R1, frames[i][1]) * v_R1
        T_perp = T[i] - (2 / c_R1) * np.dot(v_R1, T[i]) * v_R1
        v_R2 = T[i + 1] - T_perp
        c_R2 = np.dot(v_R2, v_R2)
        N_next = N_perp - (2 / c_R2) * np.dot(v_R2, N_perp) * v_R2
        N_next = normalize(N_next)
        B_next = np.cross(T[i + 1], N_next)
        frames.append([T[i + 1], N_next, B_next, ])
    
    frames = np.asarray(frames)
    return frames


def RMF(
    ctrl_p: np.ndarray,
    degree: int=4,
    sample_num: int=20
):
    '''
        Rotation-Minimizing Frame
    '''
    P,T,N,B = FSM(ctrl_p, degree, sample_num)
    RMF_frames = double_reflection_method(P, T, N[0])
    RMF_T = RMF_frames[:,0,:]
    RMF_N = RMF_frames[:,1,:]
    RMF_B = RMF_frames[:,2,:]
    RMF_T = RMF_T/np.linalg.norm(RMF_T, axis=1, keepdims=True)
    RMF_N = RMF_N/np.linalg.norm(RMF_N, axis=1, keepdims=True)
    RMF_B = RMF_B/np.linalg.norm(RMF_B, axis=1, keepdims=True)

    return P, RMF_T, RMF_N, RMF_B


if __name__ == '__main__':
    import json
    import tifffile as tiff
    import napari

    img_index = 1027
    json_path = f'/home/ryuuyou/E5/project/data/routes_wp64_1k/test/json/img_{img_index}.json'
    with open(json_path) as f:
        wp = json.load(f)
    wp = np.asarray(wp)

    img_path = f'/home/ryuuyou/E5/project/data/routes_wp64_1k/test/img/img_{img_index}.tif'
    img = tiff.imread(img_path)

    P, RMF_T, RMF_N, RMF_B = RMF(wp)

    RMF_T_v = np.stack([P,RMF_T], axis=1)
    RMF_N_v = np.stack([P,RMF_N], axis=1)
    RMF_B_v = np.stack([P,RMF_B], axis=1)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img)
    viewer.add_points(P, size=1, face_color='#00ffff')
    viewer.add_points(wp, size=1, face_color='#5500ff')
    viewer.add_vectors(RMF_T_v, edge_width=0.2, length=2, vector_style='arrow', edge_color='blue', name='Tangent')
    viewer.add_vectors(RMF_N_v, edge_width=0.2, length=2, vector_style='arrow', edge_color='red', name='Norm')
    viewer.add_vectors(RMF_B_v, edge_width=0.2, length=2, vector_style='arrow', edge_color='green', name='Binorm')
    napari.run()