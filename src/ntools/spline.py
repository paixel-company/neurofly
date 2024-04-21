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
    r_t = np.asarray(splev(new_u, tck, der=1)).transpose()
    T = r_t/np.linalg.norm(r_t, axis=1, keepdims=True)

    # Norm
    r_tt = np.asarray(splev(new_u, tck, der=2)).transpose() # second derivtive of r with respect to t
    eN = r_tt/np.linalg.norm(r_tt, axis=1, keepdims=True) # estimated norm

    projection = np.sum(eN * T, axis=1, keepdims=True) * T
    N = eN - projection
    N = N/np.linalg.norm(N, axis=1, keepdims=True) # nromalize N

    # Binorm
    B = np.cross(T, N)
    B = B/np.linalg.norm(B, axis=1, keepdims=True)

    # kappa
    k = np.linalg.norm(np.cross(r_t,r_tt),axis=1,keepdims=True)/np.linalg.norm(r_t,axis=1,keepdims=True)**3 
    
    return P,T,N,B,k


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
    P,T,N,B,k = FSM(ctrl_p, degree, sample_num)
    RMF_frames = double_reflection_method(P, T, N[0])
    RMF_T = RMF_frames[:,0,:]
    RMF_N = RMF_frames[:,1,:]
    RMF_B = RMF_frames[:,2,:]
    RMF_T = RMF_T/np.linalg.norm(RMF_T, axis=1, keepdims=True)
    RMF_N = RMF_N/np.linalg.norm(RMF_N, axis=1, keepdims=True)
    RMF_B = RMF_B/np.linalg.norm(RMF_B, axis=1, keepdims=True)

    return P, RMF_T, RMF_N, RMF_B, N ,k


if __name__ == '__main__':
    import json
    import napari
    from ntools.read_zarr import Image
    json_path = '/Users/bean/workspace/data/roi_dense1.json'
    img_path = '/Users/bean/workspace/data/roi_dense1.zarr'
    image = Image(img_path)
    with open(json_path) as f:
        neurites = json.load(f)

    trajs = [] # list of segments, each segment is a list of coordinates
    for neurite in neurites:
        for seg in neurite: 
            traj = []
            for node in seg:
                traj.append(node['pos'])
            trajs.append(traj)
    

    points = []
    curvatures = [] 
    rmf_n1s = []
    rmf_n2s = []
    rmf_ts = []
    norms = []
    for traj in trajs[10:20]:
        if len(traj)<=4:
            continue
        wp = np.array(traj) # way points
        tarj_length = len(wp)
        P, rmf_t, rmf_n1, rmf_n2, N, k = RMF(wp,sample_num=tarj_length*5)
        N = np.multiply(N,k)

        points+=P.tolist()
        rmf_ts+=rmf_t.tolist()
        rmf_n1s+=rmf_n1.tolist()
        rmf_n2s+=rmf_n2.tolist()
        curvatures+=k.tolist()
        norms+=N.tolist()


    rmf_tv = np.stack([points,rmf_ts], axis=1)
    rmf_n1v = np.stack([points,rmf_n1s], axis=1)
    rmf_n2v = np.stack([points,rmf_n2s], axis=1)
    curvature = np.stack([points,norms], axis=1)
    points = np.array(points)


    offset = []
    size = []
    for i in range(3):
        offset.append(int(np.min(points[:,i])))
        size.append(int(np.max(points[:,i]-np.min(points[:,i]))))

    roi = offset+size
    print(roi)
    img = image.from_roi(roi)


    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img,translate=offset)
    viewer.add_points(points, size=0.5, face_color='#00ffff')
    viewer.add_vectors(rmf_tv, edge_width=0.1, length=2, vector_style='arrow', edge_color='blue', name='Tangent')
    viewer.add_vectors(rmf_n1v, edge_width=0.1, length=2, vector_style='arrow', edge_color='red', name='N1')
    viewer.add_vectors(rmf_n2v, edge_width=0.1, length=2, vector_style='arrow', edge_color='orange', name='N2')
    viewer.add_vectors(curvature, edge_width=0.1, length=10, vector_style='arrow', edge_color='green', name='curvature')
    napari.run()
