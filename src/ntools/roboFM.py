import numpy as np
from scipy.interpolate import splprep, splev
from einops import reduce, einsum


def get_frenet_frame(
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
    P,T,N,_,k = get_frenet_frame(ctrl_p, degree, sample_num)
    RMF_frames = double_reflection_method(P, T, N[0])
    RMF_T = RMF_frames[:,0,:]
    RMF_N = RMF_frames[:,1,:]
    RMF_B = RMF_frames[:,2,:]
    RMF_T = RMF_T/np.linalg.norm(RMF_T, axis=1, keepdims=True)
    RMF_N = RMF_N/np.linalg.norm(RMF_N, axis=1, keepdims=True)
    RMF_B = RMF_B/np.linalg.norm(RMF_B, axis=1, keepdims=True)

    return P, RMF_T, RMF_N, RMF_B, N ,k



def predict_next_frame(position,frame,k,step_size=5):
    '''
    Given position, frame and predicted curvature, calculate parabola and sample next frame from it according to given step size.
    1. project curvature to n1, n2
    2. write down taylor expansion: r(s_0+delta_s) = r0 + delta_s*t0 + delta_s^2*k/2
    3. calculate velocity and get dr
    4. sample next position according to r(s) and dr
    5. generate new frame
    
    positon: r, (3,)
    frame: (t,n1,n2) all (3,) array
    k: k1,k2, (2,) parameterizing curvature vector k = k1*n1+k2*n2
    '''
    rmf_t, rmf_n1, rmf_n2 = frame
    r = position
    k1, k2 = k
    kappa = np.sqrt(k1**2+k2**2)

    # suppose predicted curvature vector k = N x kappa, use k and rmf_t to estimate Frenet frame (eT,eN,eB)
    phi = np.arccos(np.clip(k1 / kappa, a_min=-1, a_max=1))
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    eN = cosphi * rmf_n1 - sinphi * rmf_n2 
    eN /= np.sqrt(np.sum(eN ** 2, axis=-1))
    eT = rmf_t
    eB = np.cross(eT, eN, axis=-1)
    k_vector = k1 * rmf_n1 + k2 * rmf_n2

    # calculate velocity according to curvature (adaptive step size mentioned in roboEm paper)
    step_factor = 1
    base_step = 3
    radius = 32
    vmin = 1
    v = step_factor * base_step / (1.0 + radius * kappa)
    v = np.maximum(vmin * base_step, v)

    # calculate delta_s, then the next frame
    norm = 1 / np.sqrt(1 + v**2 * kappa**2) # norm of the forward step
    delta_s = v * eT + v**2 / 2.0 * k_vector
    next_r = r + delta_s
    next_t = norm * (eT + v * k_vector)
    next_n1 = cosphi * norm * (eN - v * kappa * eT) + sinphi * eB
    next_n2 = np.cross(next_t,next_n1)

    return next_r, next_t, next_n1, next_n2


def visalize_frames(trajs,image):
    points = []
    curvatures = [] 
    rmf_n1s = []
    rmf_n2s = []
    rmf_ts = []
    norms = []
    for traj in trajs:
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




if __name__ == '__main__':
    import napari
    import networkx as nx
    from ntools.image_reader import wrap_image
    from ntools.dbio import read_nodes, read_edges, get_edges_by
    img_path = '/Users/bean/workspace/data/RM009_axons_1.tif'
    db_path = '/Users/bean/workspace/data/RM009_axons_1.db'
    image = wrap_image(img_path)
    nodes = read_nodes(db_path)
    edges = read_edges(db_path)

    G = nx.Graph()

    for node in nodes:
        G.add_node(node['nid'], nid = node['nid'],coord = node['coord'], type = node['type'], checked = node['checked'])

    for edge in edges:
        G.add_edge(edge['src'],edge['des'],creator = edge['creator'])

    branch_nodes = [node for node, degree in G.degree() if degree >= 3]
    G.remove_nodes_from(branch_nodes)
    connected_components = list(nx.connected_components(G))
    segments = []
    for cc in connected_components:
        if len(cc) <= 10:
            continue
        sub_g = G.subgraph(list(cc))
        ends = [node for node, degree in sub_g.degree() if degree == 1]
        start_node = ends[0]
        traversal = list(nx.bfs_edges(sub_g, start_node))
        traversal_path = [start_node] + [v for u, v in traversal]
        segments.append(traversal_path)

    segments.sort(key=len)
    segments = segments[::-1]
    segments = [[G.nodes[i]['coord'] for i in seg] for seg in segments if len(seg)>10]


    # visalize_frames(trajs[10:20],image)
    '''
    given segments as list of coordinates
    calculate r,t,n1,n2,k
    '''
    traj = segments[0]
    wp = np.array(traj)
    tarj_length = len(wp)
    r, rmf_t, rmf_n1, rmf_n2, frenet_N, curvature = RMF(wp,sample_num=tarj_length*3)
    k = np.multiply(frenet_N,curvature)

    k1 = rmf_n1*(einsum(k,rmf_n1,'i j, i j -> i')/reduce(rmf_n1**2, 'i j->i', 'sum')**0.5)[:,np.newaxis]
    k2 = rmf_n2*(einsum(k,rmf_n2,'i j, i j -> i')/reduce(rmf_n2**2, 'i j->i', 'sum')**0.5)[:,np.newaxis]

    k1 = einsum(k1**2, 'i j -> i')**0.5
    k2 = einsum(k2**2, 'i j -> i')**0.5
    pred_k = np.stack((k1, k2), axis=1) 

    fi = len(r)-1 #frame index
    nr, nt, nn1, nn2 = predict_next_frame(r[fi],(rmf_t[fi],rmf_n1[fi],rmf_n2[fi]),pred_k[fi])


    r = r[0:fi]
    t = rmf_t[0:fi]
    n1 = rmf_n1[0:fi]
    n2 = rmf_n2[0:fi]

    r = np.vstack([r,nr])
    t = np.vstack([t,nt])
    n1 = np.vstack([n1,nn1])
    n2 = np.vstack([n2,nn2])

    rmf_tv = np.stack([r,t], axis=1)
    rmf_n1v = np.stack([r,n1], axis=1)
    rmf_n2v = np.stack([r,n2], axis=1)

    offset = []
    size = []
    for i in range(3):
        offset.append(int(np.min(r[:,i])))
        size.append(int(np.max(r[:,i]-np.min(r[:,i]))))

    offset = [i-20 for i in offset]
    size = [i+20*2 for i in size]
    roi = offset + size
    img = image.from_roi(roi)


    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img,translate=offset)
    viewer.add_vectors(rmf_tv, edge_width=0.1, length=2, vector_style='arrow', edge_color='blue', name='Tangent')
    viewer.add_vectors(rmf_n1v, edge_width=0.1, length=2, vector_style='arrow', edge_color='red', name='N1')
    viewer.add_vectors(rmf_n2v, edge_width=0.1, length=2, vector_style='arrow', edge_color='orange', name='N2')
    napari.run()

