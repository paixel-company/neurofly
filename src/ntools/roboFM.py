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


def gram_schmidt_orth(eT, eN1):
    """
    As errors can accumulate, this Code makes sure, that eT, eN1 and eN2 are still an ONS
    For this eT is assumed to be correct (up to normalization), eN1 is calculated by Gram-Schmidt
    and eN2 by vector cross product
    => There might be new errors due to the initial assumption of eT being correct
      (RoboEM is assumed to be able to deal with these and post-hoc correct them)
    """
    eT /= np.sqrt(np.sum(eT ** 2, axis=-1)).reshape(-1, 1, 1)
    eN1 /= np.sqrt(np.sum(eN1 ** 2, axis=-1)).reshape(-1, 1, 1)
    eN1 -= np.sum(eN1 * eT, axis=-1).reshape(-1, 1, 1) * eT
    eN1 /= np.sqrt(np.sum(eN1 ** 2, axis=-1)).reshape(-1, 1, 1)
    eN2 = np.cross(eT, eN1, axis=-1)
    eN2 /= np.sqrt(np.sum(eN2 ** 2, axis=-1)).reshape(-1, 1, 1)
    return eT, eN1, eN2



def exact_parabola_integration_step(tripod, k, vmin=0.1, stepsize_factor=1.0, base_step=11.24):
    """
    Assuming an exact parabola based on the Bishop curvatures k1, k2, we can write down the Frenet frame and from this
    also the Bishop frame of this parabola analytically (note that it's not arc length parameterized!):
    r = r0 + s t0 + s^2/2 k0
    Frenet frame:
    => kappa(s) = kappa0 / (1+(s*kappa0)^2)^(3/2), with kappa0 = sqrt(k1^2 + k2^2)
        | Define norm = 1/sqrt(1+(s*kappa0)^2)
        t (s) = norm * ( t0  + s * k0 )
        n(s) = norm * ( k0/kappa0 - s * kappa0 * t0  )
        b(s) = b0
    Bishop frame:
    using n1 =  cos(phi) n + sin(phi) b
          n2 = -sin(phi) n + cos(phi) b
          <=>
          n = cos(phi) n1 - sin(phi) n2
          with cos(phi) = k1/kappa; sin(phi) = -k2/kappa

          (Note that for a parabola the torsion tau = 0 and therefore dphi/ds = -tau = 0 => phi = const.)

        n1(s) = cos(phi) * norm * ( n - s * kappa0 * t0 )
               +sin(phi) * b

    :param tripod: [Nx4x3] 4: (r0, eT=t0, eN1=n10, eN2=n20), r is position vector, eT, eN1, eN2 are bishop vectors (all uniformed)
    :param k: [Nx2] 2:(k1, k2) Bishop curvatures
    :param vmin: minimum step size (in units of base_step)
    :param speed_factor: factor on base_step
    :param base_step: basis step size if k = 0
    :return: tripod(0 + v)
    """
    tripod = tripod.astype('float64')
    k = k.astype('float64')

    # pre calculations and reshapes
    r, eT, eN1, eN2 = np.hsplit(tripod, 4)
    k1, k2 = np.hsplit(k, 2) # bishop curvature vector (projection of k)
    kappa_sq = k1 ** 2 + k2 ** 2
    kappa = np.sqrt(kappa_sq) # curvature

    # Compute Frenet from Bishop frame and curvatures
    # TODO: check if below formulas are
    #  * correct (check e.g. whether below formulas yield the same up to +O(v^2) as Euler step) AND
    #  * do not contain analytically reducible computations AND
    #  * numerically stable - what happens for kappa close or equal to 0?

    phi = np.arccos(np.clip(k1 / kappa, a_min=-1, a_max=1))
    cosphi = np.cos(phi).reshape(-1, 1, 1)
    sinphi = np.sin(phi).reshape(-1, 1, 1)
    eN = cosphi * eN1 - sinphi * eN2
    eN /= np.sqrt(np.sum(eN ** 2, axis=-1)).reshape(-1, 1, 1)
    eB = np.cross(eT, eN, axis=-1)


    v = stepsize_factor * base_step / (1.0 + 500 * kappa)
    v = np.maximum(vmin * base_step, v).reshape([-1, 1, 1])
    v_sq = v**2

    k1 = k1.reshape([-1, 1, 1])
    k2 = k2.reshape([-1, 1, 1])
    kappa = kappa.reshape([-1, 1, 1])
    kappa_sq = kappa_sq.reshape([-1, 1, 1])

    k = k1 * eN1 + k2 * eN2
    norm = 1 / np.sqrt(1 + v_sq * kappa_sq)

    # "Parabola step" - simultaneous updates
    dr = v * eT + v_sq / 2.0 * k
    r, eT, eN1 = (
            r + dr,
            norm * (eT + v * k),
            cosphi * norm * (eN - v * kappa * eT) + sinphi * eB
    )

    act_step_size = np.sqrt(np.sum(dr**2, axis=-1)).reshape([-1])

    # Apply Gram-Schmidt Orthonormalization
    eT, eN1, eN2 = gram_schmidt_orth(eT, eN1)

    return np.concatenate((r, eT, eN1, eN2), 1), act_step_size


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
    
    # visalize_frames(trajs[10:20],image)
    '''
    get r,t,n1,n2,k
    '''
    traj = trajs[18]
    wp = np.array(traj)
    tarj_length = len(wp)
    r, rmf_t, rmf_n1, rmf_n2, frenet_N, curvature = RMF(wp,sample_num=tarj_length*5)
    k = np.multiply(frenet_N,curvature)

    k1 = rmf_n1*(einsum(k,rmf_n1,'i j, i j -> i')/reduce(rmf_n1**2, 'i j->i', 'sum')**0.5)[:,np.newaxis]
    k2 = rmf_n2*(einsum(k,rmf_n2,'i j, i j -> i')/reduce(rmf_n2**2, 'i j->i', 'sum')**0.5)[:,np.newaxis]

    k1 = einsum(k1**2, 'i j -> i')**0.5
    k2 = einsum(k2**2, 'i j -> i')**0.5
    pred_k = np.stack((k1, k2), axis=1) 

    fi = 30 #frame index
    nr, nt, nn1, nn2 = predict_next_frame(r[fi],(rmf_t[fi],rmf_n1[fi],rmf_n2[fi]),pred_k[fi])

    r = r[0:fi][::5]
    t = rmf_t[0:fi][::5]
    n1 = rmf_n1[0:fi][::5]
    n2 = rmf_n2[0:fi][::5]

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
    size = [i+40 for i in size]
    roi = offset + size
    print(roi)
    img = image.from_roi(roi)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img,translate=offset)
    viewer.add_vectors(rmf_tv, edge_width=0.1, length=2, vector_style='arrow', edge_color='blue', name='Tangent')
    viewer.add_vectors(rmf_n1v, edge_width=0.1, length=2, vector_style='arrow', edge_color='red', name='N1')
    viewer.add_vectors(rmf_n2v, edge_width=0.1, length=2, vector_style='arrow', edge_color='orange', name='N2')
    napari.run()