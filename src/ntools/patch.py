import numpy as np





def get_patch_coords(roi,block_size):
    volume_size = roi[3:6]
    origin = roi[0:3]
    grid_count = [i//block_size if i%block_size==0 else i//block_size+1 for i in volume_size]
    hist = np.zeros(grid_count, np.uint16)
    indices = np.where(hist==0)
    indices = np.array(indices).transpose()*block_size
    indices = indices[indices[:,2].argsort()]
    return indices


def get_patch_rois(roi,block_size):
    volume_size = roi[3:6]
    origin = roi[0:3]
    upper_bound = [i+j for i,j in zip(origin,volume_size)]
    block_coords = get_patch_coords(roi,block_size)
    rois = []
    for coord in block_coords:
        c1 = [i+j for i,j in zip(coord,origin)]
        c2 = [i+block_size if i+block_size<j else j for i,j in zip(c1,upper_bound)]
        size = [j-i for i,j in zip(c1,c2)]
        rois.append(c1+size)
    return rois



def get_patch_by_density(roi,block_size,segs):
    volume_size = roi[3:]
    offset = roi[0:3]
    patch_coords = get_patch_coords(roi,block_size)
    points = []
    segs = sum(segs,[])
    for seg in segs:
        for node in seg:
            points.append([i-j for i,j in zip(node['pos'],offset)])
    points = np.array(points)

    grid_count = np.array(volume_size)//block_size
    hist = np.zeros(grid_count, np.uint16)
    points = np.floor(points/block_size)
    points = points.astype(int)
    inboundx = np.less(points[:,0],grid_count[0]-1)
    inboundy = np.less(points[:,1],grid_count[1]-1) 
    inboundz = np.less(points[:,2],grid_count[2]-1) 
    inbound = np.logical_and(inboundx,inboundy)
    inbound = np.logical_and(inbound,inboundz)

    for point in points[inbound,:]:
        hist[point[0]][point[1]][point[2]] += 1

    sorted_indices = np.argsort(hist, axis=None)
    sorted_indices = sorted_indices[::-1]
    sorted_coordinates = np.unravel_index(sorted_indices, hist.shape)
    sorted_values = hist[sorted_coordinates]
    sorted_coordinates = np.array(sorted_coordinates).transpose()
    block_coords = sorted_coordinates*block_size+np.array(offset)

    return block_coords.tolist()


def patchify_without_splices(roi,patch_size,splices=300):
    rois = []
    xs = list(range(roi[0],roi[0]+roi[3],patch_size[0]))
    # if (roi[0]+roi[3])%patch_size[0]!=0:
    xs.append(roi[0]+roi[3])

    ys = list(range(roi[1],roi[1]+roi[4],patch_size[1]))
    # if (roi[1]+roi[4])%patch_size[1]!=0:
    ys.append(roi[1]+roi[4])

    zs = [z for z in range(roi[2],roi[2]+roi[5]) if z%splices==0]
    if roi[2]%splices!=0:
        zs.insert(0,roi[2])
    # if (roi[2]+roi[5])%splices!=0:
    zs.append(roi[2]+roi[5])

    for x1,x2 in zip(xs[:-1],xs[1:]):
        for y1,y2 in zip(ys[:-1],ys[1:]):
            for z1,z2 in zip(zs[:-1],zs[1:]):
                rois.append([x1,y1,z1,x2-x1,y2-y1,z2-z1])
    return rois


