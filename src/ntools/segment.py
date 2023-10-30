import torch
import torch.nn as nn
import torch.nn.functional as F

class Erosion(nn.Module):
    def __init__(self):
        super(Erosion, self).__init__()
        self.p1 = nn.MaxPool3d((3, 1, 1), stride=(1,1,1), padding=(1,0,0))
        self.p2 = nn.MaxPool3d((1, 3, 1), stride=(1,1,1), padding=(0,1,0))
        self.p3 = nn.MaxPool3d((1, 1, 3), stride=(1,1,1), padding=(0,0,1))

    def forward(self, x):
        x1 = -self.p1(-x)
        x2 = -self.p2(-x)
        x3 = -self.p3(-x)
        return torch.min(torch.min(x1, x2), x3)


class SKEL_Net(nn.Module):
    def __init__(self):
        super(SKEL_Net, self).__init__()
        self.dilate = nn.MaxPool3d((5, 5, 5), stride=(1,1,1), padding=(2,2,2))
        self.open = nn.Sequential(
            Erosion(),
            nn.MaxPool3d((3, 3, 3), stride=(1,1,1), padding=(1,1,1)),
            nn.MaxPool3d((5, 1, 1), stride=(1,1,1), padding=(2,0,0)),
            nn.MaxPool3d((1, 1, 5), stride=(1,1,1), padding=(0,0,2)),
        )

    def forward(self, img):
        img1 = self.open(img)
        skel = F.relu(img-img1)
        return skel



class Seger():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SKEL_Net().to(self.device)
    def get_fg(self,img,thres=80):
        img = torch.from_numpy(img.astype(float))
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        out = self.model(img)
        out = out.cpu().numpy()[0,0,:,:,:]
        out = out>=thres
        return out



def move_points(intensity_field, points, num_iterations, mode = 'max'):
    for _ in range(num_iterations):
        new_points = []

        for point in points:
            x, y, z = point

            neighbors = intensity_field[max(0, x - 1): min(intensity_field.shape[0], x + 2), max(0, y - 1): min(intensity_field.shape[1], y + 2), max(0, z - 1): min(intensity_field.shape[2], z + 2)]

            # Find the voxel with the highest intensity among the neighbors
            if mode == 'max':
                max_intensity_voxel = np.unravel_index(np.argmax(neighbors), neighbors.shape)
            else:
                max_intensity_voxel = np.unravel_index(np.argmin(neighbors), neighbors.shape)

            # Move the point to the voxel with the highest intensity
            new_x = max(0, min(intensity_field.shape[0] - 1, x + max_intensity_voxel[0] - 1))
            new_y = max(0, min(intensity_field.shape[1] - 1, y + max_intensity_voxel[1] - 1))
            new_z = max(0, min(intensity_field.shape[2] - 1, z + max_intensity_voxel[2] - 1))

            new_point = np.array([new_x, new_y, new_z])
            new_points.append(new_point)

        points = np.array(new_points)

    return points


def cal_inter_sphere(r1, r2, xyz1, xyz2):

    d = np.float32(cdist(xyz1.reshape(-1,3), xyz2.reshape(-1,3)).reshape(-1))

    # to check: r1 + r2 => d. If not, set r2=0 and d=r1
    indx = ~np.logical_and(abs(r1-r2)<d, d<(r1+r2))
    r2[indx] = 0
    d[indx] = r1

    inter_vol = (np.pi*(r1+r2-d)**2*(d**2+2*d*(r1+r2)-3*(r1**2+r2**2)+6*r1*r2)) / (12*d)

    return inter_vol, (~indx).sum()


def point_nms(obj_score, xyz, radius, overlap_threshold=0.25):
    radius = np.ones(shape=obj_score.shape)*radius
    vol = (4.0/3.0) * np.pi * (radius**3)
    I = np.argsort(obj_score.squeeze())
    pick = []

    # dict_num_pts_deleted = {}
    while (I.size!=0):
        last = I.size
        i = I[-1]
        # calculate IOU of two intersected spheres
        # https://math.stackexchange.com/questions/2705706/volume-of-the-overlap-between-2-spheres
        r1, xyz1 = radius[i], xyz[i,:]
        r2, xyz2 = radius[I[:last-1]], xyz[I[:last-1],:]
        inter, numInteract = cal_inter_sphere(r1, r2, xyz1, xyz2)
        o = inter / (vol[i] + vol[I[:last-1]] - inter)

        pts_deleted = np.concatenate(([last-1], np.where(o>overlap_threshold)[0]))
        # dict_num_pts_deleted[obj_score[i]] = numInteract
        I = np.delete(I, pts_deleted)

        if numInteract > 0:
            pick.append(i)

    return np.array(pick)


def get_points(image,seger,roi,move_steps=3):
    mask = seger.get_fg(image,thres=100)
    label_image, num_of_labels = measure.label(mask,return_num=True)
    label_image = label_image.astype(bool)
    mask = morphology.remove_small_objects(label_image, min_size=3, connectivity=3)
    mask = mask >= 1
    masked_image = image*mask
    points = peak_local_max(masked_image, min_distance=1.5, footprint=ball(1), exclude_border=1)
    points = move_points(image,points,move_steps)
    points = np.unique(points, axis=0)
    if len(points)==0:
        intensity = np.array([])
    else:
        intensity = image[points[:,0],points[:,1],points[:,2]] 
    # TODO calculate radius
    # sampled_points = point_nms(intensity,points,6,overlap_threshold=0.28)
    # all_points = [i for i in range(len(points))]
    # sampled_out = list(set(all_points) - set(sampled_points))
    offset = roi[0:3]
    points = np.array([[coord[0]+offset[0],coord[1]+offset[1],coord[2]+offset[2]] for coord in points])
    # return points[sampled_points], points[sampled_out]
    return points.tolist(), intensity.tolist()

