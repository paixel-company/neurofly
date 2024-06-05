import numpy as np
import random
from napari.qt.threading import create_worker, thread_worker
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline as R_spline
from scipy.interpolate import CubicSpline, interp1d
import math
import imageio
import time


class Recorder:
    def __init__(self, viewer = None):
        if viewer is None:
            raise TypeError('You need to pass a napari viewer for the myviewer argument')
        else:
            self.viewer = viewer
        
        self.camera_trajectory = []
        #establish key bindings
        self.add_callback()


    def add_callback(self):
        self.viewer.bind_key('r', self.add_frame)
        self.viewer.bind_key('t', self.play_back)
        self.viewer.bind_key('v', self.save_video)
        self.viewer.bind_key('u', self.del_frame)


    def add_frame(self,viewer):
        camera_cfg =self.viewer.camera.dict()
        self.camera_trajectory.append(camera_cfg)
        print('Added camera trajectory point')
        print(camera_cfg)
        self.viewer.status = 'Added camera trajectory point'


    def interp_camera_tra(self,ct,t_slice):
        angles = []
        zooms = []
        centers = []
        for frame in ct:
            angles.append(frame['angles'])
            zooms.append(frame['zoom'])
            centers.append(frame['center'])
        key_angles = R.from_euler('xyz', angles, degrees=True)
        key_times = np.arange(len(key_angles))
        times = np.arange(0,len(key_angles)-1,t_slice)
        #spherical interpolation
        spline = R_spline(key_times,key_angles)
        interp_angles = spline(times).as_euler('xyz', degrees=True)
        #zoom interpolation
        interp_type = 'cubic'
        f_zoom = interp1d(key_times,zooms,kind=interp_type)
        interp_zooms = f_zoom(times)
        #center interpolation
        centers = np.asarray(centers)
        x = centers[:,0].tolist()
        y = centers[:,1].tolist()
        z = centers[:,2].tolist()
        f_x = interp1d(key_times,x,interp_type)
        f_y = interp1d(key_times,y,interp_type)
        f_z = interp1d(key_times,z,interp_type)
        interp_x = f_x(times)
        interp_y = f_y(times)
        interp_z = f_z(times)
        interp_centers = np.array([interp_x,interp_y,interp_z]).transpose()

        return interp_angles,interp_zooms,interp_centers


    def play_back(self,viewer):
        ct = self.camera_trajectory.copy()
        interp_angles,interp_zooms,interp_centers = self.interp_camera_tra(ct,t_slice=0.04)
        camera_args = []
        args_temp = self.camera_trajectory[0]
        for (angel,zoom,center) in zip(interp_angles,interp_zooms,interp_centers):
            args = args_temp.copy()
            args['angles'] = angel.tolist()
            args['zoom'] = zoom
            args['center'] = tuple(center)
            camera_args.append(args)
        self.play_back_trajectory(camera_args)



    def save_video(self,viewer):
        ct = self.camera_trajectory.copy()
        interp_angles,interp_zooms,interp_centers = self.interp_camera_tra(ct,t_slice=0.02)
        camera_args = []
        args_temp = self.camera_trajectory[0]
        frames = []
        writer = imageio.get_writer('movie.mp4', fps=24)
        for (angel,zoom,center) in zip(interp_angles,interp_zooms,interp_centers):
            args = args_temp.copy()
            args['angles'] = angel.tolist()
            args['zoom'] = zoom
            args['center'] = tuple(center)
            self.viewer.camera.update(values=args)
            frames.append(viewer.screenshot(size=(1024,1024)))

        for im in frames:
            writer.append_data(im)
        writer.close()



    def del_frame(self,viewer):
        if(len(self.camera_trajectory)>0):
            self.camera_trajectory.pop()
            print('deleted key frame')
            self.viewer.status = 'deleted key frame'
        else:
            print('no key frame left')
            self.viewer.status = 'no key frame left'



    def update_camera(args):
        # self.viewer.camera.update(values=args)
        pass


    @thread_worker(connect={'yielded': update_camera})
    def play_back_trajectory(self,camera_args):
        for args in camera_args:
            self.viewer.camera.update(values=args)
            yield args
            time.sleep(0.02)

