import argparse
import numpy as np
import networkx as nx
import napari
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from tqdm import tqdm
from magicgui import widgets
from neurofly.patch import patchify_without_splices, get_patch_rois
from neurofly.dbio import segs2db
from neurofly.image_reader import wrap_image
from neurofly.vis import show_segs_as_instances, show_segs_as_paths, draw_frame
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from neurofly.common import default_dec_weight_path, default_seger_weight_path
from neurofly.common import SegNet, Deconver




class Seger():
    def __init__(self,ckpt_path,bg_thres=200):
        self.seg_net = SegNet(ckpt_path,bg_thres)
        # border width
        self.bw = 4


    def postprocess(self,mask,min_size=50):
        labeled_mask, _ = label(mask,return_num=True)
        region_sizes = np.bincount(labeled_mask.ravel())
        small_regions = np.where(region_sizes < min_size)[0]
        for region in small_regions:
            mask[labeled_mask == region] = 0
        return mask
    

    def get_large_mask(self,img,dec=None):
        '''
        process one large volume(D,W,H>100) with border (default 4), return mask
        '''
        block_size = 100
        border_size = self.bw
        bordered_size = img.shape
        actual_size = [i-border_size*2 for i in bordered_size]
        block_rois = get_patch_rois([border_size,border_size,border_size]+actual_size,block_size)
        large_mask = np.zeros(img.shape,dtype=np.uint8)
        for roi in block_rois:
            tg_size = self.bw
            # add border if possible
            x1,x2,y1,y2,z1,z2 = roi[0],roi[0]+roi[3],roi[1],roi[1]+roi[4],roi[2],roi[2]+roi[5]
            x1 = max(0,x1-tg_size)
            y1 = max(0,y1-tg_size)
            z1 = max(0,z1-tg_size)
            x2 = min(img.shape[0],x2+tg_size)
            y2 = min(img.shape[1],y2+tg_size)
            z2 = min(img.shape[2],z2+tg_size)

            block = img[x1:x2,y1:y2,z1:z2]

            x1_pad = roi[0]-x1
            y1_pad = roi[1]-y1
            z1_pad = roi[2]-z1
            x2_pad = x2-roi[0]-roi[3]
            y2_pad = y2-roi[1]-roi[4]
            z2_pad = z2-roi[2]-roi[5]

            pad_widths = [
                (tg_size-x1_pad, tg_size-x2_pad),
                (tg_size-y1_pad, tg_size-y2_pad),
                (tg_size-z1_pad, tg_size-z2_pad)
            ]
            
            # if img.shape%block_size != 0, pad to target size
            ap = [] # additional padding
            for i, (p1,p2) in enumerate(pad_widths):
                res = block_size+tg_size*2 - (block.shape[i]+p1+p2)
                ap.append(res)
                if res!=0:
                    pad_widths[i] = (p1,p2+res)

            padded_block = np.pad(block, pad_widths, mode='reflect')
            if dec is not None:
                padded_block = dec.process_one(padded_block)

            mask = self.seg_net.get_mask(padded_block,thres=0.5)
            mask = mask.astype(np.uint8)
            mask = mask[tg_size:-tg_size-ap[0],tg_size:-tg_size-ap[1],tg_size:-tg_size-ap[2]]
            large_mask[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[4],roi[2]:roi[2]+roi[5]] = mask
        processed_mask = self.postprocess(large_mask)
        return processed_mask[border_size:-border_size,border_size:-border_size,border_size:-border_size]


    def mask_to_segs(self, mask, offset=[0,0,0]):
        '''
        segment:
        {
            sid: int,
            points: [head,...,tail],
            sampled_points: points[::interval]
        }
        '''

        interval = 3

        x_border = 1
        y_border = 1
        z_border = 1

        skel = skeletonize(mask)
        skel[:x_border, :, :] = 0
        skel[-x_border:, :, :] = 0
        skel[:, :y_border, :] = 0
        skel[:, -y_border:, :] = 0
        skel[:, :, :z_border] = 0
        skel[:, :, -z_border:] = 0

        labels = label(skel, connectivity=3)
        regions = regionprops(labels)

        segments = []
        for region in regions:
            points = region.coords
            distances = cdist(points, points)
            adjacency_matrix = distances <= 1.8 # sqrt(3)
            np.fill_diagonal(adjacency_matrix, 0)
            graph = nx.from_numpy_array(adjacency_matrix.astype(np.uint8))
            spanning_tree = nx.minimum_spanning_tree(graph, algorithm='kruskal', weight=None)
            # remove circles by keeping only DFS tree
            graph.remove_edges_from(set(graph.edges) - set(spanning_tree.edges))

            branch_nodes = [node for node, degree in graph.degree() if degree >= 3]
            branch_nbrs = []
            for node in branch_nodes:
                branch_nbrs += list(graph.neighbors(node))

            for bn in branch_nodes:
                if len(list(graph.neighbors(node)))==3:
                    segments.append(
                        {
                            'sid' : None,
                            'points' : [[i+j for i,j in zip(points[bn],offset)]],
                            'sampled_points' : [[i+j for i,j in zip(points[bn],offset)]]
                        }
                    )

            graph.remove_nodes_from(branch_nbrs)
            graph.remove_nodes_from(branch_nodes)

            connected_components = list(nx.connected_components(graph))

            for nodes in connected_components:
                if len(nodes)<=interval*2:
                    continue
                subgraph = graph.subgraph(nodes).copy()
                end_nodes = [node for node, degree in subgraph.degree() if degree == 1]
                if (len(end_nodes)!=2):
                    continue
                path = nx.shortest_path(subgraph, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
                # path to segment
                seg_points = np.array([points[i].tolist() for i in path])
                seg_points = seg_points + np.array(offset)
                seg_points = seg_points.tolist()
                sampled_points = seg_points[:-(interval-1):interval]
                sampled_points.append(seg_points[-1])
                segments.append(
                    {
                        'sid' : None,
                        'points' : seg_points,
                        'sampled_points' : sampled_points
                    }
                )
        return skel, segments



    def process_whole(self,image_path,channel=0,chunk_size=300,splice=100000,roi=None,dec=None):
        '''
        cut whole brain image to [300,300,300] cubes without splices (z coordinates % 300 == 0)
        '''
        image = wrap_image(image_path)
        if roi==None:
            image_roi = image.roi
        else:
            image_roi = roi
        rois = patchify_without_splices(image_roi,chunk_size,splices=splice)
        # pad rois
        segs = []
        for roi in tqdm(rois):
            if (np.array(roi[3:])<=np.array([128,128,128])).all():
                if 'tif' in image_path:
                    img = image.from_roi(roi,padding='reflect')
                else:
                    img = image.from_roi(roi,0,channel,padding='reflect') 
                if dec is not None:
                    img = dec.process_one(img)
                mask = self.seg_net.get_mask(img)
                offset = roi[:3]
            else:
                roi[:3] = [i-self.bw for i in roi[:3]]
                roi[3:] = [i+self.bw*2 for i in roi[3:]]
                if 'tif' in image_path:
                    padded_block = image.from_roi(roi,padding='reflect')
                else:
                    padded_block = image.from_roi(roi,0,channel,padding='reflect') 
                mask = self.get_large_mask(padded_block,dec)
                offset=[i+self.bw for i in roi[:3]]
            _, segs_in_block = self.mask_to_segs(mask,offset=offset)
            segs+=segs_in_block

        for i, seg in enumerate(segs):
            seg['sid'] = i

        return segs



class SegerGUI(widgets.Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.dims.ndisplay = 3
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')
        self.seger = None
        self.deconver = None
        self.add_callback()

    def add_callback(self):
        self.image_type = widgets.CheckBox(value=False,text='read zarr format')
        self.image_path = widgets.FileEdit(label="image_path")
        self.save_dir = widgets.FileEdit(label="save dir",mode='w')
        self.x_size = widgets.Slider(label="x size", value=0, min=0, max=100000)
        self.y_size = widgets.Slider(label="y size", value=0, min=0, max=100000)
        self.z_size = widgets.Slider(label="z size", value=0, min=0, max=100000)
        self.x = widgets.LineEdit(label="x offset", value=0)
        self.y = widgets.LineEdit(label="y offset", value=0)
        self.z = widgets.LineEdit(label="z offset", value=0)
        self.channel = widgets.LineEdit(label="image channel", value=0)
        self.chunk_size = widgets.LineEdit(label="chunk size", value=300)
        self.bg_thres = widgets.LineEdit(label="background value", value=300)
        self.splices = widgets.LineEdit(label="z-splice", value=100000) 
        self.run_button = widgets.PushButton(text="run segmentation")
        self.run_button.clicked.connect(self.start_segmentation)
        self.seger_weight_path = widgets.FileEdit(label="seger weight")
        self.deconver_weight_path = widgets.FileEdit(label="deconver weight")
        self.use_deconv = widgets.CheckBox(value=False,text='deconvolve before segment')
        self.progress_bar = widgets.ProgressBar(min=0, max=100, value=0)
        self.show_progress = widgets.CheckBox(value=False,text="show progress")

        seger_weight_path = default_seger_weight_path
        deconver_weight_path = default_dec_weight_path
        self.image_path.changed.connect(self.on_image_reading)
        self.image_type.changed.connect(self.switch_image_type)
        self.seger_weight_path.changed.connect(self.load_seger)
        self.deconver_weight_path.changed.connect(self.load_deconver)
        self.seger_weight_path.value = seger_weight_path
        self.deconver_weight_path.value = deconver_weight_path


        self.extend([
            self.image_type,
            self.image_path,
            self.seger_weight_path,
            self.deconver_weight_path,
            self.save_dir,
            self.x_size,
            self.y_size,
            self.z_size,
            self.x,
            self.y,
            self.z,
            self.splices,
            self.channel,
            self.bg_thres,
            self.chunk_size,
            self.use_deconv,
            self.show_progress,
            self.run_button,
            self.progress_bar
            ])


    def switch_image_type(self,event):
        if event:
            self.image_path.mode = 'd'
        else:
            self.image_path.mode = 'r'


    def on_image_reading(self):
        self.image = wrap_image(str(self.image_path.value))
        x_offset,y_offset,z_offset,x_size,y_size,z_size = self.image.roi
        self.x.value = x_offset
        self.y.value = y_offset
        self.z.value = z_offset
        self.x_size.max = x_size
        self.y_size.max = y_size
        self.z_size.max = z_size
        self.x_size.value = x_size
        self.y_size.value = y_size
        self.z_size.value = z_size
        draw_frame(self.image.roi, self.viewer, color='white')


    def load_deconver(self):
        self.deconver = Deconver(str(self.deconver_weight_path.value))
        show_info("Deconvolution model loaded")


    def load_seger(self):
        self.seger = Seger(str(self.seger_weight_path.value),int(self.bg_thres.value))
        show_info("Segmentation model loaded")


    def start_segmentation(self):
        # Disable the run button to prevent multiple clicks
        self.run_button.enabled = False

        # Start the segmentation in a background thread
        worker = self.process_whole()
        worker.yielded.connect(self.update_progress)
        worker.returned.connect(self.on_segmentation_finished)
        worker.finished.connect(self.on_task_finished)
        worker.start()


    def update_progress(self, value):
        (current, whole, segs_in_block) = value
        self.progress_bar.max = whole
        self.progress_bar.value = current
        if self.show_progress.value == False:
            return
        points_in_block = []
        for seg in segs_in_block:
            points_in_block += seg['sampled_points']
        if len(points_in_block) == 0:
            return
        points_array = np.array(points_in_block)
        if not hasattr(self, 'points_layer') or self.points_layer is None:
            self.points_layer = self.viewer.add_points(
                points_array,
                size=1,
                face_color='orange',
                name='neurite points'
            )
        else:
            existing_data = self.points_layer.data
            updated_data = np.concatenate((existing_data, points_array), axis=0)
            self.points_layer.data = updated_data


    def on_task_finished(self):
        self.run_button.enabled = True


    def on_segmentation_finished(self, segs):
        db_path = str(self.save_dir.value)
        if db_path != '.':
            show_info(f"Saving {len(segs)} segments to {db_path}")
            segs2db(segs, db_path)

        seg_points = [seg['sampled_points'] for seg in segs]
        show_segs_as_instances(seg_points, self.viewer)
        self.viewer.layers.remove(self.points_layer)
        self.points_layer = None


    @thread_worker()
    def process_whole(self):
        # Extract parameters from widgets
        roi = [
            int(self.x.value), int(self.y.value), int(self.z.value),
            int(self.x_size.value), int(self.y_size.value), int(self.z_size.value)
        ]
        image_path = str(self.image_path.value)
        splice = int(self.splices.value)
        chunk_size = int(self.chunk_size.value)
        if chunk_size > splice:
            show_info(f"can't use chunk size {chunk_size} for it is larger than splice {splice}")
            chunk_size = splice
        dec = self.deconver if self.use_deconv.value else None

        image = wrap_image(image_path)
        image_roi = roi if roi else image.roi

        rois = patchify_without_splices(image_roi, chunk_size, splices=splice)
        total_rois = len(rois)

        segs = []

        for idx, roi in enumerate(rois):
            if (np.array(roi[3:]) <= np.array([128, 128, 128])).all():
                if 'tif' in image_path:
                    img = image.from_roi(roi,padding='reflect')
                else:
                    img = image.from_roi(roi,0,int(self.channel.value),padding='reflect') 
                if dec is not None:
                    img = dec.process_one(img)
                mask = self.seger.seg_net.get_mask(img)
                offset = roi[:3]
            else:
                bw = self.seger.bw
                roi_padded = [
                    roi[0] - bw, roi[1] - bw, roi[2] - bw,
                    roi[3] + 2 * bw, roi[4] + 2 * bw, roi[5] + 2 * bw
                ]
                if 'tif' in image_path:
                    padded_block = image.from_roi(roi_padded,padding='reflect')
                else:
                    padded_block = image.from_roi(roi_padded,0,int(self.channel.value),padding='reflect') 
                mask = self.seger.get_large_mask(padded_block, dec)
                offset = [roi[0], roi[1], roi[2]]

            _, segs_in_block = self.seger.mask_to_segs(mask, offset=offset)
            segs += segs_in_block

            yield (idx+1, total_rois, segs_in_block)

        for i, seg in enumerate(segs):
            seg['sid'] = i

        return segs


def command_line_interface():
    parser = argparse.ArgumentParser(description="args for seger")
    parser.add_argument('--weight_path', '-w', type=str, default=None, help="path to weight of the segmentation model")
    parser.add_argument('--image_path', '-i', type=str, help="path to the input image, only zarr, ims, tif are currently supported")
    parser.add_argument('--db_path', '-d', type=str, default=None, help="path to the output database file")
    parser.add_argument('-roi', type=int, nargs='+', default=None, help="image roi, if kept None, process the whole image")
    parser.add_argument('-bg_thres', type=int, default=150, help="ignore images with maximum intensity smaller than this")
    parser.add_argument('-chunk_size', type=int, default=300, help="image size for skeletonization")
    parser.add_argument('-channel', type=int, default=0, help="channel index of ims image")
    parser.add_argument('-splice', type=int, default=300, help="set this value if your image contain  at certain interval on z axis")
    parser.add_argument('-vis', action='store_true', default=False, help="whether to visualize result after segmentation")
    parser.add_argument('-path', action='store_true', default=True, help="whether to visualize result as paths")
    parser.add_argument('-deconv', action='store_true', default=False, help="deconvolve image before segmentation")
    args = parser.parse_args()
    if args.weight_path is None:
        args.weight_path = default_seger_weight_path

    print(f"Using weight: {args.weight_path}")
    print(f"Processing image: {args.image_path}, roi: {args.roi}")

    # bg_thres is used to filter out empty image like image borders
    seger = Seger(args.weight_path,bg_thres=args.bg_thres) 
    if args.deconv:
        dec_weight_path = default_dec_weight_path
        deconver = Deconver(dec_weight_path)
    else:
        deconver = None

    segs = seger.process_whole(args.image_path, args.channel, chunk_size=args.chunk_size, splice=args.splice, roi=args.roi, dec=deconver)

    if args.db_path is not None:
        segs2db(segs,args.db_path)
        show_info(f"Saving {len(segs)} segs to {args.db_path}")


    if args.vis:
        import napari
        viewer = napari.Viewer(ndisplay=3)
        image = wrap_image(args.image_path)
        if args.roi is None:
            args.roi = image.roi
        if (np.array(args.roi[3:])<np.array([1024,1024,1024])).all():
            if 'tif' in args.image_path:
                img = image.from_roi(args.roi,padding='reflect')
            else:
                img = image.from_roi(args.roi,0,args.channel,padding='reflect') 
            image_layer = viewer.add_image(img)
            image_layer.translate = args.roi[0:3]
        else:
            print(f"image size {args.roi[3:]} is too large to render")
        seg_points = []
        for seg in segs:
            seg_points.append(seg['sampled_points'])
        if args.path:
            show_segs_as_paths(seg_points, viewer, width=0.3)
        else:
            show_segs_as_instances(seg_points, viewer)
        napari.run()


if __name__ == '__main__':
    command_line_interface()
