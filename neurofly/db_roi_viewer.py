import os
import random
import numpy as np
import networkx as nx
from rtree import index

import napari
from napari_plugin_engine import napari_hook_implementation
from magicgui import widgets
from napari.utils.notifications import show_info

from neurofly.dbio import read_nodes, read_edges

class DbROIViewer(widgets.Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self.G = None
        self.rtree = None
        self.db_loaded = False

        # Points Layer显示ROI点，初始为空
        self.points_layer = self.viewer.add_points(
            data=np.empty((0, 3)),
            ndim=3,
            name="db_points",
            face_color="gray",
            size=5,
            blending="additive",
            visible=True
        )

        # 控件定义
        self.db_path = widgets.FileEdit(label="DB Path", filter="*.db")
        self.load_db_button = widgets.PushButton(text="Load DB")

        # ROI控件
        self.x_offset = widgets.LineEdit(label="X offset", value="0")
        self.y_offset = widgets.LineEdit(label="Y offset", value="0")
        self.z_offset = widgets.LineEdit(label="Z offset", value="0")
        self.x_size = widgets.LineEdit(label="X size", value="100000")
        self.y_size = widgets.LineEdit(label="Y size", value="100000")
        self.z_size = widgets.LineEdit(label="Z size", value="100000")

        self.roi_view_button = widgets.PushButton(text="ROI View")

        # 控件加入容器
        self.extend([
            self.db_path,
            self.load_db_button,
            self.x_offset,
            self.y_offset,
            self.z_offset,
            self.x_size,
            self.y_size,
            self.z_size,
            self.roi_view_button,
        ])

        # 事件绑定
        self.load_db_button.clicked.connect(self.load_database)
        self.roi_view_button.clicked.connect(self.refresh_points)

    def load_database(self):
        db_path_str = str(self.db_path.value)
        if not os.path.exists(db_path_str):
            show_info(f"Database path not found: {db_path_str}")
            self.db_loaded = False
            self.points_layer.data = np.empty((0, 3))
            return

        nodes = read_nodes(db_path_str)
        edges = read_edges(db_path_str)

        self.G = nx.Graph()
        rtree_data = []
        for node in nodes:
            nid, coord = node["nid"], node["coord"]
            self.G.add_node(nid, **node)
            rtree_data.append((nid, tuple(coord + coord), None))

        for e in edges:
            self.G.add_edge(e["src"], e["des"], **e)

        p = index.Property(dimension=3)
        self.rtree = index.Index(rtree_data, properties=p)
        self.db_loaded = True

        show_info("Database loaded successfully.")
        self.refresh_points()

    def refresh_points(self):
        if not self.db_loaded or self.G is None:
            show_info("Database not loaded or Graph is None.")
            return

        roi_offset = (
            int(float(self.x_offset.value)),
            int(float(self.y_offset.value)),
            int(float(self.z_offset.value))
        )
        roi_size = (
            int(float(self.x_size.value)),
            int(float(self.y_size.value)),
            int(float(self.z_size.value))
        )

        roi_min = roi_offset
        roi_max = tuple(o + s for o, s in zip(roi_offset, roi_size))
        query_box = (*roi_min, *roi_max)

        node_ids_in_roi = list(self.rtree.intersection(query_box, objects=False))
        coords, colors, nids = [], [], []

        for nid in node_ids_in_roi:
            coord = self.G.nodes[nid]["coord"]
            if all(mn <= c < mx for mn, c, mx in zip(roi_min, coord, roi_max)):
                adjusted_coord = [c - offset for c, offset in zip(coord, roi_offset)]
                coords.append(adjusted_coord)
                colors.append(random.random())
                nids.append(nid)

        if not coords:
            show_info("No nodes found after ROI filtering.")
            self.points_layer.data = np.empty((0, 3))
            return

        coords_arr = np.array(coords, dtype=float)
        properties = {"colors": np.array(colors), "nids": np.array(nids)}

        self.points_layer.data = coords_arr
        self.points_layer.properties = properties
        self.points_layer.face_color = "colors"
        self.points_layer.face_colormap = "hsl"
        self.points_layer.color_mode = "colormap"
        self.points_layer.face_contrast_limits = [0, 1]
        self.points_layer.size = 5

        self.viewer.reset_view()

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [DbROIViewer]
