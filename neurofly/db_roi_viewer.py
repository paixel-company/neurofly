import numpy as np
import napari
import networkx as nx
from rtree import index
from magicgui import widgets
from napari.utils.notifications import show_info
import random

from neurofly.dbio import read_nodes, read_edges  # 假定你已有这些函数

class DbROIViewer(widgets.Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')

        self.G = None
        self.rtree = None
        self.db_loaded = False

        # 一个 Points layer 用于显示节点
        self.points_layer = self.viewer.add_points(
            data=None,
            ndim=3,
            name="db_points",
            face_color="gray",
            size=3,
            blending="additive",
            visible=True
        )

        # 控件
        self.db_path = widgets.FileEdit(label="DB Path", filter="*.db")
        self.load_db_button = widgets.PushButton(text="Load DB")

        self.min_length = widgets.Slider(label="min length", value=10, min=0, max=1000)
        self.refresh_button = widgets.PushButton(text="Refresh Points")

        self.extend([
            self.db_path,
            self.load_db_button,
            self.min_length,
            self.refresh_button
        ])

        self.load_db_button.clicked.connect(self.load_database)
        self.refresh_button.clicked.connect(self.refresh_points)

    def load_database(self):
        """从数据库读取并构建 networkx.Graph, rtree."""
        db_path_str = str(self.db_path.value)
        if not db_path_str:
            show_info("No DB file selected.")
            return

        print("[DEBUG] Loading database from:", db_path_str)

        # 先读节点和边
        nodes = read_nodes(db_path_str)
        edges = read_edges(db_path_str)
        print(f"[DEBUG] read_nodes returns {len(nodes)} nodes")
        print(f"[DEBUG] read_edges returns {len(edges)} edges")

        if not nodes:
            show_info("No nodes found in the database.")
            self.G = None
            self.db_loaded = False
            self.points_layer.data = np.empty((0, 3))
            return

        # 构建图
        self.G = nx.Graph()
        rtree_data = []
        for node in nodes:
            nid = node["nid"]
            coord = node["coord"]
            self.G.add_node(nid, **node)
            rtree_data.append((nid, tuple(coord + coord), None))
        for e in edges:
            self.G.add_edge(e["src"], e["des"], **e)

        p = index.Property(dimension=3)
        self.rtree = index.Index(rtree_data, properties=p)
        self.db_loaded = True
        show_info("DB loaded successfully.")

        print(f"[DEBUG] Graph has {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        self.refresh_points()

    def refresh_points(self):
        """显示数据库里的节点，根据连通分量大小进行简单筛选。"""
        if not self.db_loaded or self.G is None:
            show_info("DB not loaded or Graph is None.")
            return

        connected_components = list(nx.connected_components(self.G))
        print(f"[DEBUG] Found {len(connected_components)} connected components in the Graph.")

        coords = []
        colors = []
        nids_list = []
        length_threshold = self.min_length.value

        for i, cc in enumerate(connected_components):
            size_cc = len(cc)
            print(f"[DEBUG] Component #{i} size = {size_cc}")
            if size_cc < length_threshold:
                print(f"[DEBUG]   -> Skip (below min_length={length_threshold})")
                continue

            color_val = random.random()
            for nid in cc:
                if not self.G.has_node(nid):
                    continue
                node_data = self.G.nodes[nid]
                coord = node_data["coord"]
                coords.append(coord)
                nids_list.append(nid)
                colors.append(color_val)

        if not coords:
            show_info("No connected component meets the min_length threshold.")
            self.points_layer.data = np.empty((0, 3))
            return

        # 打印一下最终要显示多少点
        print(f"[DEBUG] About to display {len(coords)} points in db_points layer.")

        coords_arr = np.array(coords, dtype=float)
        colors_arr = np.array(colors, dtype=float)
        properties = {"colors": colors_arr, "nids": np.array(nids_list)}

        self.points_layer.data = coords_arr
        self.points_layer.properties = properties
        self.points_layer.face_color = "colors"
        self.points_layer.face_colormap = "hsl"
        self.points_layer.color_mode = "colormap"
        self.points_layer.face_contrast_limits = [0, 1]
        self.points_layer.size = 5

        # 重置视图
        self.viewer.reset_view()
        self.viewer.layers.selection.active = self.points_layer
        print("[DEBUG] Points updated & viewer reset.")