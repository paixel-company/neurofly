import os
import random
import numpy as np
import networkx as nx
from rtree import index

import napari
from napari_plugin_engine import napari_hook_implementation
from magicgui import widgets
from napari.utils.notifications import show_info

# 假定你在 neurofly.dbio 中有 read_nodes, read_edges 函数
from neurofly.dbio import read_nodes, read_edges

class DbROIViewer(widgets.Container):
    """
    DbROIViewer 插件：
      1. 从数据库读取节点和边，构建 networkx.Graph 与 rtree；
      2. 默认显示全部节点；
      3. 用户可在界面输入 ROI 参数（X offset, Y offset, Z offset, X size, Y size, Z size），
         点击 “ROI View” 按钮后，根据 ROI 进行筛选，只显示 ROI 内的节点；
      4. 使用属性 'colors'（随机数值）进行颜色映射，避免出现 “Color 'colors' unknown” 错误。
    """

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')

        self.G = None
        self.rtree = None
        self.db_loaded = False

        # 创建 Points Layer 用于显示节点（初始时为空）
        # 初始 face_color 设置为固定颜色以避免“colors”解析问题
        self.points_layer = self.viewer.add_points(
            data=np.empty((0, 3)),
            ndim=3,
            name="db_points",
            face_color="gray",
            size=5,
            blending="additive",
            visible=True
        )

        # ----------------- 界面控件 -----------------
        # 数据库路径和加载按钮
        self.db_path = widgets.FileEdit(label="DB Path", filter="*.db")
        self.load_db_button = widgets.PushButton(text="Load DB")

        # ROI 参数：使用 LineEdit 让用户输入，单位与数据库中坐标一致
        self.x_offset = widgets.LineEdit(label="X offset", value="0")
        self.y_offset = widgets.LineEdit(label="Y offset", value="0")
        self.z_offset = widgets.LineEdit(label="Z offset", value="0")
        self.x_size = widgets.LineEdit(label="X size", value="100000")  # 默认显示全部，可自行调整
        self.y_size = widgets.LineEdit(label="Y size", value="100000")
        self.z_size = widgets.LineEdit(label="Z size", value="100000")

        # ROI View 按钮，点击后按照 ROI 参数刷新显示
        self.roi_view_button = widgets.PushButton(text="ROI View")

        # 将控件加入容器
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
        self.db_path.changed.connect(self._on_db_path_changed)
        self.load_db_button.clicked.connect(self.load_database)
        self.roi_view_button.clicked.connect(self.refresh_points)

    def _on_db_path_changed(self):
        # 可选择自动加载数据库，此处留空
        pass

    def load_database(self):
        """
        从数据库中读取节点和边，构建 networkx.Graph 与 rtree 索引。
        加载后默认显示全部节点（即不进行 ROI 筛选）。
        """
        db_path_str = str(self.db_path.value)
        if not os.path.exists(db_path_str):
            show_info(f"Database path not found: {db_path_str}")
            self.G = None
            self.db_loaded = False
            self.points_layer.data = np.empty((0, 3))
            return

        nodes = read_nodes(db_path_str)
        edges = read_edges(db_path_str)
        print(f"[DEBUG] read_nodes: {len(nodes)} nodes, read_edges: {len(edges)} edges")
        if not nodes:
            show_info("No nodes found in the database.")
            self.G = None
            self.db_loaded = False
            self.points_layer.data = np.empty((0, 3))
            return

        self.G = nx.Graph()
        rtree_data = []
        for node in nodes:
            nid = node["nid"]
            coord = node["coord"]
            self.G.add_node(nid, **node)
            # 以 (nid, (x, y, z, x, y, z), None) 插入 rtree
            rtree_data.append((nid, tuple(coord + coord), None))
        for e in edges:
            self.G.add_edge(e["src"], e["des"], **e)

        p = index.Property(dimension=3)
        self.rtree = index.Index(rtree_data, properties=p)
        self.db_loaded = True

        show_info("Database loaded successfully.")
        print(f"[DEBUG] Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")

        # 默认显示全部节点，即不做 ROI 筛选
        self._display_all_points()

    def _display_all_points(self):
        """
        显示数据库中所有节点，不进行 ROI 筛选。
        """
        if not self.db_loaded or self.G is None:
            show_info("Database not loaded.")
            self.points_layer.data = np.empty((0, 3))
            return

        coords = []
        colors = []
        nids = []
        for nid in self.G.nodes:
            node_data = self.G.nodes[nid]
            coord = node_data["coord"]
            coords.append(coord)
            nids.append(nid)
            colors.append(random.random())

        if not coords:
            show_info("No nodes to display.")
            self.points_layer.data = np.empty((0, 3))
            return

        coords_arr = np.array(coords, dtype=float)
        colors_arr = np.array(colors, dtype=float)
        properties = {"colors": colors_arr, "nids": np.array(nids)}

        self.points_layer.data = coords_arr
        self.points_layer.properties = properties
        # 这里设置 colormap 模式：确保属性 'colors' 存在
        self.points_layer.face_color = "colors"
        self.points_layer.face_colormap = "hsl"
        self.points_layer.color_mode = "colormap"
        self.points_layer.face_contrast_limits = [0, 1]
        self.points_layer.size = 5

        self.viewer.reset_view()
        self.viewer.layers.selection.active = self.points_layer
        print(f"[DEBUG] Displayed all {len(coords)} points.")

    def refresh_points(self):
        """
        根据 ROI 参数筛选节点：
          1. 从用户输入中解析 ROI：起点和尺寸；
          2. 利用 rtree.intersection 筛选出 ROI 内节点；
          3. 遍历图的连通分量，只保留 ROI 内的节点；
          4. 更新 Points Layer，使用属性 'colors' 进行颜色映射。
        """
        if not self.db_loaded or self.G is None:
            show_info("Database not loaded or Graph is None.")
            return

        try:
            x_off = int(float(self.x_offset.value))
            y_off = int(float(self.y_offset.value))
            z_off = int(float(self.z_offset.value))
        except ValueError:
            show_info("Invalid ROI offset input; please enter numeric values.")
            return

        roi_offset = (x_off, y_off, z_off)
        try:
            x_sz = int(float(self.x_size.value))
            y_sz = int(float(self.y_size.value))
            z_sz = int(float(self.z_size.value))
        except ValueError:
            show_info("Invalid ROI size input; please enter numeric values.")
            return

        roi_size = (x_sz, y_sz, z_sz)
        roi_min = roi_offset
        roi_max = (x_off + x_sz, y_off + y_sz, z_off + z_sz)
        query_box = (*roi_min, *roi_max)

        # 使用 rtree 过滤 ROI 内的节点 ID
        node_ids_in_roi = list(self.rtree.intersection(query_box, objects=False))
        roi_set = set()
        for nid in node_ids_in_roi:
            if not self.G.has_node(nid):
                continue
            coord = self.G.nodes[nid]["coord"]
            # 精确判断：左闭右开
            if all(mn <= c < mx for mn, c, mx in zip(roi_min, coord, roi_max)):
                roi_set.add(nid)
        print(f"[DEBUG] ROI: offset={roi_offset}, size={roi_size}, found {len(roi_set)} nodes in bounding box.")

        # 遍历所有连通分量，只保留 ROI 内的节点
        connected_components = list(nx.connected_components(self.G))
        coords = []
        colors = []
        nids = []

        for i, cc in enumerate(connected_components):
            # 仅保留落在 ROI 内的节点
            cc_in_roi = cc.intersection(roi_set)
            if not cc_in_roi:
                continue
            # 为该连通分量分配一个随机颜色数值
            color_val = random.random()
            for nid in cc_in_roi:
                node_data = self.G.nodes[nid]
                coord = node_data["coord"]
                coords.append(coord)
                nids.append(nid)
                colors.append(color_val)

        if not coords:
            show_info("No nodes found after ROI filtering.")
            self.points_layer.data = np.empty((0, 3))
            return

        print(f"[DEBUG] Final filtered points: {len(coords)}")

        coords_arr = np.array(coords, dtype=float)
        colors_arr = np.array(colors, dtype=float)
        properties = {"colors": colors_arr, "nids": np.array(nids)}

        self.points_layer.data = coords_arr
        self.points_layer.properties = properties
        self.points_layer.face_color = "colors"
        self.points_layer.face_colormap = "hsl"
        self.points_layer.color_mode = "colormap"
        self.points_layer.face_contrast_limits = [0, 1]
        self.points_layer.size = 5

        self.viewer.reset_view()
        self.viewer.layers.selection.active = self.points_layer
        print("[DEBUG] ROI points updated and view reset.")

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    """让 napari 识别 DbROIViewer 作为插件 Dock Widget."""
    return [DbROIViewer]
