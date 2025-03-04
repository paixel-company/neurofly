import os
import random
import numpy as np
import networkx as nx
from rtree import index

import napari
from napari_plugin_engine import napari_hook_implementation
from magicgui import widgets
from napari.utils.notifications import show_info

# 假设你在 neurofly.dbio 中有 read_nodes, read_edges, delete_nodes 等函数
from neurofly.dbio import read_nodes, read_edges, delete_nodes

class DbROIViewer(widgets.Container):
    """A minimal napari Dock Widget that shows DB ROI points, without any image display."""

    def __init__(self, viewer: napari.Viewer = None):
        """
        当 napari 以无参方式实例化本类时，viewer 为 None；
        我们尝试通过 napari.current_viewer() 获取当前激活的 viewer。
        """
        super().__init__()
        if viewer is None:
            viewer = napari.current_viewer()
            if viewer is None:
                show_info("No active napari viewer found. Some functionality will be unavailable.")
                return
        self.viewer = viewer

        # 清理已有图层（按需）
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')

        # 新建一个 Points Layer 用于显示数据库标注点
        self.points_layer = self.viewer.add_points(
            data=None,
            ndim=3,
            name="DB ROI Points",
            face_color='colors',
            face_colormap='hsl',
            size=2,
            blending='additive',
            visible=True
        )

        # 数据结构
        self.G = None        # networkx 图
        self.rtree = None    # R-tree 索引
        self.db_loaded = False

        # ---------- 创建界面控件 ----------
        self.db_path = widgets.FileEdit(label="Database Path", filter='*.db')
        self.segs_switch = widgets.CheckBox(value=True, text='Show/Hide Long Segments')
        self.min_length = widgets.Slider(label="Short Segs Filter", value=10, min=0, max=200)
        self.len_thres = widgets.Slider(label="Length Thres", value=20, min=0, max=9999)
        self.point_size = widgets.Slider(label="Point Size", value=3, min=1, max=10)
        self.refresh_button = widgets.PushButton(text="Refresh")

        # 将以上控件添加到容器
        self.extend([
            self.db_path,
            self.segs_switch,
            self.min_length,
            self.len_thres,
            self.point_size,
            self.refresh_button,
        ])

        # 绑定事件
        self.db_path.changed.connect(self._on_db_path_changed)
        self.refresh_button.clicked.connect(self._refresh_points)

    def _on_db_path_changed(self):
        """
        当数据库路径发生变化时，重新加载数据库，构建 graph 与 rtree。
        """
        db_path_str = str(self.db_path.value)
        if not os.path.exists(db_path_str):
            show_info(f"Database path {db_path_str} not found.")
            self.G = None
            self.db_loaded = False
            self.points_layer.data = np.empty((0, 3))
            return

        # 读取节点与边
        nodes = read_nodes(db_path_str)
        edges = read_edges(db_path_str)
        if not nodes:
            show_info("No nodes in the database.")
            self.G = None
            self.db_loaded = False
            self.points_layer.data = np.empty((0, 3))
            return

        # 构建 Graph
        self.G = nx.Graph()
        rtree_data = []
        for node in nodes:
            nid = node['nid']
            coord = node['coord']
            self.G.add_node(nid, **node)  # 把 node 中所有键值都作为属性加进图
            rtree_data.append((nid, tuple(coord + coord), None))
        for edge in edges:
            self.G.add_edge(edge['src'], edge['des'], creator=edge['creator'])

        # 构建 R-tree
        p = index.Property(dimension=3)
        self.rtree = index.Index(rtree_data, properties=p)

        self.db_loaded = True
        # 加载完成后自动刷新显示
        self._refresh_points()

    def _refresh_points(self, *args):
        """
        显示符合长度筛选条件的连通分量，每个连通分量随机染色。
        """
        if not self.db_loaded or self.G is None:
            show_info("Database not loaded.")
            self.points_layer.data = np.empty((0, 3))
            return

        connected_components = list(nx.connected_components(self.G))

        coords = []
        colors = []
        sizes = []
        nids = []

        for cc in connected_components:
            length_cc = len(cc)
            # 与 Annotator 中 panorama 的逻辑类似：
            # 如果连通分量大小 < len_thres.value 且 segs_switch=True，则跳过
            # 如果连通分量大小 >= len_thres.value 且 segs_switch=False，则跳过
            # 或长度 <= min_length.value，也跳过
            if (length_cc < self.len_thres.value and self.segs_switch.value) or (length_cc <= self.min_length.value):
                continue
            if (length_cc >= self.len_thres.value and not self.segs_switch.value) or (length_cc <= self.min_length.value):
                continue

            color_val = random.random()  # 用于 HSL colormap
            for nid in cc:
                if not self.G.has_node(nid):
                    continue
                node_data = self.G.nodes[nid]
                if not node_data:
                    continue  # 若节点数据已被删除或为空

                coord = node_data['coord']
                coords.append(coord)
                nids.append(nid)
                colors.append(color_val)
                sizes.append(self.point_size.value)

        if not coords:
            show_info("No segment in range or all filtered out.")
            self.points_layer.data = np.empty((0, 3))
            return

        coords = np.array(coords, dtype=float)
        colors = np.array(colors, dtype=float)
        sizes = np.array(sizes, dtype=float)

        properties = {
            'colors': colors,
            'nids': np.array(nids)
        }

        self.points_layer.data = coords
        self.points_layer.properties = properties
        self.points_layer.face_colormap = 'hsl'
        self.points_layer.face_color = 'colors'
        self.points_layer.size = sizes

        # 视角重置
        self.viewer.reset_view()
        self.viewer.layers.selection.active = self.points_layer


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    """
    让 napari 识别 DbROIViewer 作为 Dock Widget 插件。
    """
    return [DbROIViewer]
