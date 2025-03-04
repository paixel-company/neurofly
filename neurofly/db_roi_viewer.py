import os
import random
import numpy as np
import networkx as nx
from rtree import index
from magicgui import widgets
from napari.utils.notifications import show_info
from neurofly.dbio import read_nodes, read_edges, delete_nodes

class DbROIViewer(widgets.Container):
    """
    A simplified viewer that only displays annotation points from a database,
    similar to the 'panorama' part of the Annotator code, but without any image layer.
    """

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')

        # 在 napari 中添加一个 Points Layer 用于显示数据库中的节点
        self.points_layer = self.viewer.add_points(
            data=None,
            ndim=3,
            name='db roi',
            face_color='colors',
            face_colormap='hsl',
            size=2,
            blending='additive',
            visible=True
        )

        # 基础数据结构
        self.G = None           # 用于保存从数据库加载的 networkx.Graph
        self.rtree = None       # 用于快速查询
        self.db_loaded = False  # 标记是否已从数据库加载

        # ------------- 界面控件 -------------
        # 数据库路径
        self.db_path = widgets.FileEdit(label="Database Path", filter='*.db')
        self.segs_switch = widgets.CheckBox(value=True, text='Show/Hide Long Segments')
        self.min_length = widgets.Slider(label="Short Segs Filter", value=10, min=0, max=200)
        self.len_thres = widgets.Slider(label="Length Thres", value=20, min=0, max=9999)
        self.point_size = widgets.Slider(label="Point Size", value=3, min=1, max=10)
        self.refresh_button = widgets.PushButton(text="Refresh")
        
        # 将控件放进容器
        self.extend([
            self.db_path,
            self.segs_switch,
            self.min_length,
            self.len_thres,
            self.point_size,
            self.refresh_button,
        ])

        # 事件绑定
        self.db_path.changed.connect(self.on_db_changed)
        self.refresh_button.clicked.connect(self.refresh_points)

    def on_db_changed(self):
        """
        当数据库路径改变时，重新加载数据库，构建 networkx 图和 rtree。
        """
        db_path_str = str(self.db_path.value)
        if not os.path.exists(db_path_str):
            show_info("Database path does not exist.")
            self.G = None
            self.db_loaded = False
            self.points_layer.data = np.empty((0, 3))
            return
        
        # 1. 读取节点和边
        nodes = read_nodes(db_path_str)
        edges = read_edges(db_path_str)
        if not nodes:
            show_info("No nodes found in the database.")
            self.G = None
            self.db_loaded = False
            self.points_layer.data = np.empty((0, 3))
            return

        # 2. 构建 Graph
        self.G = nx.Graph()
        rtree_data = []
        for node in nodes:
            nid = node['nid']
            coord = node['coord']
            # type, checked, creator 等信息都可附加在 Graph 节点属性上
            self.G.add_node(nid, **node)
            # R-tree 插入范围： (x1, y1, z1, x2, y2, z2)
            # 这里简单将点看成 [coord, coord]
            rtree_data.append((nid, tuple(coord + coord), None))
        
        for edge in edges:
            self.G.add_edge(edge['src'], edge['des'], creator=edge['creator'])

        # 3. 构建 R-tree
        p = index.Property(dimension=3)
        self.rtree = index.Index(rtree_data, properties=p)

        self.db_loaded = True
        # 加载完成后刷新一次
        self.refresh_points()

    def refresh_points(self):
        """
        将数据库节点按照连通分量聚合，并根据长度筛选、颜色映射后，显示在 points_layer 中。
        """
        if not self.db_loaded or self.G is None:
            show_info("Database not loaded or empty.")
            self.points_layer.data = np.empty((0, 3))
            return
        
        # 对图进行连通分量分析
        connected_components = list(nx.connected_components(self.G))

        coords = []
        colors = []
        sizes = []
        nids = []

        for cc in connected_components:
            # cc 是一个节点ID的集合
            length_cc = len(cc)  # 以节点数量作为连通分量的“长度”做简单判定

            # 与示例中相同的逻辑：
            #   1. 如果 length(cc) < len_thres.value 并且 segs_switch = True，则跳过
            #   2. 如果 length(cc) >= len_thres.value 并且 segs_switch = False，则跳过
            #   3. 如果连通分量大小 <= min_length.value，则跳过
            if (length_cc < self.len_thres.value and self.segs_switch.value) or length_cc <= self.min_length.value:
                continue
            if (length_cc >= self.len_thres.value and not self.segs_switch.value) or length_cc <= self.min_length.value:
                continue

            # 给这个连通分量随机一个颜色值 0~1
            color_val = random.random()

            # 遍历该连通分量下的所有节点
            for nid in cc:
                if nid not in self.G.nodes:
                    continue
                
                node_data = self.G.nodes[nid]
                # 如果节点信息被清空等情况：
                if not node_data:
                    # 如果需要的话也可以 delete_nodes(...)
                    continue

                # coords
                c = node_data['coord']
                coords.append(c)
                nids.append(nid)
                colors.append(color_val)
                # 大小统一用 self.point_size.value
                sizes.append(self.point_size.value)

        # 如果全部被过滤掉，就报个提示
        if not coords:
            show_info("No segments in range or all filtered out.")
            self.points_layer.data = np.empty((0, 3))
            return

        # 归一化 colors 数组
        # 这里的做法：把 color_val 放在 0~1，已经满足 face_colormap='hsl' 的需求
        colors = np.array(colors, dtype=float)
        coords = np.array(coords, dtype=float)
        sizes = np.array(sizes, dtype=float)

        # 构建 napari 的 properties
        properties = {
            'colors': colors,  # 用于 hsl colormap
            'nids': np.array(nids)
        }

        # 更新 points_layer
        self.points_layer.data = coords
        self.points_layer.properties = properties
        # 告诉 napari 应用 face_color = 'colors' 列
        self.points_layer.face_colormap = 'hsl'
        self.points_layer.face_color = 'colors'
        self.points_layer.size = sizes

        # 重置视图
        self.viewer.reset_view()
        self.viewer.layers.selection.active = self.points_layer
