import numpy as np
import napari
import os
from magicgui import widgets
from napari.utils.notifications import show_info
from neurofly.dbio import read_nodes

class DbROIViewer(widgets.Container):
    """
    A simplified viewer that only displays ROI annotation points from a database,
    without loading or displaying any image data.
    """
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.clear()  # Clear existing layers if needed
        self.viewer.window.remove_dock_widget('all')

        # 新建一个 Points 图层用于显示数据库标注点
        self.roi_points_layer = self.viewer.add_points(
            np.empty((0, 3)), name="ROI Points",
            face_color='green', size=2, shading='spherical'
        )

        # 数据库路径
        self.db_path = widgets.FileEdit(label="DB Path", mode='r', filter='*.db')

        # ROI 偏移量 (x, y, z) 与 ROI 尺寸 (x_size, y_size, z_size)
        self.x = widgets.LineEdit(label="X Offset", value="0")
        self.y = widgets.LineEdit(label="Y Offset", value="0")
        self.z = widgets.LineEdit(label="Z Offset", value="0")

        self.x_size = widgets.Slider(label="X Size", value=128, min=0, max=2048)
        self.y_size = widgets.Slider(label="Y Size", value=128, min=0, max=2048)
        self.z_size = widgets.Slider(label="Z Size", value=128, min=0, max=2048)

        # 刷新按钮
        self.refresh_button = widgets.PushButton(text="Refresh ROI")

        # 将控件添加到容器
        self.extend([
            self.db_path,
            self.x, self.y, self.z,
            self.x_size, self.y_size, self.z_size,
            self.refresh_button
        ])

        # 绑定事件
        self.db_path.changed.connect(self.refresh)
        self.x_size.changed.connect(self.refresh)
        self.y_size.changed.connect(self.refresh)
        self.z_size.changed.connect(self.refresh)
        self.refresh_button.clicked.connect(self.refresh)

    def refresh(self):
        """
        根据当前 ROI 参数，从数据库里读取标注点并在 self.roi_points_layer 中显示。
        """
        db_path_str = str(self.db_path.value)
        if not os.path.exists(db_path_str):
            show_info("Database path does not exist.")
            self.roi_points_layer.data = np.empty((0, 3))
            return

        try:
            nodes = read_nodes(db_path_str)
        except Exception as e:
            show_info(f"Error reading database: {e}")
            self.roi_points_layer.data = np.empty((0, 3))
            return

        # 获取当前 ROI
        try:
            x_off = int(float(self.x.value))
            y_off = int(float(self.y.value))
            z_off = int(float(self.z.value))
        except ValueError:
            show_info("Invalid ROI offsets. Please enter integers.")
            return

        roi = [
            x_off,
            y_off,
            z_off,
            int(self.x_size.value),
            int(self.y_size.value),
            int(self.z_size.value)
        ]
        # 计算 ROI 上界
        roi_bounds = [
            roi[0] + roi[3],
            roi[1] + roi[4],
            roi[2] + roi[5]
        ]

        points = []
        for node in nodes:
            coord = node['coord']
            # 判断该点是否位于 ROI 区域内
            if all(mn <= c < mx for mn, c, mx in zip(roi[:3], coord, roi_bounds)):
                points.append(coord)

        if points:
            self.roi_points_layer.data = np.array(points)
        else:
            self.roi_points_layer.data = np.empty((0, 3))
