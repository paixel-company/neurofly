import numpy as np
import napari
import os
from magicgui import widgets
from napari.utils.notifications import show_info
from neurofly.image_reader import wrap_image
from neurofly.dbio import read_nodes

class ROIViewer(widgets.Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.clear()
        
        # 创建显示ROI图像的 layer
        self.roi_image_layer = self.viewer.add_image(np.zeros((10,10,10)), name="ROI Image", visible=True)
        # 创建显示标注点的 layer
        self.roi_points_layer = self.viewer.add_points(np.empty((0, 3)), name="ROI Points", visible=True)
        
        # 文件路径输入控件
        self.image_path = widgets.FileEdit(label="Image Path", mode="r")
        self.db_path = widgets.FileEdit(label="DB Path", filter="*.db")
        
        # ROI 参数输入控件：ROI 起点（左下角坐标）和 ROI 尺寸
        self.roi_origin = widgets.LineEdit(label="ROI Origin (x,y,z)", value="6047,4189,11275")
        self.roi_size = widgets.LineEdit(label="ROI Size (x,y,z)", value="1024,1024,1024")
        
        # 按钮：加载 ROI
        self.load_roi_button = widgets.PushButton(text="Load ROI")
        self.load_roi_button.clicked.connect(self.load_roi)
        
        # 添加控件到容器
        self.extend([
            self.image_path,
            self.db_path,
            self.roi_origin,
            self.roi_size,
            self.load_roi_button
        ])
    
    def parse_coordinates(self, text):
        """
        解析类似 "6047,4189,11275" 或 "6047 4189 11275" 格式的字符串为整数列表
        """
        parts = text.replace(",", " ").split()
        try:
            coords = [int(x) for x in parts]
            return coords
        except ValueError:
            show_info("输入的坐标格式不正确，请以逗号或空格分隔整数。")
            return None
    
    def load_roi(self):
        """
        根据用户输入的 ROI 起点和尺寸提取图像块，并在 ROI Points layer 显示该区域内的标注点
        """
        origin = self.parse_coordinates(self.roi_origin.value)
        size = self.parse_coordinates(self.roi_size.value)
        if origin is None or size is None:
            return
        
        # ROI 起点需为3个数，ROI 尺寸支持1个数（立方体）或3个数
        if len(origin) != 3 or len(size) not in (1, 3):
            show_info("ROI 起点必须为3个数，ROI 尺寸必须为1个或3个数。")
            return
        
        if len(size) == 1:
            size = size * 3  # 如果只输入一个尺寸，则在三个方向都取相同值
        
        # 计算 ROI 的上界（逐元素相加）
        roi_bounds = [o + s for o, s in zip(origin, size)]
        # 构造 ROI 参数：[origin_x, origin_y, origin_z, size_x, size_y, size_z]
        roi_param = origin + size
        
        # 加载图像
        if not os.path.exists(self.image_path.value):
            show_info("图像路径不存在。")
            return
        try:
            image = wrap_image(self.image_path.value)
        except Exception as e:
            show_info(f"加载图像出错: {e}")
            return
        
        # 这里简单采用 level=0, channel=0
        level = 0
        channel = 0
        
        try:
            roi_image = image.from_roi(roi_param, level=level, channel=channel)
        except Exception as e:
            show_info(f"提取 ROI 图像块出错: {e}")
            return
        
        # 更新图像 layer
        self.roi_image_layer.data = roi_image
        self.roi_image_layer.reset_contrast_limits()
        
        # 从数据库加载标注点，并过滤出在 ROI 内的点
        if os.path.exists(self.db_path.value):
            try:
                nodes = read_nodes(self.db_path.value)
            except Exception as e:
                show_info(f"读取数据库出错: {e}")
                return
            points = []
            for node in nodes:
                coord = node['coord']
                # 判断该点是否在 ROI 区域内
                if all(o <= c < b for o, c, b in zip(origin, coord, roi_bounds)):
                    points.append(coord)
            if points:
                self.roi_points_layer.data = np.array(points)
            else:
                self.roi_points_layer.data = np.empty((0, 3))
                show_info("指定 ROI 内没有找到标注点。")
        else:
            show_info("数据库路径不存在，未加载标注点。")
            self.roi_points_layer.data = np.empty((0, 3))
        
        # 调整摄像头视图，使中心位于 ROI 中心
        self.viewer.camera.center = [o + s / 2 for o, s in zip(origin, size)]
        show_info("ROI 加载成功。")