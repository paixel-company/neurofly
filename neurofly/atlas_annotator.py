import numpy as np
import napari
from magicgui import magicgui, widgets
from neurofly.image_reader import wrap_image


class AtlasAnnotator(widgets.Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.dims.ndisplay = 2  # 显示2D切片
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')

        # 初始化时不添加图像层，等待图像数据加载后再添加
        self.image_layer = None
        self.image = None

        # 添加 z 轴切片位置选择器
        self.z_slider = widgets.Slider(label="Z Slice", value=0, min=0, max=1)  # 初始最大值设置为1，稍后根据数据调整
        self.z_slider.changed.connect(self.on_z_slice_change)  # 绑定事件
        self.add_callback()

    def add_callback(self):
        # 为 z 轴滑动条添加回调
        self.viewer.bind_key('f', self.refresh, overwrite=True)
        self.image_path = widgets.FileEdit(label="image_path")
        self.image_path.changed.connect(self.on_image_reading)

        self.button_refresh = widgets.PushButton(text="refresh")
        self.button_refresh.clicked.connect(self.refresh)

        # 布局：添加滑动条和按钮
        self.extend([self.image_path, self.z_slider, self.button_refresh])

    def on_image_reading(self):
        self.image = wrap_image(str(self.image_path.value))
        
        # 根据图像的实际 z 轴大小设置滑动条最大值
        max_z = self.image.shape[2] - 1
        self.z_slider.max = max_z
        self.z_slider.value = 0  # 初始设置为 z=0 切片

        # 初始化图像层（根据实际图像尺寸）
        if self.image_layer is None:
            # 添加图像层，图像数据从当前 z 切片开始
            self.image_layer = self.viewer.add_image(self.image[:, :, 0], name='image')

        self.refresh()

    def refresh(self):
        if self.image is None:
            return

        # 获取当前 z 切片位置
        z_value = self.z_slider.value
        
        # 提取图像的该切片并更新显示
        self.image_layer.data = self.image[:, :, z_value]
        self.viewer.layers.selection.active = self.image_layer
        self.image_layer.reset_contrast_limits()

    def on_z_slice_change(self, event):
        """当用户通过滑动条选择 z 切片时，刷新图像显示"""
        self.refresh()
