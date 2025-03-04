import numpy as np
import napari
import os
from magicgui import widgets
from napari.utils.notifications import show_info
from neurofly.image_reader import wrap_image
from tifffile import imwrite

class ROIViewer(widgets.Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.dims.ndisplay = 3
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')
        
        # 图像层：显示从图像中提取的 ROI 区域
        self.image_layer = self.viewer.add_image(
            np.zeros((64, 64, 64), dtype=np.uint16), name='image'
        )
        # 目标层：用于交互时添加点（例如双击）
        self.goal_layer = self.viewer.add_points(
            ndim=3, face_color='red', size=1, shading='spherical', name='goal'
        )
        # DB ROI 层：显示从数据库中读取的 ROI 注释点（以绿色显示）
        self.db_roi_layer = self.viewer.add_points(
            ndim=3, face_color='green', size=2, shading='spherical', name='DB ROI'
        )
        
        self.image = None
        self.previous_level = 0
        
        # =========== 初始化控件，包含 db_path ===========

        # 图像路径
        self.image_path = widgets.FileEdit(label="image_path", mode='r')
        
        # 数据库路径
        self.db_path = widgets.FileEdit(label="DB Path", mode='r', filter='*.db')
        
        # 保存目录
        self.save_dir = widgets.FileEdit(label="save dir", mode='d')
        
        # 读 zarr 格式
        self.image_type = widgets.CheckBox(value=False, text='read zarr format')
        
        # 分辨率和通道下拉框
        self.resolution_dropdown = widgets.ComboBox(
            choices=[], label='Resolution Level', tooltip='Select resolution level'
        )
        self.channel_dropdown = widgets.ComboBox(
            choices=[], label='Channel', tooltip='Select channel'
        )
        
        # block size sliders 和 offset
        self.x_size = widgets.Slider(label="x size", value=128, min=0, max=2048)
        self.y_size = widgets.Slider(label="y size", value=128, min=0, max=2048)
        self.z_size = widgets.Slider(label="z size", value=128, min=0, max=2048)
        self.clip = widgets.CheckBox(value=False, text='clip slider bars')
        self.x = widgets.LineEdit(label="x offset", value=0)
        self.y = widgets.LineEdit(label="y offset", value=0)
        self.z = widgets.LineEdit(label="z offset", value=0)
        
        # 其他按钮
        self.button_refresh = widgets.PushButton(text="refresh")
        self.button_level_up = widgets.PushButton(text="level up")
        self.button_level_down = widgets.PushButton(text="level down")
        self.button_save_image = widgets.PushButton(text="save image")
        self.button_toggle_full = widgets.PushButton(text="toggle full view")
        
        self.level_info = widgets.TextEdit(label='level info')

        # =========== 将控件添加到容器中并绑定事件 ===========
        self.extend([
            self.image_type,
            self.image_path,
            self.db_path,            # 将 db_path 加到容器中
            self.save_dir,
            self.resolution_dropdown,
            self.channel_dropdown,
            self.clip,
            self.x_size,
            self.y_size,
            self.z_size,
            self.x,
            self.y,
            self.z,
            self.level_info,
            self.button_save_image,
            self.button_level_up,
            self.button_level_down,
            self.button_toggle_full,
            self.button_refresh,
        ])
        
        # 绑定事件
        self.image_path.changed.connect(self.on_image_reading)
        self.image_type.changed.connect(self.switch_image_type)
        self.resolution_dropdown.changed.connect(self.on_resolution_change)
        self.channel_dropdown.changed.connect(self.on_channel_change)
        self.x_size.changed.connect(self.clip_x)
        self.y_size.changed.connect(self.clip_y)
        self.z_size.changed.connect(self.clip_z)
        
        self.button_refresh.clicked.connect(self.refresh)
        self.button_level_up.clicked.connect(self.level_up)
        self.button_level_down.clicked.connect(self.level_down)
        self.button_save_image.clicked.connect(self.save_image)
        self.button_toggle_full.clicked.connect(self.toggle_full_view)
        
        # 在 viewer 上绑定键盘和鼠标事件
        self.viewer.bind_key('f', self.refresh, overwrite=True)
        self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)

    # =========== 工具函数和事件回调 ===========

    def switch_image_type(self, event):
        """根据 checkBox 切换文件选择器模式。"""
        if event:
            self.image_path.mode = 'd'
        else:
            self.image_path.mode = 'r'

    def on_image_reading(self):
        """当图像路径变化时，读取图像并初始化 ROI 参数。"""
        try:
            self.image = wrap_image(str(self.image_path.value))
        except Exception as e:
            show_info(f"Error loading image: {e}")
            return

        # 假设默认使用第 0 个 ROI 信息来初始化
        x_offset, y_offset, z_offset, x_size, y_size, z_size = self.image.rois[0]
        self.x.value = x_offset + x_size // 2
        self.y.value = y_offset + y_size // 2
        self.z.value = z_offset + z_size // 2

        resolution_levels = self.image.resolution_levels
        channels = self.image.channels
        
        # 更新分辨率和通道下拉框
        self.resolution_dropdown.changed.disconnect(self.on_resolution_change)
        self.resolution_dropdown.choices = resolution_levels
        self.resolution_dropdown.value = resolution_levels[0]
        self.resolution_dropdown.changed.connect(self.on_resolution_change)

        self.channel_dropdown.changed.disconnect(self.on_channel_change)
        self.channel_dropdown.choices = channels
        self.channel_dropdown.value = channels[0]
        self.channel_dropdown.changed.connect(self.on_channel_change)
        
        self.refresh()

    def on_resolution_change(self, event):
        """切换分辨率级别时，根据 spacing 调整 ROI 的偏移量。"""
        if not self.image:
            return
        if isinstance(self.resolution_dropdown.value, str):
            cx = int(float(self.x.value)) + self.x_size.value // 2
            cy = int(float(self.y.value)) + self.y_size.value // 2
            cz = int(float(self.z.value)) + self.z_size.value // 2
            cl = self.previous_level
            tl = self.image.resolution_levels.index(self.resolution_dropdown.value)
            self.previous_level = tl
            c_spacing = self.image.info[cl]['spacing']
            t_spacing = self.image.info[tl]['spacing']
            scale = [i / j for i, j in zip(c_spacing, t_spacing)]
            tx = int(cx * scale[0])
            ty = int(cy * scale[1])
            tz = int(cz * scale[2])
            self.x.value = tx - self.x_size.value // 2
            self.y.value = ty - self.y_size.value // 2
            self.z.value = tz - self.z_size.value // 2
            self.goal_layer.data = [tx, ty, tz]
            self.refresh()

    def on_channel_change(self, event):
        """切换通道后刷新显示。"""
        if isinstance(self.channel_dropdown.value, str):
            self.refresh()

    def clip_x(self):
        """若 clip 选中，则将 x_size 强制为 32 的整数倍。"""
        if self.clip.value:
            self.x_size.value = (self.x_size.value // 32) * 32

    def clip_y(self):
        if self.clip.value:
            self.y_size.value = (self.y_size.value // 32) * 32

    def clip_z(self):
        if self.clip.value:
            self.z_size.value = (self.z_size.value // 32) * 32

    def refresh(self, *args, **kwargs):
        """根据当前 ROI 和分辨率、通道加载图像，并根据 ROI 显示数据库标注点。"""
        if not self.image:
            return
        
        roi = [
            int(float(self.x.value)),
            int(float(self.y.value)),
            int(float(self.z.value)),
            int(self.x_size.value),
            int(self.y_size.value),
            int(self.z_size.value),
        ]
        resolution_level = str(self.resolution_dropdown.value)
        channel = str(self.channel_dropdown.value)

        try:
            roi_image = self.image.from_roi(roi, resolution_level, channel)
        except Exception as e:
            show_info(f"Error extracting ROI: {e}")
            return

        # 更新图像层
        self.image_layer.data = roi_image
        self.image_layer.translate = roi[:3]
        camera_state = self.viewer.camera.angles
        self.viewer.reset_view()
        self.viewer.camera.angles = camera_state
        self.viewer.layers.selection.active = self.image_layer
        self.image_layer.reset_contrast_limits()

        # 更新 level_info
        lvl_idx = self.image.resolution_levels.index(resolution_level)
        info = "\n".join(
            f"{key}: {value}"
            for key, value in self.image.info[lvl_idx].items()
        )
        self.level_info.value = info

        # ---------------- 读取数据库并在 db_roi_layer 中显示 ----------------
        db_path_str = str(self.db_path.value)
        if os.path.exists(db_path_str):
            try:
                from neurofly.dbio import read_nodes
                nodes = read_nodes(db_path_str)
            except Exception as e:
                show_info(f"Error reading database: {e}")
                self.db_roi_layer.data = np.empty((0, 3))
                return

            # 计算当前 ROI 的上界
            roi_bounds = [
                roi[0] + roi[3],
                roi[1] + roi[4],
                roi[2] + roi[5],
            ]
            points = []
            for node in nodes:
                coord = node['coord']
                # 判断节点坐标是否落在当前 ROI 范围内
                if all(
                    roi_min <= c < roi_max
                    for roi_min, c, roi_max in zip(roi[:3], coord, roi_bounds)
                ):
                    points.append(coord)

            if points:
                self.db_roi_layer.data = np.array(points)
            else:
                self.db_roi_layer.data = np.empty((0, 3))
        else:
            self.db_roi_layer.data = np.empty((0, 3))

    def toggle_full_view(self):
        """尝试显示当前分辨率下的整块图像（若尺寸太大就跳过）。"""
        if not self.image:
            return
        
        rl = self.resolution_dropdown.value
        lvl_idx = self.image.resolution_levels.index(rl)
        image_size = self.image.info[lvl_idx]['image_size']
        if min(image_size) > 1024:
            show_info("This image is too large to display fully.")
            return
        
        roi_full = self.image.rois[lvl_idx]
        try:
            image = self.image.from_roi(roi_full, rl, self.channel_dropdown.value)
        except Exception as e:
            show_info(f"Error extracting full view: {e}")
            return

        self.image_layer.data = image
        self.image_layer.translate = roi_full[:3]
        camera_state = self.viewer.camera.angles
        self.viewer.reset_view()
        self.viewer.camera.angles = camera_state
        self.viewer.layers.selection.active = self.image_layer
        self.image_layer.reset_contrast_limits()

    def level_up(self):
        """将分辨率向更低精度（索引更大）切换。"""
        if not self.image:
            return
        cx = int(float(self.x.value)) + self.x_size.value // 2
        cy = int(float(self.y.value)) + self.y_size.value // 2
        cz = int(float(self.z.value)) + self.z_size.value // 2
        cl = self.image.resolution_levels.index(self.resolution_dropdown.value)
        if cl == len(self.image.rois) - 1:
            return
        tl = cl + 1
        self.previous_level = tl
        c_spacing = self.image.info[cl]['spacing']
        t_spacing = self.image.info[tl]['spacing']
        scale = [i / j for i, j in zip(c_spacing, t_spacing)]
        self.resolution_dropdown.changed.disconnect(self.on_resolution_change)
        self.resolution_dropdown.value = self.image.resolution_levels[tl]
        self.resolution_dropdown.changed.connect(self.on_resolution_change)
        tx = int(cx * scale[0])
        ty = int(cy * scale[1])
        tz = int(cz * scale[2])
        self.x.value = tx - self.x_size.value // 2
        self.y.value = ty - self.y_size.value // 2
        self.z.value = tz - self.z_size.value // 2
        self.goal_layer.data = [tx, ty, tz]
        self.refresh()

    def level_down(self):
        """将分辨率向更高精度（索引更小）切换。"""
        if not self.image:
            return
        cx = int(float(self.x.value)) + self.x_size.value // 2
        cy = int(float(self.y.value)) + self.y_size.value // 2
        cz = int(float(self.z.value)) + self.z_size.value // 2
        cl = self.image.resolution_levels.index(self.resolution_dropdown.value)
        if cl == 0:
            return
        tl = cl - 1
        self.previous_level = tl
        c_spacing = self.image.info[cl]['spacing']
        t_spacing = self.image.info[tl]['spacing']
        scale = [i / j for i, j in zip(c_spacing, t_spacing)]
        self.resolution_dropdown.changed.disconnect(self.on_resolution_change)
        self.resolution_dropdown.value = self.image.resolution_levels[tl]
        self.resolution_dropdown.changed.connect(self.on_resolution_change)
        tx = int(cx * scale[0])
        ty = int(cy * scale[1])
        tz = int(cz * scale[2])
        self.x.value = tx - self.x_size.value // 2
        self.y.value = ty - self.y_size.value // 2
        self.z.value = tz - self.z_size.value // 2
        self.goal_layer.data = [tx, ty, tz]
        self.refresh()

    def on_double_click(self, layer, event):
        """双击图像，根据射线投射找到最大强度点并更新 ROI。"""
        near_point, far_point = layer.get_ray_intersections(
            event.position,
            event.view_direction,
            event.dims_displayed
        )
        sample_ray = far_point - near_point
        length_sample_vector = np.linalg.norm(sample_ray)
        increment_vector = sample_ray / (2 * length_sample_vector)
        n_iterations = int(2 * length_sample_vector)
        bbox = np.array([
            [0, layer.data.shape[0] - 1],
            [0, layer.data.shape[1] - 1],
            [0, layer.data.shape[2] - 1]
        ])
        sample_points = []
        values = []
        for i in range(n_iterations):
            sample_point = np.asarray(near_point + i * increment_vector, dtype=int)
            sample_point = np.clip(sample_point, bbox[:, 0], bbox[:, 1])
            value = layer.data[sample_point[0], sample_point[1], sample_point[2]]
            sample_points.append(sample_point)
            values.append(value)

        max_point_index = values.index(max(values))
        max_point = sample_points[max_point_index]
        max_point = [a + b for a, b in zip(max_point, self.image_layer.translate)]
        print('Put point at: ', max_point)
        
        if event.button == 1:
            self.goal_layer.data = max_point
            self.x.value = max_point[0] - self.x_size.value // 2
            self.y.value = max_point[1] - self.y_size.value // 2
            self.z.value = max_point[2] - self.z_size.value // 2
            self.refresh()

    def save_image(self, *args, **kwargs):
        """保存当前 ROI 图像至指定目录。"""
        if not self.image:
            return
        image = self.image_layer.data
        all_files = os.listdir(str(self.save_dir.value))
        tif_files = [file for file in all_files if file.endswith('.tif')]
        next_image_index = len(tif_files) + 1
        image_name = f'img_{next_image_index}.tif'
        image_name = os.path.join(str(self.save_dir.value), image_name)

        imwrite(image_name, image, compression='zlib', compressionargs={'level': 8})
        print(image_name + ' saved')
