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
        
        # Create an Image layer for the ROI
        self.roi_image_layer = self.viewer.add_image(
            np.zeros((10, 10, 10)), name="ROI Image", visible=True
        )
        # Create a Points layer for annotations
        self.roi_points_layer = self.viewer.add_points(
            np.empty((0, 3)), name="ROI Points", visible=True
        )
        
        # File input widgets (note that FileEdit may return a Path object)
        self.image_path = widgets.FileEdit(label="Image Path", mode="r")
        self.db_path = widgets.FileEdit(label="DB Path", filter="*.db")
        
        # ROI parameters: origin (lower-left corner) and size
        self.roi_origin = widgets.LineEdit(label="ROI Origin (x,y,z)", value="6047,4189,11275")
        self.roi_size = widgets.LineEdit(label="ROI Size (x,y,z)", value="1024,1024,1024")
        
        # Button to load the ROI
        self.load_roi_button = widgets.PushButton(text="Load ROI")
        self.load_roi_button.clicked.connect(self.load_roi)
        
        # Add all widgets to the container
        self.extend([
            self.image_path,
            self.db_path,
            self.roi_origin,
            self.roi_size,
            self.load_roi_button
        ])
    
    def parse_coordinates(self, text):
        """
        Parse a string like "6047,4189,11275" or "6047 4189 11275" into a list of integers.
        """
        parts = text.replace(",", " ").split()
        try:
            coords = [int(x) for x in parts]
            return coords
        except ValueError:
            show_info("Invalid coordinate format. Please separate integers with commas or spaces.")
            return None
    
    def load_roi(self):
        """
        Extract the ROI specified by the user and display its annotation points.
        """
        origin = self.parse_coordinates(self.roi_origin.value)
        size = self.parse_coordinates(self.roi_size.value)
        if origin is None or size is None:
            return
        
        # ROI origin must have 3 values, ROI size must be either 1 or 3 values
        if len(origin) != 3 or len(size) not in (1, 3):
            show_info("ROI origin must have 3 values, and ROI size must be either 1 or 3 values.")
            return
        
        # If only one dimension is provided for size, expand it to (x, y, z)
        if len(size) == 1:
            size = size * 3
        
        # Calculate the upper bounds of the ROI
        roi_bounds = [o + s for o, s in zip(origin, size)]
        # Build the ROI parameter: [origin_x, origin_y, origin_z, size_x, size_y, size_z]
        roi_param = origin + size
        
        # Convert the FileEdit (possibly Path) to a string
        image_path_str = str(self.image_path.value)
        db_path_str = str(self.db_path.value)
        
        # Check if the image path exists
        if not os.path.exists(image_path_str):
            show_info("Image path does not exist.")
            return
        
        # Wrap the image
        try:
            image = wrap_image(image_path_str)
        except Exception as e:
            show_info(f"Error loading image: {e}")
            return
        
        # For simplicity, use level=0 and channel=0
        level = 0
        channel = 0
        
        # Extract the ROI
        try:
            roi_image = image.from_roi(roi_param, level=level, channel=channel)
        except Exception as e:
            show_info(f"Error extracting ROI image: {e}")
            return
        
        # Update the Image layer
        self.roi_image_layer.data = roi_image
        self.roi_image_layer.reset_contrast_limits()
        
        # Load annotation points from the database, if it exists
        if os.path.exists(db_path_str):
            try:
                nodes = read_nodes(db_path_str)
            except Exception as e:
                show_info(f"Error reading database: {e}")
                return
            
            points = []
            for node in nodes:
                coord = node['coord']
                # Check if the point is within the ROI bounds
                if all(o <= c < b for o, c, b in zip(origin, coord, roi_bounds)):
                    points.append(coord)
            
            if points:
                self.roi_points_layer.data = np.array(points)
            else:
                self.roi_points_layer.data = np.empty((0, 3))
                show_info("No annotation points found in the specified ROI.")
        else:
            show_info("Database path does not exist. No annotation points loaded.")
            self.roi_points_layer.data = np.empty((0, 3))
        
        # Adjust the camera view to center on the ROI
        self.viewer.camera.center = [o + s / 2 for o, s in zip(origin, size)]
        show_info("ROI loaded successfully.")