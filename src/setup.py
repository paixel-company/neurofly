from setuptools import setup

setup(
    name='ntools',
    version='0.1',
    packages=['ntools'],
    entry_points={
        'console_scripts': [
            'segs_annotator = ntools.segs_annotator:main_function',
            'simple_viewer = ntools.simple_viewer:main_function',
            'skel_annotator = ntools.skel_annotator:main_function'
        ]
    },
    requires = ["setuptools"],
    dependencies = [
        "brightest_path_lib==1.0.12",
        "h5py==3.7.0",
        "imageio==2.21.1",
        "magicgui==0.5.1",
        "napari==0.4.16",
        "networkx==2.8.5",
        "numpy==1.23.1",
        "scikit_image==0.19.3",
        "scipy==1.9.0",
        "setuptools==61.2.0",
        "tifffile==2022.8.8",
        "tqdm==4.64.0",
        "zarr==2.13.2"
    ]
)
