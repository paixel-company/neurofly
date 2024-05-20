from setuptools import setup

setup(
    name='ntools',
    version='0.1',
    packages=['ntools'],
    entry_points={
        'console_scripts': [
            'segs_annotator = ntools.segs_annotator:main',
            'simple_viewer = ntools.simple_viewer:main',
            'skel_annotator = ntools.skel_annotator:main',
            'seg = ntools.seger:command_line_interface',
            'aug = ntools.aug_segs:command_line_interface',
            'interp = ntools.interp_edges:command_line_interface'
        ]
    },
    requires = ["setuptools"]
)
