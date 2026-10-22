import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'camera_droid'


def tree(src):
    """Install every file under `src` (keeps the folder structure)."""
    out = []
    for root, _, files in os.walk(src):
        if files:
            out.append((os.path.join('share', package_name, root),
                        [os.path.join(root, f) for f in files]))
    return out

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ] + tree('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Weed-spotting, weed-cutting camera droid for Gazebo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'area_patrol = camera_droid.area_patrol:main',
            'weed_detector = camera_droid.weed_detector:main',
        ],
    },
)
