import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'camera_droid'

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
    ],
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
