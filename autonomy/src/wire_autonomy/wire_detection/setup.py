from setuptools import setup, find_packages
from glob import glob

package_name = 'wire_detection'  # This should match your package directory name

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name]),  # Automatically find packages
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/wire_detection.launch.xml']),  # If you have a launch file
        ('share/' + package_name + '/rviz', ['rviz/wire_detection.rviz']),
        ('share/' + package_name + '/config', ['config/wire_detection_config.yaml']),
        ('share/' + package_name + '/config', ['config/mask.png']),
        ('share/' + package_name + '/utils', ['wire_detection/wire_detection_utils.py']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='wire detection node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'wire_detection_node = wire_detection.wire_detection_node:main',  # Point to the main function
        ],
    },
)
