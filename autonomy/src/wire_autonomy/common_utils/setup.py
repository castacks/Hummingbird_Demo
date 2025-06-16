from setuptools import find_packages, setup
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext

package_name = 'common_utils'

ext_modules = [
    Pybind11Extension(
        f"{package_name}.ransac_bindings",  # module name inside the package
        [f"{package_name}/ransac_bindings.cpp"],
        include_dirs=["/usr/include/eigen3"],
        libraries=["opencv_core", "opencv_imgproc", "opencv_highgui"],  # add more if needed
    ),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['config/wire_detection_config.yaml']),
        ('share/' + package_name, ['config/wire_tracking_config.yaml']),
        ('share/' + package_name + '/src', glob('common_utils/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tyler',
    maintainer_email='tharp@andrew.cmu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
)
