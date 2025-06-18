from setuptools import setup

package_name = 'ransac_pybind'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': '.'},
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Pybind11 ROS 2 package for fast RANSAC inference',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)