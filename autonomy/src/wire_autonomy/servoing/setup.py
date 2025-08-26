from setuptools import setup, find_packages

package_name = 'servoing'  # This should match your package directory name

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name]),  # Automatically find packages
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/servoing.launch.xml']),  # If you have a launch file
        ('share/' + package_name + '/config', ['config/servoing_config.yaml']),  # If you have a config file
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A visual servoing ROS 2 package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'servoing_node = servoing.servoing_node:main',  # Point to the main function
        ],
    },
)
