ll
cd
ll
kk
ll
mkdir work
cd work/
ll
git clone https://github.com/opencv/opencv.git -b 4.8.0 --depth 1
git clone https://github.com/opencv/opencv_contrib.git -b 4.8.0 --depth 1
ll
cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D WITH_IPP=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_adas=OFF -D BUILD_opencv_bgsegm=OFF -D BUILD_opencv_bioinspired=OFF -D BUILD_opencv_ccalib=OFF -D BUILD_opencv_datasets=ON -D BUILD_opencv_datasettools=OFF -D BUILD_opencv_face=OFF -D BUILD_opencv_latentsvm=OFF -D BUILD_opencv_line_descriptor=OFF -D BUILD_opencv_matlab=OFF -D BUILD_opencv_optflow=ON -D BUILD_opencv_reg=OFF -D BUILD_opencv_saliency=OFF -D BUILD_opencv_surface_matching=OFF -D BUILD_opencv_text=OFF -D BUILD_opencv_tracking=ON -D BUILD_opencv_xobjdetect=OFF -D BUILD_opencv_xphoto=OFF -D BUILD_opencv_stereo=OFF -D BUILD_opencv_hdf=OFF -D BUILD_opencv_cvv=OFF -D BUILD_opencv_fuzzy=OFF -D BUILD_opencv_dnn=OFF -D BUILD_opencv_dnn_objdetect=OFF -D BUILD_opencv_dnn_superres=OFF -D BUILD_opencv_dpm=OFF -D BUILD_opencv_quality=OFF -D BUILD_opencv_rapid=OFF -D BUILD_opencv_rgbd=OFF -D BUILD_opencv_sfm=OFF -D BUILD_opencv_shape=ON -D BUILD_opencv_stitching=OFF -D BUILD_opencv_structured_light=OFF -D BUILD_opencv_alphamat=OFF -D BUILD_opencv_aruco=OFF -D BUILD_opencv_phase_unwrapping=OFF -D BUILD_opencv_photo=OFF -D BUILD_opencv_gapi=OFF -D BUILD_opencv_video=ON -D BUILD_opencv_ml=OFF -D BUILD_opencv_python2=OFF -D WITH_GSTREAMER=OFF -D ENABLE_PRECOMPILED_HEADERS=OFF -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ ../
make -j $(nproc)
sudo make install
sudo ldconfig
cd ../../


apt update
sudo apt-get install wget unzip
wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar zxf ceres-solver-2.1.0.tar.gz
ll
cd ceres-solver-2.1.0
mkdir build && cd build
cmake -DEXPORT_BUILD_DIR=ON         -DCMAKE_INSTALL_PREFIX=/usr/local         ../
sudo apt-get install -y cmake libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev
cmake -DEXPORT_BUILD_DIR=ON         -DCMAKE_INSTALL_PREFIX=/usr/local         ../
make -j $(nproc)
make test -j $(nproc)
sudo make install -j $(nproc)
cd ../../

cd ../../
ll
mkdir colcon_ws
cd c
cd colcon_ws/
mkdir src
cd src/
git clone -b humble https://github.com/ros-perception/vision_opencv.git
cd ../
colcon build --symlink-install
cd ../
sudo apt install libboost-python-dev
cd colcon_ws/
colcon build --symlink-install
ll
cd src/
git clone https://github.com/JanekDev/VINS-Fusion-ROS2-humble-arm.git
cd ../
colcon build --symlink-install
sudo apt install libboost-dev
colcon build --symlink-install
sudo apt -y install libboost-filesystem-dev
colcon build --symlink-install
sudo apt -y install libboost-program-options-dev
colcon build --symlink-install
cd ../
ll

wget -O eigen-3.4.0.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip 
ll
unzip eigen-3.4.0.zip 
ll

cd ../eigen-3.4.0
mkdir build && cd build
cmake ../ && sudo make install -j $(nproc)
ll
cd ../
ll
cd ../

ll
cd ceres-solver-2.1.0/build/
ll
cmake -DEXPORT_BUILD_DIR=ON         -DCMAKE_INSTALL_PREFIX=/usr/local         ../
make -j $(nproc) # number of cores
make test -j $(nproc)
sudo make install -j $(nproc)
cd ../../colcon_ws/
ll
rm build/ install/ log/ -rf
colcon build --symlink-install
sudo apt install ros-humble-image-transport
colcon build --symlink-install
source install/setup.bash 
clear
ros2 launch vins vins_rviz.launch.xml 
sudo apt install ros-humble-rviz2
clear
ros2 run vins vins_node 
ros2 run vins vins_node src/VINS-Fusion-ROS2-humble-arm/config/euroc/euroc_stereo_config.yaml 
history