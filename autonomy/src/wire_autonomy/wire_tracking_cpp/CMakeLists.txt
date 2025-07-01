cmake_minimum_required(VERSION 3.8)
project(wire_tracking_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)  
find_package(sensor_msgs REQUIRED)
find_package(wire_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cv_bridge REQUIRED)

set(SOURCES
    src/wire_tracking_node.cpp
    src/coord_transforms.cpp
    src/direction_kf.cpp
    src/position_kf.cpp
)

add_executable(wire_tracking_node ${SOURCES})

target_include_directories(wire_tracking_node
    PRIVATE
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(wire_tracking_node
    PRIVATE
    ${OpenCV_LIBS}
)

ament_target_dependencies(wire_tracking_node
    rclcpp
    sensor_msgs
    wire_interfaces
    geometry_msgs
    nav_msgs
    cv_bridge
)

install(TARGETS wire_tracking_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
    DESTINATION include/
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY rviz
    DESTINATION share/${PROJECT_NAME}/rviz
)
ament_package()
