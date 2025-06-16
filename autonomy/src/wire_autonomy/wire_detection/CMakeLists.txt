cmake_minimum_required(VERSION 3.14)
project(wire_detection)

find_package(ament_cmake REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

pybind11_add_module(ransac_bindings src/ransac_bindings.cpp)

target_include_directories(ransac_bindings PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ransac_bindings
    PRIVATE
    ${OpenCV_LIBS}
)

install(TARGETS ransac_bindings
        LIBRARY DESTINATION lib/${PROJECT_NAME})

ament_package()