cmake_minimum_required(VERSION 3.14)
project(ransac_bindings)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

ament_python_install_package(${PROJECT_NAME})

pybind11_add_module(ransac_bindings src/ransac_bindings.cpp)

target_include_directories(ransac_bindings PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ransac_bindings
    PRIVATE
    ${OpenCV_LIBS}
)

install(TARGETS ransac_bindings
        LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})

ament_package()