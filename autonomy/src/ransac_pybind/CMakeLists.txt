cmake_minimum_required(VERSION 3.14)
project(ransac_pybind)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

ament_python_install_package(${PROJECT_NAME})

pybind11_add_module(ransac_pybind src/ransac_bindings.cpp)

target_include_directories(ransac_pybind PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ransac_pybind
    PRIVATE
    ${OpenCV_LIBS}
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optional: Add high-performance compile options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -march=native -ffast-math)
endif()

install(TARGETS ransac_pybind
        LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})

ament_package()