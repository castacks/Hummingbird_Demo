cmake_minimum_required(VERSION 3.14)
project(ransac_pybind)

# Ensure Release build for performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# High-performance compile options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -march=native -ffast-math)
endif()

# Required packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Install Python package (ensure ransac_pybind/ has __init__.py)
ament_python_install_package(${PROJECT_NAME})

# Build the Pybind11 module (name must match Python import name)
pybind11_add_module(ransac_bindings src/ransac_bindings.cpp)

# Include and link C++ libraries
target_include_directories(ransac_bindings PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ransac_bindings
    PRIVATE
    ${OpenCV_LIBS}
)

# Install the compiled .so inside the Python package directory
install(TARGETS ransac_bindings
    LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
)

ament_package()