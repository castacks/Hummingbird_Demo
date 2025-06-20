cmake_minimum_required(VERSION 3.8)
project(wire_interfaces)

# Enable warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Required build tools
find_package(ament_cmake REQUIRED)

# Dependencies for message generation
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Generate interfaces
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/WireDetection.msg"
  "msg/WireDetections.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

# Tests
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export package
ament_package()
find_package(rosidl_default_generators REQUIRED)
