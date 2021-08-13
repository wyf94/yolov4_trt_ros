cmake_minimum_required(VERSION 3.5)
project(yolov4_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -Wall -g -Wno-sign-compare -pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN/'" )
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN/'" )
#set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN/'" )
set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}")
set(LIBRARY_BUILD_TYPE "STATIC")

set(msg_files 
    "msg/Detector2D.msg"
    "msg/Detector2DArray.msg"
    "msg/BoundingBox2D.msg"
    "msg/BoundingBox.msg"
    "msg/BoundingBoxes.msg"
    "msg/ObjectHypothesis.msg"
)

find_package(OpenCV REQUIRED)
# find_package(jetson-utils REQUIRED)
# find_package(jetson-inference REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
#find_package(message_generation REQUIRED)
#find_package(rclpy REQUIRED)

file(COPY ${PROJECT_SOURCE_DIR}/lib/yaml-cpp/libyaml-cpp.so.0.7 DESTINATION ${PROJECT_BINARY_DIR})

rosidl_generate_interfaces(${PROJECT_NAME}_Message
    ${msg_files}
    DEPENDENCIES std_msgs geometry_msgs
)

ament_register_extension(
    "rosidl_generate_interfaces"
    "rosidl_generator_cpp"
    "rosidl_generator_cpp_generate_interfaces.cmake"
)

ament_export_dependencies(rosidl_default_runtime)

#find_package(${PROJECT_NAME}_Message REQUIRED)

############ADD_LIB########################

CUDA_ADD_LIBRARY(TrtNet STATIC src/yolo_layer.cu)

add_executable(${PROJECT_NAME} src/trt_yolo_v4.cpp src/display.cpp src/yolo_with_plugins.cpp)

############TARGET_LINK#####################

target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${ament_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_include_directories(TrtNet PRIVATE
    include
    ${CUDA_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME} 
    rclcpp
    rclcpp_lifecycle
    rclcpp_components
    rosidl_default_generators
    std_msgs
    geometry_msgs
    message_filters
    sensor_msgs
    image_transport
    cv_bridge
)

rosidl_target_interfaces(${PROJECT_NAME}  
    ${PROJECT_NAME}_Message 
    "rosidl_typesupport_cpp"
)

target_link_libraries(${PROJECT_NAME} 
    ${ament_LIBRARIES}
    ${OpenCV_LIBS} 
    #jetson-inference 
    #${CMAKE_CURRENT_SOURCE_DIR}/plugins/libyolo_layer.so 
    cuda 
    nvinfer 
    TrtNet
)

############Install##########################

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.yaml"
)

install(PROGRAMS 
    ${PROJECT_BINARY_DIR}/libyaml-cpp.so.0.7 
    DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION lib/${PROJECT_NAME}
    ARCHIVE DESTINATION lib/${PROJECT_NAME}
    LIBRARY DESTINATION lib/${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
