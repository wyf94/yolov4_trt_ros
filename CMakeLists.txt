cmake_minimum_required(VERSION 3.5)
project(yolov4_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -Wall -g -Wno-sign-compare -pthread -Wno-unused-variable -Wno-unused-but-set-variable")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN/'" )
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN/'" )
#set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN/'" )
set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}")
set(LIBRARY_BUILD_TYPE "STATIC")

set(CUDA_LIB_PATH "/usr/local/cuda-10.2/targets/aarch64-linux/lib")

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

############ADD_MESSAGE########################

set(msg_files 
    "msg/Detector2D.msg"
    "msg/Detector2DArray.msg"
    "msg/BoundingBox2D.msg"
    "msg/BoundingBox.msg"
    "msg/BoundingBoxes.msg"
    "msg/ObjectHypothesis.msg"
)

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

############ADD_CUDA_LIB########################

set(__cuda_arch_bin "72")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -std=c++11 -Xcompiler -fPIC -arch=compute_72")

cuda_add_library(TrtNet SHARED src/yolo_layer.cu)

target_include_directories(TrtNet PRIVATE
    include
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(TrtNet
    cuda 
    nvinfer 
)

############ADD_LIB########################

add_library(${PROJECT_NAME} SHARED src/${PROJECT_NAME}.cpp src/display.cpp src/yolo_with_plugins.cpp)

add_executable(${PROJECT_NAME}_node nodes/${PROJECT_NAME}_node.cpp)

############TARGET_LINK#####################

target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${ament_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_include_directories(${PROJECT_NAME}_node PRIVATE
    include
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

ament_target_dependencies(${PROJECT_NAME}_node
    rclcpp
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
    ${PROJECT_BINARY_DIR}/libyaml-cpp.so.0.7 
    cuda 
    nvinfer 
    TrtNet
)

target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
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

install(TARGETS ${PROJECT_NAME}_node ${PROJECT_NAME} TrtNet
    RUNTIME DESTINATION lib/${PROJECT_NAME}
    ARCHIVE DESTINATION lib/${PROJECT_NAME}
    LIBRARY DESTINATION lib/${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
