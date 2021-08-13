cmake_minimum_required(VERSION 3.5)
project(yolov4_trt_ros)

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

set(util_src src/display.cpp src/yolo_with_plugins.cpp)
set(PLUGIN_SOURCES src/yolo_layer.cu)

find_package(OpenCV REQUIRED)
find_package(jetson-utils REQUIRED)
find_package(jetson-inference REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_generation REQUIRED)
find_package(rclpy REQUIRED)

file(COPY ${PROJECT_SOURCE_DIR}/lib/yaml-cpp/libyaml-cpp.so.0.7 DESTINATION ${PROJECT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/lib/yolo/yolo_layer.cu DESTINATION ${PROJECT_BINARY_DIR})

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

####################ADD_TARGET####################

add_executable(${PROJECT_NAME}  src/trt_yolo_v4.cpp ${util_src})
CUDA_ADD_LIBRARY(TrtNet STATIC ${PROJECT_BINARY_DIR}/yolo_layer.cu)

####################TARGET_LINK####################

target_include_directories(trt_yolo_v4 PRIVATE
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

target_link_libraries(trt_yolo_v4 
    ${ament_LIBRARIES}
    ${OpenCV_LIBS} 
    jetson-inference 
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins/libyolo_layer.so 
    cuda 
    nvinfer 
    TrtNet
    
)


# add_message_files(
#   FILES
#   Detector2D.msg
#   Detector2DArray.msg
#   BoundingBox2D.msg
#   BoundingBox.msg
#   BoundingBoxes.msg
#   ObjectHypothesis.msg
# )

# generate_messages(DEPENDENCIES std_msgs vision_msgs)
	 
# catkin_package(
#   CATKIN_DEPENDS nodelet rospy image_transport sensor_msgs vision_msgs cv_bridge message_runtime
# )

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# enable c++11 (TensorRT requirement)
# #set(CMAKE_BUILD_TYPE Debug)
# include_directories(${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
# add_definitions(-DROS1)

# set(PLUGIN_SOURCES src/yolo_layer.cu)

# set(common_src src/image_converter.cpp src/ros_compat.cpp src/usb_name.cpp)
# set(util_src src/display.cpp src/yolo_with_plugins.cpp)
# set(demo_src src/http_stream.cpp src/image_opencv.cpp src/common_struct.cpp)

# add_executable(video_source src/node_video_source.cpp ${common_src})
# add_executable(trt_yolo_v4 src/trt_yolo_v4.cpp ${util_src})
# add_executable(demo_view src/demo_view.cpp ${demo_src})
# add_dependencies(trt_yolo_v4 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# CUDA_ADD_LIBRARY(TrtNet STATIC ${PLUGIN_SOURCES})
# target_link_libraries(video_source ${catkin_LIBRARIES} jetson-inference)
# target_link_libraries(trt_yolo_v4 ${catkin_LIBRARIES} ${OpenCV_LIBS} jetson-inference ${CMAKE_CURRENT_SOURCE_DIR}/plugins/libyolo_layer.so cuda nvinfer TrtNet)
# target_link_libraries(demo_view ${catkin_LIBRARIES} ${OpenCV_LIBS})

# #target_link_libraries(trt_yolo_v4 ${catkin_LIBRARIES} yaml-cpp)
# link_directories(/usr/local/lib)
# include_directories(/usr/local/include/yaml-cpp)
# FIND_LIBRARY(YAML_CPP_LIBRARIES yaml-cpp)
# target_link_libraries( trt_yolo_v4  ${YAML_CPP_LIBRARIES})

# catkin_install_python(PROGRAMS trt_yolo_v4.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

ament_package()