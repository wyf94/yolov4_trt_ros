cmake_minimum_required(VERSION 3.0.2)
project(yolov4_trt_ros)

find_package(OpenCV 3.0 REQUIRED)
#find_package(jetson-utils REQUIRED)
#find_package(jetson-inference REQUIRED)
find_package(CUDA REQUIRED)

find_package(Boost REQUIRED COMPONENTS python)

catkin_python_setup()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/package.xml ${CMAKE_CURRENT_SOURCE_DIR}/package.xml COPYONLY)

find_package(catkin REQUIRED COMPONENTS
  image_transport
  rospy
  roscpp
  roscpp_serialization
  std_msgs
  sensor_msgs
  vision_msgs
  cv_bridge
  message_generation
)

add_message_files(
  FILES
  Detector2D.msg
  Detector2DArray.msg
  BoundingBox2D.msg
  BoundingBox.msg
  BoundingBoxes.msg
  ObjectHypothesis.msg
)

generate_messages(DEPENDENCIES std_msgs vision_msgs)
	 
catkin_package(
  CATKIN_DEPENDS nodelet rospy image_transport sensor_msgs vision_msgs cv_bridge message_runtime
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# enable c++11 (TensorRT requirement)
#set(CMAKE_BUILD_TYPE Debug)
include_directories(${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_definitions(-DROS1)

set(PLUGIN_SOURCES src/yolo_layer.cu)

set(common_src src/image_converter.cpp src/ros_compat.cpp src/usb_name.cpp)
set(util_src src/display.cpp src/yolo_with_plugins.cpp)
set(demo_src src/http_stream.cpp src/image_opencv.cpp src/common_struct.cpp)

#add_executable(video_source src/node_video_source.cpp ${common_src})
add_executable(trt_yolo_v4 src/trt_yolo_v4.cpp ${util_src})
add_executable(demo_view src/demo_view.cpp ${demo_src})
add_dependencies(trt_yolo_v4 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
CUDA_ADD_LIBRARY(TrtNet STATIC ${PLUGIN_SOURCES})
#target_link_libraries(video_source ${catkin_LIBRARIES} jetson-inference)
target_link_libraries(trt_yolo_v4 ${catkin_LIBRARIES} ${OpenCV_LIBS}  ${CMAKE_CURRENT_SOURCE_DIR}/plugins/libyolo_layer.so cuda nvinfer TrtNet)
target_link_libraries(demo_view ${catkin_LIBRARIES} ${OpenCV_LIBS})

target_link_libraries(trt_yolo_v4 ${catkin_LIBRARIES} yaml-cpp)
#link_directories(/usr/local/lib)
#include_directories(/usr/local/include/yaml-cpp)
#FIND_LIBRARY(YAML_CPP_LIBRARIES yaml-cpp)
#target_link_libraries( trt_yolo_v4  ${YAML_CPP_LIBRARIES})

catkin_install_python(PROGRAMS trt_yolo_v4.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
