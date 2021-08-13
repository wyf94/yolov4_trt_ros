#!/bin/bash
camera="video_source"
result=`rosnode list`
choice=false
if [[ $result == *$camera* ]]
then
	choice=false
else
	choice=true
fi
roslaunch yolov4_trt_ros demo_view.launch foo:=$choice

