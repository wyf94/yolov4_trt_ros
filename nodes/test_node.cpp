#include "trt_yolo_v4.cpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "yolov4_detection", ros::init_options::AnonymousName);
    ros::NodeHandle n("~");
    Yolov4 Yolov4(n);
    ros::spin();
    return 0;
}