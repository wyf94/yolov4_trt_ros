#include "yolov4_trt.h"

int main(int argc, char** argv)
{
    // ros::init(argc, argv, "yolov4_detection", ros::init_options::AnonymousName);
    // ros::NodeHandle n("~");
    // Yolov4 Yolov4(n);
    // ros::spin();
    // return 0;

    rclcpp::init(argc, argv);

    auto yolov4 = std::make_shared<Yolov4>(rclcpp::NodeOptions{});

    rclcpp::spin(yolov4);

    rclcpp::shutdown();
}