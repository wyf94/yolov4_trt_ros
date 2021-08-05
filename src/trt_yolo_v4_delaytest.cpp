#include <time.h>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <ros/console.h>
#include "ros/ros.h"
#include "ros/package.h"
#include "cv_bridge/cv_bridge.h"
#include "yaml-cpp/yaml.h"

#include "boost/function.hpp"
#include "boost/bind.hpp"

#include "yolov4_trt_ros/Detector2DArray.h"
#include "yolov4_trt_ros/Detector2D.h"
#include "yolov4_trt_ros/BoundingBox.h"
#include "yolov4_trt_ros/BoundingBoxes.h"

#include "display.h"
#include "yolo_with_plugins.h"
#include "common.h"
#include "yolo_classes.h"
#include "visualization.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
using namespace message_filters;
using namespace sensor_msgs;

typedef image_transport::SubscriberFilter ImageSubscriber;
// typedef message_filters::Subscriber<sensor_msgs::Image> ImageSubscriber;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
typedef message_filters::Synchronizer<syncPolicy> Sync;

ImageSubscriber *image_sub_0, *image_sub_1;
Sync *synca;
ros::Publisher detection_publisher_0, detection_publisher_1, detection_publisher_nms;
image_transport::Publisher image_nms_publisher;
sensor_msgs::ImagePtr msg;

void imageCalllback(const sensor_msgs::ImageConstPtr& msg_0)
{
	// ROS_INFO("Received \n");
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64 start_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;

    ros::Time begin = ros::Time::now();

    std::cout<<"received image successful."<<std::endl;

    cv_bridge::CvImagePtr cv_ptr_0;
    cv::Mat img_0;
    //convert msg to mat image
    try
    {
        cv_ptr_0 = cv_bridge::toCvCopy(msg_0, sensor_msgs::image_encodings::BGR8);
        cv_ptr_0->image.copyTo(img_0);
    }
    catch (cv_bridge::Exception& e_0)
    {
        ROS_ERROR("could not convert from '%s' to 'bgr8'.", msg_0->encoding.c_str());
    }

    struct timeval tz;
    gettimeofday(&tz, NULL);
    int64 end_time = tz.tv_sec * 1000 + tz.tv_usec / 1000;
    // std::cout<<"start_time: "<<start_time<<"  end_time: "<<end_time<<std::endl;
    cv::Point p_start = cv::Point(10, 100);
    std::string str_start = "Befor detection: " + std::to_string(start_time);
    cv::putText(img_0, str_start, p_start, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
    cv::Point p_end = cv::Point(10, 150);
    std::string str_end = "After detection: " + std::to_string(end_time);
    // std::cout<<"start_time: "<<str_start<<"  end_time: "<<str_end<<std::endl;
    cv::putText(img_0, str_end, p_end, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);

    if (1)
    {
        std_msgs::Header img_header;
        sensor_msgs::ImagePtr msg;
        img_header.stamp = begin;
        img_header.frame_id = "image";
        msg = cv_bridge::CvImage(img_header, "bgr8", img_0).toImageMsg();
        image_nms_publisher.publish(msg);
    }
}

void imageSyncCallback(const sensor_msgs::ImageConstPtr& msg_0, const sensor_msgs::ImageConstPtr& msg_1)
{
	// ROS_INFO("Received \n");
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64 start_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;

    ros::Time begin = ros::Time::now();

    cv_bridge::CvImagePtr cv_ptr_0;
    cv::Mat img_0;
    //convert msg to mat image
    try
    {
        cv_ptr_0 = cv_bridge::toCvCopy(msg_0, sensor_msgs::image_encodings::BGR8);
        cv_ptr_0->image.copyTo(img_0);
    }
    catch (cv_bridge::Exception& e_0)
    {
        ROS_ERROR("could not convert from '%s' to 'bgr8'.", msg_0->encoding.c_str());
    }
    cv_bridge::CvImagePtr cv_ptr_1;
    cv::Mat img_1;
    //convert msg to mat image
    try
    {
        cv_ptr_1 = cv_bridge::toCvCopy(msg_1, sensor_msgs::image_encodings::BGR8);
        cv_ptr_1->image.copyTo(img_1);
    }
    catch (cv_bridge::Exception& e_1)
    {
        ROS_ERROR("could not convert from '%s' to 'bgr8'.", msg_1->encoding.c_str());
    }

    struct timeval tz;
    gettimeofday(&tz, NULL);
    int64 end_time = tz.tv_sec * 1000 + tz.tv_usec / 1000;
    std::cout<<"format time: "<< end_time - start_time<<std::endl;

    cv::Point p_start = cv::Point(10, 100);
    std::string str_start = "Befor detection: " + std::to_string(start_time);
    cv::putText(img_0, str_start, p_start, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
    cv::Point p_end = cv::Point(10, 150);
    std::string str_end = "After detection: " + std::to_string(end_time);
    // std::cout<<"start_time: "<<str_start<<"  end_time: "<<str_end<<std::endl;
    cv::putText(img_0, str_end, p_end, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);

    if (1)
    {
        std_msgs::Header img_header;
        sensor_msgs::ImagePtr msg;
        img_header.stamp = begin;
        img_header.frame_id = "image";
        msg = cv_bridge::CvImage(img_header, "bgr8", img_0).toImageMsg();
        image_nms_publisher.publish(msg);
    }
    std::cout<<"received image successful."<<std::endl;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "image_listener");
	ros::NodeHandle n_;
    cv::namedWindow("video");
    cv::startWindowThread();
         // 定义节点句柄   
    image_transport::ImageTransport it_(n_);
    image_nms_publisher = it_.advertise("/output_image_nms", 1);
    ros::Rate loop_rate(5);

    // 单张图片测试
    // image_transport::Subscriber sub = it.subscribe( "/image_source_0", 1, imageCalllback);
    // image_transport::TransportHints hints("compressed");
    // image_transport::Subscriber image_sub= it.subscribe( "/image_source_0", 1, imageCalllback, ros::VoidPtr(), hints);

    // 两张图片测试
    // image_transport::TransportHints hints("compressed");
    // image_sub_0 = new ImageSubscriber(it_, "/image_source_0", 1, hints);
    // image_sub_1 = new ImageSubscriber(it_, "/image_source_1", 1, hints);
    image_sub_0 = new ImageSubscriber(it_, "/image_source_0", 1);
    image_sub_1 = new ImageSubscriber(it_, "/image_source_1", 1);
    synca = new Sync(syncPolicy(20), *image_sub_0, *image_sub_1);
    synca->registerCallback(boost::bind(&imageSyncCallback,_1, _2));

	    
	ros::spin();
    cv::destroyWindow("video");
	return 0;
}
