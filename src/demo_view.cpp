#include "demo_view.h"
#include <string.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cv_bridge/cv_bridge.h"
#include "ros/ros.h"
#include "ros/package.h"
#include "image_opencv.h"
#include "http_stream.h"

//Send jpeg through http stream.
class DemoView {
  private:
    ros::NodeHandle n_;
    image_transport::ImageTransport it;
    //video topic name
    std::string video_topic;
    image_transport::Subscriber image_sub;
  public:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg );
    void init_params();
    DemoView(const ros::NodeHandle& node_handle): n_(node_handle),it(node_handle) {
        init_params();
    }
};
void DemoView::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat img;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
        cv_ptr->image.copyTo(img);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    int port = 8090;
    int timeout = 400000;
    int jpeg_quality = 40;    // 1 - 100
    send_mjpeg(img, port, timeout, jpeg_quality);
}

void DemoView::init_params() {
    std::string package_path = ros::package::getPath("yolov4_trt_ros");
    n_.param<std::string>("/demo_view/topic_name",video_topic,"/usb_cam/image_raw");
    const boost::function<void(const sensor_msgs::ImageConstPtr&)> f(boost::bind(&DemoView::imageCallback,this,_1));
    image_sub= it.subscribe(video_topic,1,f);
}

int main(int argc, char** argv) {
    ros::init(argc,argv,"demo_view",ros::init_options::AnonymousName);
    ros::NodeHandle n("~");
    DemoView DemoView(n);
    ros::spin();
    return 0;
}
