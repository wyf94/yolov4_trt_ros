#include <time.h>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <ros/console.h>
#include "ros/ros.h"
#include "ros/package.h"
#include "cv_bridge/cv_bridge.h"

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

typedef std::function<void (const sensor_msgs::ImageConstPtr& msg)> CallBackFunction;
class Yolov4 {
  private:
    ros::NodeHandle n_;
    std::string video_topic;
    std::string model;
    std::string model_path;
    std::string input_shape="416";
    int w;
    int h;
    int count = 0;
    int category_num=0;
    float conf_th=0;
    bool show_img=false;
    int64_t pro_start = 0;
    int callback_times = 0;
    std::chrono::high_resolution_clock::time_point t1;
    util::TrtYOLO trt_yolo;
    image_transport::Subscriber image_sub;
    ros::Publisher detection_publisher;
    ros::Publisher detection_pub;
    image_transport::ImageTransport it;
    image_transport::Publisher overlay_pub;
    std::vector<util::Box> box_result;
    util::BBoxVisualization vis;
    void init_params() {
        std::string package_path = ros::package::getPath("yolov4_trt_ros");
        n_.param<std::string>("/yolov4_trt/topic_name",video_topic,"/image_source_0");
        n_.param<std::string>("/yolov4_trt/model",model, "yolo3");
        n_.param<std::string>("/yolov4_trt/model_path",model_path, package_path+"/yolo/");
        n_.param<std::string>("/yolov4_trt/input_shape",input_shape, "416");
        n_.param<int>("/yolov4_trt/category_number",category_num, 80);
        n_.param<float>("/yolov4_trt/confidence_threshold",conf_th, 0.5);
        n_.param<bool>("/yolov4_trt/show_image",show_img,true);
        const boost::function<void(const sensor_msgs::ImageConstPtr&)> f(boost::bind(&Yolov4::imageCallback,this,_1));
        image_sub= it.subscribe(video_topic,1,f);
        detection_pub= n_.advertise<yolov4_trt_ros::Detector2DArray>("detections",1);
        detection_publisher= n_.advertise<yolov4_trt_ros::BoundingBoxes>("/bounding_boxes",1);
        overlay_pub = it.advertise("/result/overlay",1);
    }

  public:
    void publisher(std::vector<util::Box> boxes);
    void bboxes_publisher(std::vector<util::Box> boxes);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg );
    Yolov4(const ros::NodeHandle& node_handle): n_(node_handle),it(node_handle) {
        init_params();
        init_yolo();
        trt_yolo.trt_init(model_path+model,h,w,category_num);
    }

    void init_yolo() {
        std::string yolo_dim;
        std::string dim_split;
        if(model.find('-')==std::string::npos) {
            model = model + "-" + input_shape;
            yolo_dim = model.substr((model.find_last_of('-')+1));
        }
        if(yolo_dim.find('x')!=std::string::npos) {
            if(yolo_dim.find_last_of('x')!=yolo_dim.find_first_of('x')) {
                std::cout<<"ERROR: bad yolo_dim "<<yolo_dim<<std::endl;
                exit(1);
            } else {
                int pos = yolo_dim.find('x');
                w = stoi(yolo_dim.substr(0,pos));
                h = stoi(yolo_dim.substr(pos));
            }
        } else {
            w =stoi(yolo_dim);
            h = w;
        }

        if ( (h%32 != 0 ) || (w%32 != 0)) {
            std::cout<<"ERROR: bad yolo_dim "<<yolo_dim<<std::endl;
            exit(1);
        }
    }
};

void Yolov4::imageCallback(const sensor_msgs::ImageConstPtr& msg ) {
    auto start = std::chrono::high_resolution_clock::now();
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat img;
    //convert msg to mat image
    try {
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
        cv_ptr->image.copyTo(img);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    cv::Mat drawImage;
    drawImage = img;
    if(cv_ptr != nullptr) {
	//preprocess,detection,postprocess
        trt_yolo.detect(img,box_result,conf_th);
    }
    
    for(int i =0; i<box_result.size(); i++) {
        vis.draw_bboxes(drawImage,box_result[i]);
    }
    //fps calculation
    auto elapsed= std::chrono::high_resolution_clock::now() - start;
    int64_t time_us  =std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();;
    double fps = 1.0/((double)time_us/1000000);
    //publisher(box_result); comment for not use
    bboxes_publisher(box_result);

     std::cout<<"Subscribe image:"<<count<<"    fps:"<<fps<<std::endl;
     count ++;

    if(show_img) {
        img = util::show_fps(img,fps);
        cv::namedWindow("YOLOv4 DETECTION RESULTS", cv::WINDOW_NORMAL);
        cv::imshow("YOLOv4 DETECTION RESULTS",img);
        cv::waitKey(1);
    }

    try {
        overlay_pub.publish(cv_ptr->toImageMsg());
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("could not convert from to 'bgr8'.");
    }

    if(show_img) {
        img = util::show_fps(drawImage,fps);
        cv::imshow("YOLOv4 DETECTION RESULTS",img);
        cv::waitKey(1);
    }
}

void Yolov4::publisher(std::vector<util::Box> boxes) {
    yolov4_trt_ros::Detector2DArray detection2d;
    yolov4_trt_ros::Detector2D detection;
    detection.header.stamp = ros::Time::now();
    detection.header.frame_id = "camera";
    for(int i =0; i <boxes.size(); i++) {
        util::Box temp_box = boxes[i];
        detection.results.id= temp_box.box_class;
        detection.results.score = temp_box.score;

        detection.bbox.center.x = ((float)(temp_box.end_x - temp_box.start_x))/2;
        detection.bbox.center.y = (float)((temp_box.end_y - temp_box.start_y))/2;
        detection.bbox.center.theta = 0.0;
        detection.bbox.size_x = abs(temp_box.start_x - temp_box.end_x);
        detection.bbox.size_y = abs(temp_box.start_y - temp_box.end_y);
        detection2d.detections.push_back(detection);
    }
    detection_pub.publish(detection2d);
}

//publish box detection result
void Yolov4::bboxes_publisher(std::vector<util::Box> boxes) {
    yolov4_trt_ros::BoundingBoxes boxes_msg;
    boxes_msg.header.stamp = ros::Time::now();
    boxes_msg.header.frame_id = "detection";
    for(int i =0; i<boxes.size(); i++) {
        yolov4_trt_ros::BoundingBox bounding_box_msg;
        util::Box temp_box = boxes[i];
        bounding_box_msg.probability = temp_box.score;
        bounding_box_msg.xmin = (int)temp_box.start_x;
        bounding_box_msg.ymin = (int)temp_box.start_y;
        bounding_box_msg.xmax = (int)temp_box.end_x;
        bounding_box_msg.ymax = (int)temp_box.end_y;

        bounding_box_msg.id = i;
        bounding_box_msg.Class = util::classes_list[(int)temp_box.box_class];
        boxes_msg.bounding_boxes.push_back(bounding_box_msg);
    }
    detection_publisher.publish(boxes_msg);
}
int main(int argc, char** argv) {
    ros::init(argc,argv,"yolov4_detection",ros::init_options::AnonymousName);
    ros::NodeHandle n("~");
    Yolov4 Yolov4(n);
    ros::spin();
    return 0;
}
