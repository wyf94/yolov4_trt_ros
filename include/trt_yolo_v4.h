#include <time.h>
#include <cmath>
#include <chrono>
#include "boost/function.hpp"
#include "boost/bind.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <ros/console.h>
#include "ros/ros.h"
#include "ros/package.h"
#include "cv_bridge/cv_bridge.h"

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
// #include <message_filters/synchronizer.h>
// #include <message_filters/sync_policies/approximate_time.h>
// #include <sensor_msgs/Image.h>
// #include "yolov4_trt_ros/Detector2DArray.h"
// #include "yolov4_trt_ros/Detector2D.h"
// #include "yolov4_trt_ros/BoundingBox.h"
// #include "yolov4_trt_ros/BoundingBoxes.h"

#include "yaml-cpp/yaml.h"
#include "display.h"
#include "yolo_with_plugins.h"
#include "common.h"
#include "yolo_classes.h"
#include "visualization.h"


typedef image_transport::SubscriberFilter ImageSubscriber;
// typedef message_filters::Subscriber<sensor_msgs::Image> ImageSubscriber;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
typedef message_filters::Synchronizer<syncPolicy> Sync;

typedef std::function<void(const sensor_msgs::ImageConstPtr& msg)> CallBackFunction;


class Yolov4
{
private:
    ros::NodeHandle n_;
    std::string video_topic_0;
    std::string video_topic_1;
    std::string model;
    std::string model_path;
    std::string yaml_path;
    std::string input_shape = "416";
    int w;
    int h;
    int count = 0;
    int category_num = 0;
    float conf_th = 0;
    float iou_th = 0;
    float match_factor = 0;
    int match_x = 0;
    int match_y = 0;
    bool show_img = false;
    bool boxes_0_pub = true;
    bool boxes_1_pub = true;
    bool boxes_nms_pub = true;
    bool result_img_pub = true;
    int64_t pro_start = 0;
    int callback_times = 0;
    std::chrono::high_resolution_clock::time_point t1;
    util::TrtYOLO trt_yolo;
    util::BBoxVisualization vis;

    int64 diff = 0;
    int64 last_time = 0;

    ImageSubscriber* image_sub_0, * image_sub_1;
    Sync* sync;
    ros::Publisher detection_publisher_0, detection_publisher_1, detection_publisher_nms;
    image_transport::ImageTransport it;
    image_transport::Publisher image_nms_publisher;
    image_transport::Subscriber image_sub;
    sensor_msgs::ImagePtr msg;
    std::vector<util::Box> box_result_0, box_result_1, box_result_merge, box_result_nms;
    cv::Mat drawImage_0, drawImage_1, drawImage_merge, drawImage_nms;

    YAML::Node match_conf;

    void init_params()
    {
        std::string package_path = ros::package::getPath("yolov4_trt_ros");
        image_transport::TransportHints hints("compressed");
        n_.param<std::string>("/yolov4_trt/topic_name_0", video_topic_0, "/image_source_0");
        n_.param<std::string>("/yolov4_trt/topic_name_1", video_topic_1, "/image_source_1");
        n_.param<std::string>("/yolov4_trt/model", model, "yolo3");
        n_.param<std::string>("/yolov4_trt/model_path", model_path, package_path + "/yolo/");
        n_.param<std::string>("/yolov4_trt/input_shape", input_shape, "416");
        n_.param<int>("/yolov4_trt/category_number", category_num, 80);
        n_.param<float>("/yolov4_trt/confidence_threshold", conf_th, 0.5);
        n_.param<bool>("/yolov4_trt/show_image", show_img, true);

        n_.param<std::string>("/yolov4_trt/match_yaml_path", yaml_path, "match_info.yaml");
        n_.param<float>("/yolov4_trt/nms_iou_threshold", iou_th, 0.8);
        n_.param<bool>("/yolov4_trt/bounding_boxes_0_publisher", boxes_0_pub, true);
        n_.param<bool>("/yolov4_trt/bounding_boxes_1_publisher", boxes_1_pub, true);
        n_.param<bool>("/yolov4_trt/bounding_boxes_nms_publisher", boxes_nms_pub, true);
        n_.param<bool>("/yolov4_trt/dectec_result_img_publisher", result_img_pub, true);

        match_conf = YAML::LoadFile(yaml_path);
        match_factor = match_conf["factor"].as<float>();
        match_x = match_conf["x"].as<int>();
        match_y = match_conf["y"].as<int>();
        std::cout << "match_factor:" << match_factor << std::endl;
        std::cout << "match_x:" << match_x << std::endl;
        std::cout << "match_y:" << match_y << std::endl;

        //Image topic 0 and 1
        if (boxes_0_pub)
        {
            detection_publisher_0 = n_.advertise<yolov4_trt_ros::BoundingBoxes>("/bounding_boxes_0", 1);
        }
        if (boxes_1_pub)
        {
            detection_publisher_1 = n_.advertise<yolov4_trt_ros::BoundingBoxes>("/bounding_boxes_1", 1);
        }
        if (boxes_nms_pub)
        {
            detection_publisher_nms = n_.advertise<yolov4_trt_ros::BoundingBoxes>("/bounding_boxes_nms", 1);
        }
        if (result_img_pub)
        {
            image_nms_publisher = it.advertise("/output_image_nms", 1);
        }

        // image_sub_0 = new ImageSubscriber(it, video_topic_0, 1, hints);
        // image_sub_1 = new ImageSubscriber(it, video_topic_1, 1, hints);
        // image_sub_0 = new ImageSubscriber(it, video_topic_0, 1);
        // image_sub_1 = new ImageSubscriber(it, video_topic_1, 1);
        // sync = new Sync(syncPolicy(20), *image_sub_0, *image_sub_1);
        // sync->registerCallback(boost::bind(&Yolov4::imageSyncCallback, this, _1, _2));
        const boost::function<void(const sensor_msgs::ImageConstPtr&)> f(boost::bind(&Yolov4::imageCallback, this, _1));
        image_sub = it.subscribe(video_topic_1, 1, f, ros::VoidPtr(), hints);
        // image_sub= it.subscribe(video_topic_1,1,f);
    }

public:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg_0);
    void bboxes_publisher(std::vector<util::Box> boxes, ros::Publisher det_pub, ros::Time pub_time);
    void imageSyncCallback(const sensor_msgs::ImageConstPtr& msg_0, const sensor_msgs::ImageConstPtr& msg_1);
    void translatePoint(std::vector<util::Box>& box_result, float factor, int x, int y);
    Yolov4(const ros::NodeHandle& node_handle) : n_(node_handle), it(node_handle)
    {
        init_params();
        init_yolo();
        trt_yolo.trt_init(model_path + model, h, w, category_num);
    }

    void init_yolo()
    {
        std::string yolo_dim;
        std::string dim_split;
        if (model.find('-') == std::string::npos)
        {
            model = model + "-" + input_shape;
            yolo_dim = model.substr((model.find_last_of('-') + 1));
        }
        if (yolo_dim.find('x') != std::string::npos)
        {
            if (yolo_dim.find_last_of('x') != yolo_dim.find_first_of('x'))
            {
                std::cout << "ERROR: bad yolo_dim " << yolo_dim << std::endl;
                exit(1);
            }
            else
            {
                int pos = yolo_dim.find('x');
                w = stoi(yolo_dim.substr(0, pos));
                h = stoi(yolo_dim.substr(pos));
            }
        }
        else
        {
            w = stoi(yolo_dim);
            h = w;
        }

        if ((h % 32 != 0) || (w % 32 != 0))
        {
            std::cout << "ERROR: bad yolo_dim " << yolo_dim << std::endl;
            exit(1);
        }
    }
};

class Boxes_NMS
{
public:
    static bool sort_score(util::Box box1, util::Box box2)
    {
        return box1.score > box2.score ? true : false;
    }

    float iou(util::Box box1, util::Box box2)
    {
        int x1 = std::max(box1.start_x, box2.start_x);
        int y1 = std::max(box1.start_y, box2.start_y);
        int x2 = std::min(box1.end_x, box2.end_x);
        int y2 = std::min(box1.end_y, box2.end_y);
        int w = std::max(0, x2 - x1 + 1);
        int h = std::max(0, y2 - y1 + 1);
        float over_area = w * h;
        return over_area / ((box1.end_x - box1.start_x) * (box1.end_y - box1.start_y) + (box2.end_x - box2.start_x) * (box2.end_y - box2.start_y) - over_area);
    }

    std::vector<util::Box> nms(std::vector<util::Box>& vec_boxs, float threshold)
    {
        std::vector<util::Box> results;
        std::sort(vec_boxs.begin(), vec_boxs.end(), sort_score);
        while (vec_boxs.size() > 0)
        {
            results.push_back(vec_boxs[0]);
            int index = 1;
            while (index < vec_boxs.size())
            {
                float iou_value = iou(vec_boxs[0], vec_boxs[index]);
                // std::cout << "iou:" << iou_value << std::endl;
                if (iou_value > threshold)
                    vec_boxs.erase(vec_boxs.begin() + index);
                else
                    index++;
            }
            vec_boxs.erase(vec_boxs.begin());
        }
        return results;
    }
};