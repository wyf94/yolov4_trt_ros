#ifndef YOLOV4_TRT_H
#define YOLOV4_TRT_H

#include <time.h>
#include <cmath>
#include <chrono>

#include "boost/function.hpp"
#include "boost/bind.hpp"

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include "yaml-cpp/yaml.h"

#include "yolov4_trt/msg/bounding_box2_d.hpp"
#include "yolov4_trt/msg/object_hypothesis.hpp"
#include "yolov4_trt/msg/detector2_d_array.hpp"
#include "yolov4_trt/msg/detector2_d.hpp"
#include "yolov4_trt/msg/bounding_box.hpp"
#include "yolov4_trt/msg/bounding_boxes.hpp"

#include "display.h"
#include "yolo_with_plugins.h"
#include "common.h"
#include "yolo_classes.h"
#include "visualization.h"

// typedef image_transport::SubscriberFilter ImageSubscriber;
// // typedef message_filters::Subscriber<sensor_msgs::Image> ImageSubscriber;
// typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
// typedef message_filters::Synchronizer<syncPolicy> Sync;

// typedef std::function<void(const sensor_msgs::ImageConstPtr& msg)> CallBackFunction;


class Yolov4 : public rclcpp::Node
{
private:
    template <class T>
    bool getParameter(std::string parameterName, T& parameterDestination);

    void init_params();
    void init_yolo();

    void bboxes_publisher(std::vector<util::Box> boxes, std::shared_ptr<rclcpp::Publisher<yolov4_trt::msg::BoundingBoxes>> det_pub, rclcpp::Time pub_time);
    void translatePoint(std::vector<util::Box>& box_result, float factor, int x, int y);

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_0);
    void imageSyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_0, const sensor_msgs::msg::Image::ConstSharedPtr& msg_1);

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

    // ImageSubscriber* image_sub_0, * image_sub_1;
    // Sync* sync;
    // ros::Publisher detection_publisher_0, detection_publisher_1, detection_publisher_nms;
    // image_transport::ImageTransport it;
    // image_transport::Publisher image_nms_publisher;
    // image_transport::Subscriber image_sub;
    // sensor_msgs::ImagePtr msg;
    std::vector<util::Box> box_result_0, box_result_1, box_result_merge, box_result_nms;
    cv::Mat drawImage_0, drawImage_1, drawImage_merge, drawImage_nms;

    YAML::Node match_conf;

    rmw_qos_profile_t custom_qos;
    std::shared_ptr<rclcpp::Publisher<yolov4_trt::msg::BoundingBoxes>> detection_publisher_0, detection_publisher_1, detection_publisher_nms;
    image_transport::Publisher image_nms_publisher;
    image_transport::Subscriber image_sub_0, image_sub_1;

    sensor_msgs::msg::Image::SharedPtr msg;

public:
    explicit Yolov4(rclcpp::NodeOptions const&);
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

#endif