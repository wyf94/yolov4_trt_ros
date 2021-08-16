#include "yolov4_trt.h"

Yolov4::Yolov4(rclcpp::NodeOptions const& options) : rclcpp::Node{ "yolov4_trt_node", options }
{
    RCLCPP_INFO(this->get_logger(), "[%s]: Initializing...", this->get_name());

    init_params();

    init_yolo();

    trt_yolo.trt_init(model_path + model, h, w, category_num);

    //Image topic 0 and 1
    if (boxes_0_pub)
    {
        detection_publisher_0 = this->create_publisher<yolov4_trt::msg::BoundingBoxes>("bounding_boxes_0", 1);
    }
    if (boxes_1_pub)
    {
        detection_publisher_1 = this->create_publisher<yolov4_trt::msg::BoundingBoxes>("bounding_boxes_1", 1);
    }
    if (boxes_nms_pub)
    {
        detection_publisher_nms = this->create_publisher<yolov4_trt::msg::BoundingBoxes>("bounding_boxes_nms", 1);
    }
    if (result_img_pub)
    {
        image_nms_publisher = image_transport::create_publisher(this, "output_image_nms", custom_qos);
    }

    image_sub_0 = image_transport::create_subscription(this, video_topic_0, std::bind(&Yolov4::imageCallback, this, std::placeholders::_1), "raw", custom_qos);
    image_sub_1 = image_transport::create_subscription(this, video_topic_1, std::bind(&Yolov4::imageCallback, this, std::placeholders::_1), "raw", custom_qos);

    // // image_sub_0 = new ImageSubscriber(it, video_topic_0, 1, hints);
    // // image_sub_1 = new ImageSubscriber(it, video_topic_1, 1, hints);
    // // image_sub_0 = new ImageSubscriber(it, video_topic_0, 1);
    // // image_sub_1 = new ImageSubscriber(it, video_topic_1, 1);
    // // sync = new Sync(syncPolicy(20), *image_sub_0, *image_sub_1);
    // // sync->registerCallback(boost::bind(&Yolov4::imageSyncCallback, this, _1, _2));
    // const boost::function<void(const sensor_msgs::ImageConstPtr&)> f(boost::bind(&Yolov4::imageCallback, this, _1));
    // image_sub = it.subscribe(video_topic_1, 1, f, ros::VoidPtr(), hints);
    // // image_sub= it.subscribe(video_topic_1,1,f);
}

void Yolov4::init_params()
{
    // std::string package_path = ros::package::getPath("yolov4_trt_ros");
    // image_transport::TransportHints hints("compressed");

    getParameter("topic_name_0", video_topic_0);
    getParameter("topic_name_1", video_topic_1);
    getParameter("model", model);
    getParameter("model_path", model_path);
    getParameter("input_shape", input_shape);
    getParameter("category_number", category_num);
    getParameter("confidence_threshold", conf_th);
    getParameter("show_image", show_img);
    getParameter("nms_iou_threshold", iou_th);
    getParameter("match_yaml_path", yaml_path);
    getParameter("bounding_boxes_0_publisher", boxes_0_pub);
    getParameter("bounding_boxes_1_publisher", boxes_1_pub);
    getParameter("bounding_boxes_nms_publisher", boxes_nms_pub);
    getParameter("dectec_result_img_publisher", result_img_pub);

    match_conf = YAML::LoadFile(yaml_path);
    match_factor = match_conf["factor"].as<float>();
    match_x = match_conf["x"].as<int>();
    match_y = match_conf["y"].as<int>();
    std::cout << "match_factor:" << match_factor << std::endl;
    std::cout << "match_x:" << match_x << std::endl;
    std::cout << "match_y:" << match_y << std::endl;

    custom_qos = rmw_qos_profile_default;
}

void Yolov4::init_yolo()
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

void Yolov4::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_0)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64 start_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    std::cout << "diff time: " << start_time - last_time << std::endl;
    last_time = start_time;

    rclcpp::Time begin;

    cv_bridge::CvImageConstPtr cv_ptr_0;
    cv::Mat img_0;
    try
    {
        img_0 = cv_bridge::toCvShare(msg_0, "bgr8")->image;
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Unable to get '%s' image for send: '%s'", msg_0->encoding.c_str(), e.what());
    }

    // yolo detect and publish the boxes of image 0/1 
    if (cv_ptr_0 != nullptr)
    {
        //preprocess,detection,postprocess
        trt_yolo.detect(img_0, box_result_0, conf_th);
    }
    if (boxes_0_pub)
    {
        // publish the bounding boxes of image topic 1
        bboxes_publisher(box_result_0, detection_publisher_0, begin);
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

    if (result_img_pub)
    {
        std_msgs::msg::Header img_header;
        img_header.stamp = begin;
        img_header.frame_id = "image";
        msg = cv_bridge::CvImage(img_header, "bgr8", img_0).toImageMsg();
        image_nms_publisher.publish(msg);
    }

    if (show_img)
    {
        drawImage_0 = img_0.clone();
        //Draw boxes in  image
        for (int i = 0; i < box_result_0.size(); i++)
        {
            vis.draw_bboxes(drawImage_0, box_result_0[i]);
        }
        cv::resize(drawImage_0, drawImage_0, cv::Size(), 0.6, 0.6, CV_INTER_CUBIC);
        cv::namedWindow("YOLOv4 DETECTION RESULTS", cv::WINDOW_NORMAL);
        cv::imshow("YOLOv4 DETECTION RESULTS", drawImage_0);
        cv::waitKey(1);
    }
}

void Yolov4::imageSyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_0, const sensor_msgs::msg::Image::ConstSharedPtr& msg_1)
{
    auto start = std::chrono::high_resolution_clock::now();

    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64 start_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    std::cout << "diff time: " << start_time - last_time << std::endl;
    // std::cout<<"last time: "<<last_time<<std::endl;
    last_time = start_time;

    rclcpp::Time begin;

    // format: ros-msg 2 format: cv-Mat
    cv_bridge::CvImageConstPtr cv_ptr_0, cv_ptr_1;
    cv::Mat img_0, img_1;
    //convert msg to mat image
    try
    {
        img_0 = cv_bridge::toCvShare(msg_0, "bgr8")->image;
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Unable to get '%s' image for send: '%s'", msg_0->encoding.c_str(), e.what());
    }
    try
    {
        img_1 = cv_bridge::toCvShare(msg_1, "bgr8")->image;
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Unable to get '%s' image for send: '%s'", msg_1->encoding.c_str(), e.what());
    }

    // yolo detect and publish the boxes of image 0/1 
    if (cv_ptr_0 != nullptr)
    {
        //preprocess,detection,postprocess
        trt_yolo.detect(img_0, box_result_0, conf_th);
    }
    if (boxes_0_pub)
    {
        // publish the bounding boxes of image topic 1
        bboxes_publisher(box_result_0, detection_publisher_0, begin);
    }
    if (cv_ptr_1 != nullptr)
    {
        //preprocess,detection,postprocess
        trt_yolo.detect(img_1, box_result_1, conf_th);
    }
    if (boxes_1_pub)
    {
        // publish the bounding boxes of image topic 1
        bboxes_publisher(box_result_1, detection_publisher_1, begin);
    }

    struct timeval tz;
    gettimeofday(&tz, NULL);
    int64 end_time = tz.tv_sec * 1000 + tz.tv_usec / 1000;

    if (show_img)
    {
        drawImage_0 = img_0.clone();
        //Draw boxes in  image
        for (int i = 0; i < box_result_0.size(); i++)
        {
            vis.draw_bboxes(drawImage_0, box_result_0[i]);
        }
        drawImage_1 = img_1.clone();
        //Draw boxes in  image
        for (int i = 0; i < box_result_1.size(); i++)
        {
            vis.draw_bboxes(drawImage_1, box_result_1[i]);
        }
    }

    //image's boxes zoom and translation
    translatePoint(box_result_1, match_factor, match_x, match_y);
    //merge the boxes of image topic 0 and image topic 1
    box_result_merge.assign(box_result_0.begin(), box_result_0.end());
    box_result_merge.insert(box_result_merge.end(), box_result_1.begin(), box_result_1.end());
    // std::cout << "boxes size 0:" << box_result_0.size() << "  boxes size 1:" << box_result_1.size() << "  boxes size merge:" << box_result_merge.size() << std::endl;

    if (show_img)
    {
        drawImage_merge = img_0.clone();
        //Draw boxes in  image
        for (int i = 0; i < box_result_merge.size(); i++)
        {
            vis.draw_bboxes(drawImage_merge, box_result_merge[i]);
        }
    }

    // Remove the duplicate boxes in the merge boxes vector by NMS
    Boxes_NMS boxes_nms;
    box_result_nms = boxes_nms.nms(box_result_merge, iou_th);
    if (boxes_nms_pub)
    {
        // publish the bounding boxes of image topic nms  
        bboxes_publisher(box_result_nms, detection_publisher_nms, begin);
        // std::cout << "boxes size nms:" << box_result_nms.size() << std::endl;
    }

    //Draw boxes in  image
    drawImage_nms = img_0.clone();
    for (int i = 0; i < box_result_nms.size(); i++)
    {
        vis.draw_bboxes(drawImage_nms, box_result_nms[i]);
    }

    struct timeval te;
    gettimeofday(&te, NULL);
    int64 det_time = te.tv_sec * 1000 + te.tv_usec / 1000;
    // std::cout<<"trans format time: "<<end_time - start_time<<std::endl;
    // std::cout<<"detect + nms time: "<<det_time - end_time<<std::endl;

    cv::Point p_start = cv::Point(10, 100);
    std::string str_start = "Befor detection: " + std::to_string(start_time);
    cv::putText(img_0, str_start, p_start, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
    cv::Point p_end = cv::Point(10, 150);
    std::string str_end = "After detection: " + std::to_string(end_time);
    // std::cout<<"start_time: "<<str_start<<"  end_time: "<<str_end<<std::endl;
    cv::putText(img_0, str_end, p_end, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);

    if (result_img_pub)
    {
        std_msgs::msg::Header img_header;
        img_header.stamp = begin;
        img_header.frame_id = "image";
        msg = cv_bridge::CvImage(img_header, "bgr8", img_0).toImageMsg();
        image_nms_publisher.publish(msg);
    }

    //fps calculation
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    int64_t time_us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    double fps = 1.0 / ((double)time_us / 1000000);

    // std::cout << "Subscribe image:" << count << "    fps:" << fps << std::endl;
    count++;

    if (show_img)
    {
        drawImage_0 = util::show_fps(drawImage_0, fps);
        cv::resize(drawImage_0, drawImage_0, cv::Size(), 0.6, 0.6, CV_INTER_CUBIC);
        cv::namedWindow("YOLOv4 DETECTION RESULTS", cv::WINDOW_NORMAL);
        cv::imshow("YOLOv4 DETECTION RESULTS", drawImage_0);
        cv::waitKey(1);

        drawImage_1 = util::show_fps(drawImage_1, fps);
        cv::resize(drawImage_1, drawImage_1, cv::Size(), 0.6, 0.6, CV_INTER_CUBIC);
        cv::namedWindow("YOLOv4_1 DETECTION RESULTS", cv::WINDOW_NORMAL);
        cv::imshow("YOLOv4_1 DETECTION RESULTS", drawImage_1);
        cv::waitKey(1);

        drawImage_merge = util::show_fps(drawImage_merge, fps);
        cv::resize(drawImage_merge, drawImage_merge, cv::Size(), 0.6, 0.6, CV_INTER_CUBIC);
        cv::namedWindow("YOLOv4_0_1 DETECTION merge", cv::WINDOW_NORMAL);
        cv::imshow("YOLOv4_0_1 DETECTION merge", drawImage_merge);
        cv::waitKey(1);

        drawImage_nms = util::show_fps(drawImage_nms, fps);
        cv::resize(drawImage_nms, drawImage_nms, cv::Size(), 0.6, 0.6, CV_INTER_CUBIC);
        cv::namedWindow("YOLOv4_0_1 DETECTION NMS", cv::WINDOW_NORMAL);
        cv::imshow("YOLOv4_0_1 DETECTION NMS", drawImage_nms);
        cv::waitKey(1);
    }
}

// 把小图中boxes的点转化到大图中的点
void Yolov4::translatePoint(std::vector<util::Box>& box_result, float factor, int x, int y)
{
    for (int i = 0; i < box_result.size(); i++)
    {
        box_result[i].start_x = int(box_result[i].start_x * factor) + x;
        box_result[i].start_y = int(box_result[i].start_y * factor) + y;
        box_result[i].end_x = int(box_result[i].end_x * factor) + x;
        box_result[i].end_y = int(box_result[i].end_y * factor) + y;
    }
}

//publish box detection result
void Yolov4::bboxes_publisher(std::vector<util::Box> boxes, std::shared_ptr<rclcpp::Publisher<yolov4_trt::msg::BoundingBoxes>> det_pub, rclcpp::Time pub_time)
{
    yolov4_trt::msg::BoundingBoxes boxes_msg;
    boxes_msg.header.stamp = pub_time;
    boxes_msg.header.frame_id = "detection";
    for (int i = 0; i < boxes.size(); i++)
    {
        yolov4_trt::msg::BoundingBox bounding_box_msg;
        util::Box temp_box = boxes[i];
        bounding_box_msg.probability = temp_box.score;
        bounding_box_msg.xmin = (int)temp_box.start_x;
        bounding_box_msg.ymin = (int)temp_box.start_y;
        bounding_box_msg.xmax = (int)temp_box.end_x;
        bounding_box_msg.ymax = (int)temp_box.end_y;

        bounding_box_msg.id = i;
        bounding_box_msg.classes = util::classes_list[(int)temp_box.box_class];
        boxes_msg.bounding_boxes.push_back(bounding_box_msg);
    }
    det_pub->publish(boxes_msg);
}

template <class T>
bool Yolov4::getParameter(std::string parameterName, T& parameterDestination)
{
    const std::string param_path = "" + parameterName;
    this->declare_parameter(param_path);

    if (!this->get_parameter(param_path, parameterDestination))
    {
        RCLCPP_ERROR(this->get_logger(), "[%s]: Could not load param '%s'", this->get_name(), parameterName.c_str());
        return false;
    }
    else
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "[" << this->get_name() << "]: Loaded '" << parameterName << "' = '" << parameterDestination << "'");
    }

    return true;
}

template bool Yolov4::getParameter<int>(std::string parameterName, int& parameterDestination);
template bool Yolov4::getParameter<double>(std::string parameterName, double& parameterDestination);
template bool Yolov4::getParameter<float>(std::string parameterName, float& parameterDestination);
template bool Yolov4::getParameter<std::string>(std::string parameterName, std::string& parameterDestination);
template bool Yolov4::getParameter<bool>(std::string parameterName, bool& parameterDestination);
template bool Yolov4::getParameter<unsigned int>(std::string parameterName, unsigned int& parameterDestination);