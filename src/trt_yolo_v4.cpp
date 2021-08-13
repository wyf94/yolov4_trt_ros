#include "trt_yolo_v4.h"

void Yolov4::imageCallback(const sensor_msgs::ImageConstPtr& msg_0)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64 start_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    std::cout << "diff time: " << start_time - last_time << std::endl;
    // std::cout<<"last time: "<<last_time<<std::endl;
    last_time = start_time;

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
        std_msgs::Header img_header;
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

void Yolov4::imageSyncCallback(const sensor_msgs::ImageConstPtr& msg_0, const sensor_msgs::ImageConstPtr& msg_1)
{
    auto start = std::chrono::high_resolution_clock::now();

    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64 start_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    std::cout << "diff time: " << start_time - last_time << std::endl;
    // std::cout<<"last time: "<<last_time<<std::endl;
    last_time = start_time;

    ros::Time begin = ros::Time::now();

    // format: ros-msg 2 format: cv-Mat
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
        std_msgs::Header img_header;
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
void Yolov4::bboxes_publisher(std::vector<util::Box> boxes, ros::Publisher det_pub, ros::Time pub_time)
{
    yolov4_trt_ros::BoundingBoxes boxes_msg;
    boxes_msg.header.stamp = pub_time;
    boxes_msg.header.frame_id = "detection";
    for (int i = 0; i < boxes.size(); i++)
    {
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
    det_pub.publish(boxes_msg);
}