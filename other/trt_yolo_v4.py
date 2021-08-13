#!/usr/bin/python
import os
import time

import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda

from utils.yolo_classes import get_cls_dict
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

import rospy
import rospkg
from yolov4_trt_ros.msg import Detector2DArray
from yolov4_trt_ros.msg import Detector2D
from yolov4_trt_ros.msg import BoundingBox
from yolov4_trt_ros.msg import BoundingBoxes
from yolov4_trt_ros.msg import Detector2D
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class yolov4(object):
    def __init__(self):
        """ Constructor """

        self.bridge = CvBridge()
        self.init_params()
        self.init_yolo()
        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_yolo = TrtYOLO(
            (self.model_path + self.model), (self.h, self.w), self.category_num)

    def __del__(self):
        """ Destructor """

        self.cuda_ctx.pop()
        del self.trt_yolo
        del self.cuda_ctx

    def clean_up(self):
        """ Backup destructor: Release cuda memory """

        if self.trt_yolo is not None:
            self.cuda_ctx.pop()
            del self.trt_yolo
            del self.cuda_ctx

    def init_params(self):
        """ Initializes ros parameters """
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("yolov4_trt_ros")
        self.video_topic = rospy.get_param("yolov4_trt/topic_name", "/usb_cam/image_raw")
        # self.video_topic = rospy.get_param("/video_topic", "/video_source/raw")
        self.model = rospy.get_param("yolov4_trt/model", "yolov4")
        self.model_path = rospy.get_param(
            "yolov4_trt/model_path", package_path + "/yolo/")
        self.input_shape = rospy.get_param("yolov4_trt/input_shape", default = "416")
        #print(self.input_shape)
        #print(rospy.get_param("/category_number")
        self.category_num = rospy.get_param("yolov4_trt/category_number", 80)
        self.conf_th = rospy.get_param("yolov4_trt/confidence_threshold", 0.5)
        self.show_img = rospy.get_param("yolov4_trt/show_image",True)
        #print(self.show_img)
        self.image_sub = rospy.Subscriber(
            self.video_topic, Image, self.img_callback, queue_size=1, buff_size=1920*1080*3)
        self.detection_pub = rospy.Publisher(
            "yolov4_trt/detections", Detector2DArray, queue_size=1)

        self.detection_publisher = rospy.Publisher(
            "yolov4_trt/bounding_boxes", BoundingBoxes, queue_size=1)
        self.overlay_pub = rospy.Publisher(
            "/result/overlay", Image, queue_size=1)

    def init_yolo(self):
        """ Initialises yolo parameters required for trt engine """

        if self.model.find('-') == -1:
            self.model = self.model + "-" + self.input_shape
            yolo_dim = self.model.split('-')[-1]

        if 'x' in yolo_dim:
            dim_split = yolo_dim.split('x')
            if len(dim_split) != 2:
                raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
            self.w, self.h = int(dim_split[0]), int(dim_split[1])
        else:
            self.h = self.w = int(yolo_dim)
        if self.h % 32 != 0 or self.w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self.category_num)
        self.vis = BBoxVisualization(cls_dict)


    def img_callback(self, ros_img):
      #"""Continuously capture images from camera and do object detection """
      #config = Config()
      #graphviz = GraphvizOutput()
      #graphviz.output_file = 'graph_1.png'
      #with PyCallGraph(output=graphviz, config=config):

        tic = time.time()

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        if cv_img is not None:
            boxes, confs, clss = self.trt_yolo.detect(cv_img, self.conf_th)
            # print('boxes:',boxes)
            # print('confs:',confs)
            # print('clss:',clss)

        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        print("fps:",fps)

#        self.publisher(boxes, confs, clss)
        self.bboxes_publisher(boxes, confs, clss)

        if self.show_img:
            cv_img = show_fps(cv_img, fps)
            #cv2.namedWindow("YOLOv4 DETECTION RESULTS", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLOv4 DETECTION RESULTS", cv_img)
            cv2.waitKey(1)

        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="passthrough")
            rospy.logdebug("CV Image converted for publishing")
            self.overlay_pub.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))


    def bboxes_publisher(self,boxes,confs,clss):
        boxes_msg = BoundingBoxes()

        boxes_msg.header.stamp = rospy.Time.now()
        boxes_msg.header.frame_id = "detection" 
        

        for i in range(len(boxes)):
            bounding_box_msg = BoundingBox()

            bounding_box_msg.probability = confs[i]
            bounding_box_msg.xmin = int(boxes[i][0])
            bounding_box_msg.ymin = int(boxes[i][1])
            bounding_box_msg.xmax = int(boxes[i][2])
            bounding_box_msg.ymax = int(boxes[i][3])

            bounding_box_msg.id = i
            cls_dict = get_cls_dict(self.category_num)
            # print(class)
            cls_name = cls_dict.get(clss[i], 'CLS{}'.format(clss[i]))
            bounding_box_msg.Class = str(cls_name)
            boxes_msg.bounding_boxes.append(bounding_box_msg)
        self.detection_publisher.publish(boxes_msg)
        

    def publisher(self, boxes, confs, clss):
        """ Publishes to detector_msgs

        Parameters:
        boxes (List(List(int))) : Bounding boxes of all objects
        confs (List(double))	: Probability scores of all objects
        clss  (List(int))	: Class ID of all classes
        """
        detection2d = Detector2DArray()
        detection = Detector2D()

        for i in range(len(boxes)):
            # boxes : xmin, ymin, xmax, ymax
            for _ in boxes:
                detection.header.stamp = rospy.Time.now()
                detection.header.frame_id = "camera" # change accordingly
                detection.results.id = clss[i]
                detection.results.score = confs[i]

                detection.bbox.center.x = (boxes[i][2] - boxes[i][0])/2
                detection.bbox.center.y = (boxes[i][3] - boxes[i][1])/2
                detection.bbox.center.theta = 0.0  # change if required

                detection.bbox.size_x = abs(boxes[i][0] - boxes[i][2])
                detection.bbox.size_y = abs(boxes[i][1] - boxes[i][3])

            detection2d.detections.append(detection)

        self.detection_pub.publish(detection2d)


def main():
    yolo = yolov4()
    rospy.init_node('yolov4_detection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.on_shutdown(yolo.clean_up())
        print("Shutting down")



if __name__ == '__main__':
    main()
