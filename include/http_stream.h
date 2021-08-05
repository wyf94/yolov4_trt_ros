#ifndef HTTP_STREAM_H
#define HTTP_STREAM_H
#include <stdint.h>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include "common_struct.h"

void send_mjpeg(cv::Mat& mat, int port, int timeout, int quality);
void temp_send_mjpeg(cap_cv* mat, int port, int timeout, int quality);

void this_thread_yield();


#endif // HTTP_STREAM_H
