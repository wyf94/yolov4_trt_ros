#ifndef DISPLAY_H
#define DISPLAY_H

#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"

namespace util
{
    void open_window(const std::string window_name, std::string title, float width = 0, float height = 0);

    cv::Mat& show_help_text(cv::Mat& img, std::string help_text);

    void drawBox(cv::Mat& img, Box box_result);
    cv::Mat& show_fps(cv::Mat& img, float fps);

    void set_display(std::string window_name, bool full_scrn);
}
#endif
