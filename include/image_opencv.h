#ifndef IMAGE_OPENCV_H
#define IMAGE_OPENCV_H

//#include "image.h"
//#include "matrix.h"
//#include "darknet.h"
#include <iostream>

//#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <atomic>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

// includes for OpenCV >= 3.x
#include <opencv2/core/types.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "http_stream.h"
#include "common_struct.h"



image make_empty_image(int w, int h, int c);

cap_cv* get_capture_video_stream(const char *path);

mat_cv* get_capture_frame_cv(cap_cv *cap);
image get_image_from_stream_resize(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close);

#endif // IMAGE_OPENCV_H
