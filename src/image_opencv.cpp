#include "image_opencv.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>


//using namespace cv;

using std::cerr;
using std::endl;


#ifndef CV_RGB
#define CV_RGB(r, g, b) cvScalar( (b), (g), (r), 0 )
#endif

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

#ifndef CV_AA
#define CV_AA cv::LINE_AA
#endif

image mat_to_image(cv::Mat mat);
image make_image(int w, int h, int c) {
    image out = make_empty_image(w,h,c);
    out.data = (float*)xcalloc(h * w * c, sizeof(float));
    return out;
}
int wait_for_stream(cap_cv *cap, cv::Mat* src, int dont_close) {
    if (!src) {
        if (dont_close) src = new cv::Mat(416, 416, CV_8UC(3)); // cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
        else return 0;
    }
    if (src->cols < 1 || src->rows < 1 || src->channels() < 1) {
        if (dont_close) {
            delete src;// cvReleaseImage(&src);
            int z = 0;
            for (z = 0; z < 20; ++z) {
                src = (cv::Mat*)get_capture_frame_cv(cap);
                delete src;// cvReleaseImage(&src);
            }
            src = new cv::Mat(416, 416, CV_8UC(3)); // cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
        } else return 0;
    }
    return 1;
}
image make_empty_image(int w, int h, int c) {
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}
cap_cv* get_capture_video_stream(const char *path) {
    cv::VideoCapture* cap = NULL;
    try {
        cap = new cv::VideoCapture(path);
    } catch (...) {
        cerr << " OpenCV exception: video-stream " << path << " can't be opened! \n";
    }
    return (cap_cv*)cap;
}
mat_cv* get_capture_frame_cv(cap_cv *cap) {
    cv::Mat *mat = NULL;
    try {
        mat = new cv::Mat();
        if (cap) {
            cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
            if (cpp_cap.isOpened()) {
                cpp_cap >> *mat;
            }
            //else std::cout << " Video-stream stopped! \n";
        } else cerr << " cv::VideoCapture isn't created \n";
    } catch (...) {
        //std::cout << " OpenCV exception: Video-stream stoped! \n";
    }
    return (mat_cv *)mat;
}
image mat_to_image(cv::Mat mat) {
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}
image get_image_from_stream_resize(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close) {
    c = c ? c : 3;
    cv::Mat *src = NULL;

    static int once = 1;
    if (once) {
        once = 0;
        do {
            if (src) delete src;
            src = (cv::Mat*)get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
        printf("Video stream: %d x %d \n", src->cols, src->rows);
    } else
        src = (cv::Mat*)get_capture_frame_cv(cap);

    if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);

    *(cv::Mat **)in_img = src;

    cv::Mat new_img = cv::Mat(h, w, CV_8UC(c));
    cv::resize(*src, new_img, new_img.size(), 0, 0, cv::INTER_LINEAR);
    if (c>1) cv::cvtColor(new_img, new_img, cv::COLOR_RGB2BGR);
    image im = mat_to_image(new_img);

    //show_image_cv(im, "im");
    //show_image_mat(*in_img, "in_img");
    return im;
}
// ----------------------------------------

