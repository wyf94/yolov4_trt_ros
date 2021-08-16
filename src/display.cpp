#include "display.h"

namespace util {

    void open_window(const std::string& window_name, std::string title, float width, float height) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::setWindowTitle(window_name, title);
        if (width && height) {
            cv::resizeWindow(window_name, width, height);
        }
    }


    cv::Mat& show_help_text(cv::Mat& img, std::string help_text) {
        cv::Point origin;
        origin.x = 11;
        origin.y = 20;
        cv::putText(img, help_text, origin, cv::FONT_HERSHEY_PLAIN, 1.0,
            cv::Scalar(32, 32, 32), 4, cv::LINE_AA);
        origin.x = 10;
        cv::putText(img, help_text, origin, cv::FONT_HERSHEY_PLAIN, 1.0,
            cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
        return img;
    }

    cv::Mat& show_fps(cv::Mat& img, float fps) {
        char* chCode;
        chCode = new char[10];
        sprintf(chCode, "%.2f", fps);
        std::string fps_text(chCode);
        delete[] chCode;
        cv::Point origin;
        origin.x = 11;
        origin.y = 20;
        cv::putText(img, fps_text, origin, cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(32, 32, 32), 4, cv::LINE_AA);
        origin.x = 10;
        cv::putText(img, fps_text, origin, cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
        return img;
    }

    void drawBox(cv::Mat& img, util::Box box_result) {

        cv::rectangle(img, cv::Point(box_result.start_x, box_result.start_y), cv::Point(box_result.end_x, box_result.end_y), cv::Scalar(0, 255, 255), 2, 4);
        cv::waitKey(1);

    }

    void set_display(std::string window_name, bool full_scrn) {
        if (full_scrn) {
            cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN,
                cv::WINDOW_FULLSCREEN);
        }
        else {
            cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN,
                cv::WINDOW_NORMAL);
        }
    }

}
