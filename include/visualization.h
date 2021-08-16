#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <math.h>
#include <random>
#include <algorithm>

namespace util
{
    std::vector<int> hsvToBgr(float h, float s, float v)
    {
        float r, g, b;
        float i = floor(h * 6);
        float f = h * 6 - i;
        float p = v * (1 - s);
        float q = v * (1 - f * s);
        float t = v * (1 - (1 - f) * s);

        switch (((int)i) % 6) {
        case 0: r = v;g = t;b = p;break;
        case 1: r = q;g = v;b = p;break;
        case 2: r = p;g = v;b = t;break;
        case 3: r = p;g = q;b = v;break;
        case 4: r = t;g = p;b = v;break;
        case 5: r = v;g = p;b = q;break;
        }
        std::vector<int> retVec;
        retVec.push_back(int(b * 255));
        retVec.push_back(int(g * 255));
        retVec.push_back(int(r * 255));
        return retVec;
    }

    std::vector<std::vector<int>> gen_colors(int num_colors)
    {
        std::vector<std::vector<float>> color_vec;
        for (int i = 0;i < num_colors;i++)
        {
            std::vector<float> temp_vec;
            temp_vec.push_back(float(i) / float(num_colors));
            temp_vec.push_back(1.0);
            temp_vec.push_back(0.7);
            color_vec.push_back(temp_vec);
        }
        random_shuffle(color_vec.begin(), color_vec.end());//vector容器测试
        std::vector<std::vector<int>> color_vec_int;
        for (auto i : color_vec)
        {
            std::vector<int> result = hsvToBgr(i[0], i[1], i[2]);
            color_vec_int.push_back(result);
        }
        return color_vec_int;
    }

    class BBoxVisualization
    {
    private:
        std::vector<std::vector<int>> colors;
    public:
        BBoxVisualization()
        {
            int length = sizeof(util::classes_list) / sizeof(util::classes_list[0]);
            colors = gen_colors(length);
        }
        cv::Mat& draw_boxed_text(cv::Mat& img, std::string text, float topleft_w, float topleft_h, std::vector<int> color)
        {
            int img_h = img.rows;
            int img_w = img.cols;
            if ((topleft_w >= img_w) || (topleft_h >= img_h))
            {
                return img;
            }
            int base_l = 1;
            cv::Size size = getTextSize(text, cv::FONT_HERSHEY_PLAIN, 1.0, 1, &base_l);
            int margin = 3;
            int h_r = size.height + margin * 2;
            int w_r = size.width + margin * 2;
            cv::Point origin;
            origin.x = margin + 1;
            origin.y = h_r - margin - 2;
            cv::Mat recMat(h_r, w_r, CV_8UC3, cv::Scalar(color[2], color[1], color[0]));
            cv::putText(recMat, text, origin, cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 1, cv::LINE_8);
            w_r = std::min(w_r, (int)(img_w - topleft_w));
            h_r = std::min(h_r, (int)(img_h - topleft_h));
            float alpha = 0.5;
            cv::Mat roi = img(cv::Range(topleft_h, topleft_h + h_r), cv::Range(topleft_w, topleft_w + w_r));
            cv::addWeighted(recMat(cv::Range(0, h_r), cv::Range(0, w_r)), alpha, roi, 1 - alpha, 0.0, roi);

        }

        void draw_bboxes(cv::Mat& img, util::Box box_result)
        {
            std::vector<int> bgrColor = colors[(int)box_result.box_class];
            cv::rectangle(img, cv::Point(box_result.start_x, box_result.start_y), cv::Point(box_result.end_x, box_result.end_y), cv::Scalar(bgrColor[2], bgrColor[1], bgrColor[0]), 2, 4);
            std::string class_name = util::classes_list[(int)box_result.box_class];
            char* chCode;
            chCode = new char[20];
            sprintf(chCode, "%.2lf", box_result.score);
            std::string str(chCode);
            delete[]chCode;
            std::string txt = class_name + str;
            float zero = 0;
            draw_boxed_text(img, txt, std::max(box_result.start_x + 2, zero), std::max(box_result.start_y + 2, zero), bgrColor);
        }
    };
}

#endif
