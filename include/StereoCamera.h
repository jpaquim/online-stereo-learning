#ifndef STEREO_CAMERA_H
#define STEREO_CAMERA_H

#include <opencv2/opencv.hpp>

#if USE_ZED_SDK
#   include <zed/Camera.hpp>
#endif

class StereoCamera {
public:
    StereoCamera();
    ~StereoCamera() = default;
    int getHeight();
    int getWidth();
    cv::Mat &getImageLeft();
    cv::Mat &getImageRight();
    cv::Mat &getDepthMap();
    bool read();
    void calibrate();
private:
#if USE_ZED_SDK
    sl::zed::Camera zed;
#else
    cv::VideoCapture cap;
    cv::Mat side_by_side;
#endif
    int height;
    int width;
    cv::Size image_size;
    cv::Mat images[2];
    cv::Mat &image_left;
    cv::Mat &image_right;
    cv::Mat depth_map;
    using image_index = uint8_t;
};

#endif
