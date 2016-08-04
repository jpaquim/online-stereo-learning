#include "StereoCamera.h"

#include <opencv2/opencv.hpp>
 
#if USE_ZED_SDK
#   include "StereoCameraZED.inc"
#else
#   include "StereoCameraNoZED.inc"
#endif

int StereoCamera::getHeight() const {
    return height;
}

int StereoCamera::getWidth() const {
    return width;
}

const cv::Size &StereoCamera::getSize() const {
    return image_size;
}

const cv::Mat &StereoCamera::getImageLeft() const {
    return image_left;
}

const cv::Mat &StereoCamera::getImageRight() const {
    return image_right;
}

const cv::Mat &StereoCamera::getDepthMap() const {
    return depth_map;
}
