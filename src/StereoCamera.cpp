#include "StereoCamera.h"

#include <opencv2/opencv.hpp>
 
#if USE_ZED_SDK
#   include "StereoCameraZED.inc"
#else
#   include "StereoCameraNoZED.inc"
#endif

int StereoCamera::getHeight() {
    return height;
}

int StereoCamera::getWidth() {
    return width;
}

cv::Mat &StereoCamera::getImageLeft() {
    return image_left;
}

cv::Mat &StereoCamera::getImageRight() {
    return image_right;
}

cv::Mat &StereoCamera::getDepthMap() {
    return depth_map;
}
