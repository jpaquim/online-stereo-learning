#include <opencv2/opencv.hpp>

#include <exception>
#include <iostream>

StereoCamera::StereoCamera()
: cap(0)
    // height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // width = cap.get(cv::CAP_PROP_FRAME_WIDTH)/2;
, height(cap.get(CV_CAP_PROP_FRAME_HEIGHT))
, width(cap.get(CV_CAP_PROP_FRAME_WIDTH)/2) {
    if (!cap.isOpened()) {
        std::cerr << "Error initializing camera" << std::endl;
        throw std::exception();
    }
    std::cout << "Camera initialized without ZED SDK" << std::endl;

    // side_by_side = cv::Mat(height, 2*width, CV_8UC4);
    side_by_side = cv::Mat::ones(height, 2*width, CV_8UC4);
    cap.read(side_by_side); // make side_by_side point to the right data
    // access left and right images as sub-matrices (data not copied)
    image_left  = side_by_side(cv::Rect(0,     0, width, height));
    image_right = side_by_side(cv::Rect(width, 0, width, height));
    depth_map = cv::Mat(height, width, CV_8UC4);
}

bool StereoCamera::read() {
    return cap.read(side_by_side);
    // bool res = cap.read(side_by_side);
    // return res;
}

void StereoCamera::calibrate() {
    // char key = ' ';
    // for () {
        // grab();
        // key = cv::waitKey(1);
    // }
}
