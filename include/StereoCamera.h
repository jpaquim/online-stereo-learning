#ifndef STEREO_CAMERA_H
#define STEREO_CAMERA_H

#include <opencv2/opencv.hpp>

#if USE_ZED_SDK
#   include <zed/Camera.hpp>
#endif

class StereoCamera {
public:
    // default constructor, initializes the ZED camera or VideoCapture
    // object, and the image buffers
    StereoCamera();

    // automatically generated destructor
    ~StereoCamera() = default;

    // returns the camera images' height in pixels
    int getHeight() const;

    // returns the camera images' width in pixels
    int getWidth() const;

    // returns the camera images' width in pixels
    const cv::Size &getSize() const;

    // returns the left camera image
    const cv::Mat &getImageLeft() const;

    // returns the left camera image
    const cv::Mat &getImageRight() const;

    // returns the depth map
    const cv::Mat &getDepthMap() const;

    // reads a frame from the camera, changing the contents of the image
    // buffers
    // returns true on success, false if reading fails
    bool read();

    // performs stereo camera calibration, either automatically (ZED), or
    // otherwise using a chessboard, populating the internal camera matrices
    // and distortion parameters.
    void calibrate();
    // TODO: finish function, and create function for image calibration,
    // given the parameters
private:
#if USE_ZED_SDK
    sl::zed::Camera zed;
#else
    cv::VideoCapture cap;
    cv::Mat side_by_side;
#endif // USE_ZED_SDK
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
