#include <opencv2/opencv.hpp>
#include <zed/Camera.hpp>

#include <exception>

StereoCamera::StereoCamera()
: image_left(images[0])
, image_right(images[1]) {
    sl::zed::InitParams params; // default MODE is PERFORMANCE
    // sl::zed::InitParams params(sl::zed::MODE::MEDIUM);
    // sl::zed::InitParams params(sl::zed::MODE::QUALITY);
    sl::zed::ERRCODE err = zed.init(params);
    if (err != sl::zed::SUCCESS) {
        std::cerr << "Error code: " << sl::zed::errcode2str(err) << std::endl;
        throw std::exception();
    }
    std::cout << "Camera initialized with ZED SDK" << std::endl;
    zed.setDepthClampValue(5000); // 
    zed.setConfidenceThreshold(100); // number in [1,100], controls density
    sl::zed::Camera::sticktoCPUCore(2);

    height = zed.getImageSize().height;
    width =  zed.getImageSize().width;
    image_size = cv::Size(height, width);
    for (image_index i = 0; i < 2; ++i) {
        images[i] =  cv::Mat(height, width, CV_8UC4);
    }
}

bool StereoCamera::read() {
    bool res = zed.grab();
    if (res) {
        slMat2cvMat(zed.retrieveImage(sl::zed::LEFT)).copyTo(image_left);
        slMat2cvMat(zed.retrieveImage(sl::zed::RIGHT)).copyTo(image_right);
    slMat2cvMat(zed.normalizeMeasure(sl::zed::MEASURE::DEPTH))
        .copyTo(depth_map);
    // slMat2cvMat(zed.normalizeMeasure(sl::zed::MEASURE::CONFIDENCE))
    //     .copyTo(confidence_map);
        // image_left = slMat2cvMat(zed.retrieveImage(sl::zed::LEFT));
        // image_right = slMat2cvMat(zed.retrieveImage(sl::zed::RIGHT));
    }
    return res;
}

void StereoCamera::calibrate() {
    // the SDK continuously self calibrates the camera in the background
    zed.resetSelfCalibration();
}
