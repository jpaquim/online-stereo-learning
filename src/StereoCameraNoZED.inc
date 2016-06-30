#include <opencv2/opencv.hpp>

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

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
    std::cout << "Starting camera calibration" << std::endl;
    std::cout << "Press 's' to save image, 'q' when done" << std::endl;

    float ratio = float(width)/float(height);
    const int display_height = 320;
    cv::Size display_size(ratio*display_height, display_height);
    cv::Mat display(display_size, CV_8UC4);

    std::vector<std::string> image_list;

    int image_num = 0;
    char key = ' ';
    while (key != 'q') {
        if (read()) {
            cv::resize(side_by_side, display, display_size);
            cv::imshow("Calibration", display);

            if (key == 's') {
                char filename_left[20];
                char filename_right[20];
                std::sprintf(filename_left, "left-%02d.png", image_num);
                std::sprintf(filename_right, "right-%02d.png", image_num);
                cv::imwrite(filename_left, image_left);
                std::cout << "saved " << filename_left << std::endl;
                cv::imwrite(filename_right, image_right);
                std::cout << "saved " << filename_right << std::endl;
                image_list.push_back(filename_left);
                image_list.push_back(filename_right);
                ++image_num;
            }
        }
        key = cv::waitKey(1);
    }
    std::vector<std::vector<cv::Point2f>> image_points[2];
    std::vector<std::vector<cv::Point3f>> object_points[2];
    image_points[0].resize(image_num);
    image_points[1].resize(image_num);
    cv::Size board_size(30, 30);

    for (size_t i = 0; i < image_num; ++i) {
        const std::string &filename = image_list[i*2];
        cv::Mat img = cv::imread(filename, 0);

        auto &corners = image_points[0][i];
        bool found = findChessboardCorners(image_left, board_size,
            corners);

        // cv::Mat image_corner(image_left.size(), CV_8UC4);
        // drawChessboardCorners(image_corner, board_size, corners, found);
        // cv::resize(image_corner, display, display_size);
        // cv::imshow("Calibration", display);
        // cv::waitKey(1);
    }

    cv::destroyWindow("Calibration");

    // cv::stereoCalibrate(object_points, image_points[0], image_points[1],
        // camera_matrix[0], dist_coeffs[0],
        // camera_matrix[1], dist_coeffs[1],
        // image_size, r, t, e);
}
