#include <opencv2/opencv.hpp>

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

StereoCamera::StereoCamera()
: cap(0)
, height(cap.get(CV_CAP_PROP_FRAME_HEIGHT))
, width(cap.get(CV_CAP_PROP_FRAME_WIDTH)/2)
, image_size(width, height)
, image_left(images[0])
, image_right(images[1]) {
    if (!cap.isOpened()) {
        std::cerr << "Error initializing camera" << std::endl;
        throw std::exception();
    }
    std::cout << "Camera initialized without ZED SDK" << std::endl;

    // side_by_side = cv::Mat(height, 2*width, CV_8UC4);
    side_by_side = cv::Mat::ones(height, 2*width, CV_8UC4);
    cap.read(side_by_side); // make side_by_side point to the right data
    // access left and right images as sub-matrices (data not copied)
    images[0] = side_by_side(cv::Rect(0,     0, width, height));
    images[1] = side_by_side(cv::Rect(width, 0, width, height));
    depth_map = cv::Mat(height, width, CV_8UC4);
}

bool StereoCamera::read() {
    return cap.read(side_by_side);
    // bool res = cap.read(side_by_side);
    // return res;
}

void StereoCamera::calibrate() {
    std::cout << "Starting stereo camera calibration" << std::endl;
    std::cout << "Place the checkerboard inside the frame, "
                 "then press 's' to use it" << std::endl;
    std::cout << "Press 'q' when done" << std::endl;

    float ratio = float(width)/float(height);
    const int display_height = 320;
    cv::Size display_size(ratio*display_height, display_height);
    cv::Mat displays[2] = {{display_size, CV_8UC4},
                           {display_size, CV_8UC4}};
    char window_names[2][32];

    // cv::Size board_size(6, 4); // number of squares along each dimension
    cv::Size board_size(5, 3); // number of squares along each dimension
    float square_side = 60; // length of the square's side, in mm
    // object points are the positions of corners on the chessboard, as
    // measured from the chessboard's origin
    std::vector<cv::Point3f> board_points;
    for (int i = 0; i < board_size.width; ++i) {
        for (int j = 0; j < board_size.height; ++j) {
            cv::Point3f board_point(i*square_side, j*square_side, 0);
            board_points.push_back(board_point);
        }
    }

    std::vector<std::vector<cv::Point2f>> image_points[2];
    std::vector<std::vector<cv::Point3f>> object_points;

    int image_num = 0;
    char key = ' ';
    while (key != 'q') {
        if (!read()) {
            key = cv::waitKey(1);
            continue;
        }
        for (image_index i = 0; i < 2; ++i) {
            cv::resize(images[i], displays[i], display_size);
            std::sprintf(window_names[i], "Calibration %d", i);
            cv::imshow(window_names[i], displays[i]);
        }

        key = cv::waitKey(1); // flush window buffer and read key

        if (key == 's') {
            cv::Mat images_gray[2];
            std::vector<cv::Point2f> corners[2];
            bool found[2];

            for (image_index i = 0; i < 2; ++i) {
                images_gray[i] = cv::Mat(image_size, CV_8UC1);
                cv::cvtColor(images[i], images_gray[i], CV_RGB2GRAY);
                found[i] = findChessboardCorners(images_gray[i],
                    board_size, corners[i]);
            }
            // if valid corners found in both left and right images,
            // draw them, and add them to image_points
            if (found[0] && found[1]) {
                ++image_num;
                std::cout << "Found all corners in frame" << std::endl;
                object_points.push_back(board_points);
                for (image_index i = 0; i < 2; ++i) {
                    cv::cornerSubPix(images_gray[i], corners[i],
                        cv::Size(11, 11), cv::Size(-1, -1),
                        cv::TermCriteria(CV_TERMCRIT_EPS
                            | CV_TERMCRIT_ITER, 30, 0.1));
                    image_points[i].push_back(corners[i]);
                    drawChessboardCorners(images[i], board_size,
                        corners[i], true);
                    cv::resize(images[i], displays[i], display_size);
                    cv::imshow(window_names[i], displays[i]);
                }
                cv::waitKey(1000); // draw window, and wait 1 second
            } else {
                std::cout << "Not enough corners in frame" << std::endl;
            }
        }
    }

    for (image_index i = 0; i < 2; ++i) {
        cv::destroyWindow(window_names[i]);
    }

    std::cout << "Running stereo camera calibration" << std::endl;

    cv::Mat dist_coeffs[2];
    cv::Mat camera_matrix[2];
    for (image_index i = 0; i < 2; ++i) {
        camera_matrix[i] = cv::initCameraMatrix2D(object_points,
            image_points[i], image_size, 0);
    }
    cv::Mat r_mat, t_mat, e_mat, f_mat;
    cv::stereoCalibrate(object_points, image_points[0], image_points[1],
        camera_matrix[0], dist_coeffs[0],
        camera_matrix[1], dist_coeffs[1],
        image_size, r_mat, t_mat, e_mat, f_mat /*, term_criteria, flags */);
    std::cout << "Finished stereo camera calibration" << std::endl;
}
