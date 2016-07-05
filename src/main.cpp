#include "StereoCamera.h"

#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char **argv) {
    StereoCamera camera;

    if (argc > 1 && !strcmp(argv[1], "calibrate")) {
        camera.calibrate();
    }
    char key = ' ';
    float ratio = float(camera.getWidth())/float(camera.getHeight());
    const int display_height = 320;
    // display window size, preserve aspect ratio
    cv::Size display_size(ratio*display_height, display_height);

    while (key != 'q') {
        if (!camera.read()) {
            continue;
        }
        cv::Mat display(display_size, CV_8UC4);
        cv::resize(camera.getImageLeft(), display, display_size);
        cv::imshow("Display", display);
        key = cv::waitKey(1);
    }
    return 0;
}
