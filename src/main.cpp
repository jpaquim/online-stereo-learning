#include "StereoCamera.h"
#include "FeatureExtractor.h"
#include "LinearRegression.h"

#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char **argv) {
    StereoCamera camera;
    if (argc > 1 && !strcmp(argv[1], "calibrate")) {
        camera.calibrate();
    }
    cv::Mat depth_map;
    cv::Mat predicted_depth_map;
    cv::Size map_size(200, 100);
    int num_patches = map_size.height * map_size.width;

    cv::Mat features;
    FeatureExtractor feature_extractor(camera.getSize(), map_size);

    LinearRegression linear_regression(feature_extractor.getNumFeatures());

    char key = ' ';
    float ratio = float(camera.getWidth()) / float(camera.getHeight());
    const int display_height = 320;
    // display window size, preserve aspect ratio
    cv::Size display_size(ratio*display_height, display_height);

    while (key != 'q') {
        if (!camera.read()) {
            continue;
        }
        // extract features
        feature_extractor.compute(camera.getImageLeft(), features);
        // resize depth map to standard size
        resize(camera.getDepthMap(), depth_map, map_size);
        extractChannel(depth_map, depth_map, 0);
        depth_map.convertTo(depth_map, CV_64F);
        // train the regression algorithm
        linear_regression.train(features, depth_map.
                reshape(0, num_patches));
        // predict using the trained model
        linear_regression.predict(features, predicted_depth_map);
        // display stereo and predicted depths
        cv::Mat display(display_size, CV_8UC4);
        resize(depth_map, display, display_size);
        imshow("Stereo depths", display);

        resize(predicted_depth_map
                .reshape(0, map_size.height), display, display_size);
        imshow("Predicted depths", display);
        key = cv::waitKey(1);
    }
    return 0;
}
