#include "LinearRegression.h"

#include <opencv2/opencv.hpp>

#include <iostream>

#define DEBUG(variable) do {                                 \
    std::cout << #variable << ": " << variable << std::endl; \
} while(0)

void print_double_mat(cv::Mat mat) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            std::cout << mat.at<double>(i, j) << std::endl;
        }
    }
}

LinearRegression::LinearRegression(int num_features)
: weights(num_features, 1, CV_64FC1) {
    // initialize with random weights
    cv::randn(weights, 0, 1);
}

void LinearRegression::train(const cv::Mat &features,
        const cv::Mat &targets) {
    // simple gradient descent training
    double learning_rate = 0.00001;
    int num_samples = features.rows;
    weights -=
        learning_rate/num_samples*features.t()*(features*weights-targets);

    // print_double_mat(features);
    // print_double_mat(weights);
}

void LinearRegression::predict(const cv::Mat &features, cv::Mat &targets) {
    targets = features*weights;
}
