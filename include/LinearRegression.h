#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <opencv2/opencv.hpp>

class LinearRegression {
public:
    LinearRegression(int num_features);
    void train(const cv::Mat &features, const cv::Mat &targets);
    void predict(const cv::Mat &features, cv::Mat &targets);
private:
    int lambda;
    cv::Mat weights;
};

#endif
