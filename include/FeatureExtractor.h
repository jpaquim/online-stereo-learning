#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>

class FeatureExtractor {
public:
    // default constructor
    FeatureExtractor(cv::Size image_size, cv::Size map_size);

    // 
    ~FeatureExtractor() = default;

    //
    void compute(const cv::Mat &image);

private:
    cv::Size patch_size;
    int n_scales;
    int image_height;
    int image_width;
    int map_height;
    int map_width;
    int patch_height;
    int patch_width;
    std::vector<int> patch_center_rows;
    std::vector<int> patch_center_cols;
};

#endif // FEATURE_EXTRACTOR_H
