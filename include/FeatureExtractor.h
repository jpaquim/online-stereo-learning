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
    int num_scales;
    int image_height;
    int image_width;
    int map_height;
    int map_width;
    int num_patches;
    int patch_height;
    int patch_width;

    std::vector<int> half_heights;
    std::vector<int> half_widths;
    std::vector<int> patch_center_rows;
    std::vector<int> patch_center_cols;
    const int num_filters = 17; // 9 Laws + 6 edge + 2 average
    std::vector<cv::Mat> filters;
    std::vector<int> filter_channels;
    int num_features;
    std::vector<std::vector<float>> feature_map;

    // populates the filter bank with Laws' masks, local averaging, and
    // Nevatia-Babu oriented edge detectors
    void populate_filter_bank();
};

#endif // FEATURE_EXTRACTOR_H
