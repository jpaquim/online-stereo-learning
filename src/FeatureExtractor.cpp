#include "FeatureExtractor.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#define DEBUG(variable) do {                                 \
    std::cout << #variable << ": " << variable << std::endl; \
} while(0)

FeatureExtractor::FeatureExtractor(cv::Size image_size, cv::Size map_size)
: patch_size(5, 5)
, n_scales(3)
, image_height(image_size.height)
, image_width(image_size.width)
, map_height(map_size.height)
, map_width(map_size.width)
, patch_height(patch_size.height)
, patch_width(patch_size.width)
, patch_center_rows(map_height)
, patch_center_cols(map_width) {
    // place the first and last patch centers such that the largest scale
    // patches are adjacent to the image edges
    int first_patch_center_row =
        std::ceil(std::pow(3, n_scales-1)*patch_height/2);
    int first_patch_center_col =
        std::ceil(std::pow(3, n_scales-1)*patch_width/2);
    // interval between patch centers should be such that the number of
    // patches per column is map_height, and per row is map_width
    double patch_center_row_interval =
        double(image_height-2*first_patch_center_row)/map_height;
    double patch_center_col_interval =
        double(image_width-2*first_patch_center_col)/map_width;
    // populate the patch center rows and columns
    for (int i = 0; i < map_height; ++i) {
        patch_center_rows[i] =
            std::round(first_patch_center_row + i*patch_center_row_interval);
    }
    for (int j = 0; j < map_width; ++j) {
        patch_center_cols[j] =
            std::round(first_patch_center_col + j*patch_center_col_interval);
    }
}

void FeatureExtractor::compute(const cv::Mat &image) {
}
