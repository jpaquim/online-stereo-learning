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

static const int num_filters = 17; // 9 Laws + 6 edge + 2 average

FeatureExtractor::FeatureExtractor(cv::Size image_size, cv::Size map_size)
: patch_size(5, 5)
, num_scales(3)
, image_height(image_size.height)
, image_width(image_size.width)
, map_height(map_size.height)
, map_width(map_size.width)
, num_patches(map_height*map_width)
, patch_height(patch_size.height)
, patch_width(patch_size.width)
, half_heights(num_scales)
, half_widths(num_scales)
, patch_center_rows(map_height)
, patch_center_cols(map_width)
, filters(num_filters)
, filter_channels(num_filters)
, num_features(2*num_scales*num_filters)
, feature_map(num_patches, std::vector<float>(num_features)) {
    // place the first and last patch centers such that the largest scale
    // patches are adjacent to the image edges
    for (int scl = 0; scl < num_scales; ++scl) {
        half_heights[scl] =
            std::floor(std::pow(3, scl)*(patch_height/2));;
        half_widths[scl] =
            std::floor(std::pow(3, scl)*(patch_width/2));;
    }
    int first_patch_center_row = half_heights[num_scales-1];
    int first_patch_center_col = half_widths[num_scales-1];
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

    // populate the filter bank
    populate_filter_bank();
}

void FeatureExtractor::populate_filter_bank() {
    // Laws' masks
    float L3_array[] = { 1, 2,  1};
    float E3_array[] = {-1, 0,  1};
    float S3_array[] = {-1, 2, -1};
    cv::Mat L3(1, 3, CV_32FC1, L3_array);
    cv::Mat E3(1, 3, CV_32FC1, E3_array);
    cv::Mat S3(1, 3, CV_32FC1, S3_array);
    filters[0] = L3.t()*L3;
    filters[1] = L3.t()*E3;
    filters[2] = L3.t()*S3;
    filters[3] = E3.t()*L3;
    filters[4] = E3.t()*E3;
    filters[5] = E3.t()*S3;
    filters[6] = S3.t()*L3;
    filters[7] = S3.t()*E3;
    filters[8] = S3.t()*S3;
    for (int i = 0; i < 9; ++i) {
        filter_channels[i] = 0; // Y channel
    }
    // local averaging filters
    filters[9] = filters[0];
    filters[10] = filters[0];
    filter_channels[9] = 1; // Cr channel
    filter_channels[10] = 2; // Cb channel
    // Nevatia-Babu oriented edge detectors
    float nevatia_babu_0_deg[] = {-100, -100, 0, 100, 100, // 0 deg
                                  -100, -100, 0, 100, 100,
                                  -100, -100, 0, 100, 100,
                                  -100, -100, 0, 100, 100,
                                  -100, -100, 0, 100, 100};
    float nevatia_babu_30_deg[] = {-100,   32,  100, 100, 100, // 30 deg
                                   -100,  -78,   92, 100, 100,
                                   -100, -100,    0, 100, 100,
                                   -100, -100,  -92,  78, 100,
                                   -100, -100, -100, -32, 100};
    filters[11] = cv::Mat(5, 5, CV_32FC1, nevatia_babu_0_deg);
    filters[12] = cv::Mat(5, 5, CV_32FC1, nevatia_babu_30_deg);
    filters[13] = -filters[12].t();        // 60 deg
    filters[14] = -filters[11].t();        // 90 deg
    cv::flip(filters[13], filters[15], 1); // 120 deg
    filters[16] = filters[15].t();         // 150 deg
    for (int i = 11; i < 17; ++i) {
        filter_channels[i] = 0; // Y channel
    }
}

void FeatureExtractor::compute(const cv::Mat &image) {
    // convert to YCrCb colorspace
    cv::Mat image_ycrcb;
    cv::cvtColor(image, image_ycrcb, CV_RGB2YCrCb);
    // split the channels into a Mat vector
    cv::Mat image_split[3];
    cv::split(image_ycrcb, image_split);
    // loop over the filter bank
    for (int flt = 0; flt < num_filters; ++flt) {
        // filter the whole image using the specified filter and channel
        cv::Mat image_filt;
        cv::filter2D(image_split[filter_channels[flt]], image_filt,
                CV_32F, filters[flt]);
        imshow("Display", image_filt);
        cv::waitKey(1);
        for (int scl = 0; scl < num_scales; ++scl) {
            for (int i = 0; i < map_height; ++i) {
                for (int j = 0; j < map_width; ++j) {
                    cv::Range row_range(
                            patch_center_rows[i]-half_heights[scl],
                            patch_center_rows[i]+half_heights[scl]);
                    cv::Range col_range(
                            patch_center_cols[j]-half_widths[scl],
                            patch_center_cols[j]+half_widths[scl]);
                    cv::Mat image_patch =
                        image_filt(col_range, row_range);
                        // image(col_range, row_range);
                    // cv::Mat display;
                    // cv::resize(image_patch, display, cv::Size(300, 300));
                    // imshow("Display", display);
                    // cv::waitKey(1);
                    int ptc = i*map_height + j;
                    int feat = 2*(scl+(flt-1)*num_scales);
                    feature_map[ptc][feat] =
                        cv::sum(abs(image_patch))[0];
                    feature_map[ptc][feat] =
                        cv::sum(image_patch.mul(image_patch))[0];
                }
            }
        }
    }
}

