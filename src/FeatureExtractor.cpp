#include "FeatureExtractor.h"

#include <opencv2/opencv.hpp>
// #include <opencv2/gpu/gpu.hpp>

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
, num_scales(1)
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
, features(num_patches, num_features, CV_64FC1) {
    // place the first and last patch centers such that the largest scale
    // patches are adjacent to the image edges
    for (int scl = 0; scl < num_scales; ++scl) {
        half_heights[scl] =
            std::floor(std::pow(3, scl)*patch_height/2);;
        half_widths[scl] =
            std::floor(std::pow(3, scl)*patch_width/2);;
    }
    int first_patch_center_row = half_heights[num_scales-1];
    int first_patch_center_col = half_widths[num_scales-1];
    // interval between patch centers should be such that the number of
    // patches per column is map_height, and per row is map_width
    double patch_center_row_interval =
        double(image_height-1-2*first_patch_center_row)/(map_height-1);
    double patch_center_col_interval =
        double(image_width-1-2*first_patch_center_col)/(map_width-1);
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

int FeatureExtractor::getNumFeatures() {
    return num_features;
}

void FeatureExtractor::populate_filter_bank() {
    // Laws' masks
    double L3_array[] = { 1, 2,  1};
    double E3_array[] = {-1, 0,  1};
    double S3_array[] = {-1, 2, -1};
    cv::Mat L3 = cv::Mat(1, 3, CV_64FC1, L3_array)/10;
    cv::Mat E3 = cv::Mat(1, 3, CV_64FC1, E3_array)/10;
    cv::Mat S3 = cv::Mat(1, 3, CV_64FC1, S3_array)/10;
    filters[0] = L3.t()*L3/100;
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
    double nevatia_babu_0_deg[] = {-100, -100, 0, 100, 100, // 0 deg
                                  -100, -100, 0, 100, 100,
                                  -100, -100, 0, 100, 100,
                                  -100, -100, 0, 100, 100,
                                  -100, -100, 0, 100, 100};
    double nevatia_babu_30_deg[] = {-100,   32,  100, 100, 100, // 30 deg
                                   -100,  -78,   92, 100, 100,
                                   -100, -100,    0, 100, 100,
                                   -100, -100,  -92,  78, 100,
                                   -100, -100, -100, -32, 100};
    filters[11] = cv::Mat(5, 5, CV_64FC1, nevatia_babu_0_deg)/10000;
    filters[12] = cv::Mat(5, 5, CV_64FC1, nevatia_babu_30_deg)/10000;
    filters[13] = -filters[12].t();        // 60 deg
    filters[14] = -filters[11].t();        // 90 deg
    flip(filters[13], filters[15], 1); // 120 deg
    filters[16] = filters[15].t();         // 150 deg
    for (int i = 11; i < 17; ++i) {
        filter_channels[i] = 0; // Y channel
    }
}

void FeatureExtractor::compute(const cv::Mat &image,
        cv::Mat &features_dest) {
    // convert to YCrCb colorspace
    cv::Mat image_ycrcb;
    // TODO: upload image to GPU, for faster conversion
    // and, if possible, write a CUDA kernel to perform feature extraction
    cvtColor(image, image_ycrcb, CV_RGB2YCrCb);
    // split the channels into a Mat vector
    cv::Mat image_split[3];
    split(image_ycrcb, image_split);
    // loop over the filter bank
    for (int flt = 0; flt < num_filters; ++flt) {
        // filter the whole image using the specified filter and channel
        cv::Mat image_filt;
        filter2D(image_split[filter_channels[flt]], image_filt,
                CV_64F, filters[flt]);
        // imshow("Display", image_filt);
        // cv::waitKey(100);
        for (int scl = 0; scl < num_scales; ++scl) {
            int feat = 2*(scl+flt*num_scales);
            for (int i = 0; i < map_height; ++i) {
                cv::Range row_range(
                        patch_center_rows[i]-half_heights[scl],
                        patch_center_rows[i]+half_heights[scl]);
                for (int j = 0; j < map_width; ++j) {
                    cv::Range col_range(
                            patch_center_cols[j]-half_widths[scl],
                            patch_center_cols[j]+half_widths[scl]);
                    cv::Mat image_patch =
                        image_filt(row_range, col_range);
                    int ptc = i*map_width + j;
                    features.at<double>(ptc, feat) =
                        cv::sum(abs(image_patch))[0];
                    features.at<double>(ptc, feat+1) =
                        cv::sum(image_patch.mul(image_patch))[0];
                }
            }
        }
    }
    features_dest = features;
}

