#ifndef DWPOSE_POSE_H
#define DWPOSE_POSE_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"

struct Keypoint {
    float x, y;
    float score;
};

class DWPoseEstimator {
public:
    DWPoseEstimator(const std::string& model_path);

    // process all detected persons, return keypoints per person
    std::vector<std::vector<Keypoint>> estimate(
        const cv::Mat& img,
        const std::vector<std::vector<float>>& bboxes);

private:
    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    int m_model_w, m_model_h; // model input size (w, h)

    // matches onnxpose.py helper functions exactly
    static void bbox_xyxy2cs(float x1, float y1, float x2, float y2,
                             float padding, float& cx, float& cy, float& sw, float& sh);

    static void fix_aspect_ratio(float& sw, float& sh, float aspect_ratio);

    static cv::Mat get_warp_matrix(float cx, float cy, float sw, float sh,
                                    int dst_w, int dst_h);

    // 3rd point for affine
    static void get_3rd_point(float a0, float a1, float b0, float b1,
                              float& c0, float& c1);
};

#endif // DWPOSE_POSE_H
