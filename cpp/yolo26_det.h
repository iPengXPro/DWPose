#ifndef YOLO26_DET_H
#define YOLO26_DET_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include "yolox_det.h"  // use DetectBox from yolox_det.h

class YOLO26Detector {
public:
    YOLO26Detector(const std::string& model_path);
    std::vector<DetectBox> detect(const cv::Mat& img);

private:
    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;

    // YOLO26 preprocessing: letterbox + normalize (/255)
    static std::vector<float> preprocess(const cv::Mat& img, int input_h, int input_w,
                                          float& ratio, float& pad_x, float& pad_y);
};

#endif // YOLO26_DET_H