#ifndef YOLOV8_DET_H
#define YOLOV8_DET_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include "yolox_det.h"  // use DetectBox from yolox_det.h

class YOLOv8Detector {
public:
    YOLOv8Detector(const std::string& model_path);
    std::vector<DetectBox> detect(const cv::Mat& img);

private:
    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;

    // YOLOv8 preprocessing: resize + normalize (/255)
    static std::vector<float> preprocess(const cv::Mat& img, int input_h, int input_w, float& ratio);

    // NMS for YOLOv8
    static std::vector<int> nms(const std::vector<cv::Rect2f>& boxes,
                                const std::vector<float>& scores, float nms_thr);
};

#endif // YOLOV8_DET_H