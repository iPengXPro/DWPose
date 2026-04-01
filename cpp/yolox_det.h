#ifndef YOLOX_DET_H
#define YOLOX_DET_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"

struct DetectBox {
    float x1, y1, x2, y2;
};

class YOLOXDetector {
public:
    YOLOXDetector(const std::string& model_path);
    std::vector<DetectBox> detect(const cv::Mat& img);

private:
    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;

    // matches onnxdet.py::preprocess exactly
    static cv::Mat preprocess(const cv::Mat& img, int input_h, int input_w, float& ratio);

    // matches onnxdet.py::demo_postprocess
    static void demo_postprocess(float* output, int num_preds, int num_attrs,
                                  int img_h, int img_w);

    // matches onnxdet.py::nms
    static std::vector<int> nms(const std::vector<cv::Rect2f>& boxes,
                                const std::vector<float>& scores, float nms_thr);

    // matches onnxdet.py::multiclass_nms
    std::vector<DetectBox> multiclass_nms(const std::vector<cv::Rect2f>& boxes,
                                          const std::vector<std::vector<float>>& scores,
                                          float nms_thr, float score_thr);
};

#endif // YOLOX_DET_H
