#include "yolov8_det.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <iostream>

YOLOv8Detector::YOLOv8Detector(const std::string& model_path)
    : m_env(ORT_LOGGING_LEVEL_ERROR, "yolov8")
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try CUDA first, fallback to CPU
    OrtCUDAProviderOptions cuda_opts;
    try {
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "  [YOLOv8] Using CUDA GPU" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cout << "  [YOLOv8] CUDA not available (" << e.what() << "), using CPU" << std::endl;
        OrtSessionOptionsAppendExecutionProvider_CPU(opts, 0);
    }

#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    m_session = std::make_unique<Ort::Session>(m_env, wpath.c_str(), opts);
#else
    m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), opts);
#endif
}

std::vector<float> YOLOv8Detector::preprocess(const cv::Mat& img, int input_h, int input_w, float& ratio)
{
    // Letterbox resize (same ratio logic as YOLOX but normalize /255)
    cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));

    ratio = std::min(static_cast<float>(input_h) / img.rows,
                     static_cast<float>(input_w) / img.cols);

    int new_h = static_cast<int>(img.rows * ratio);
    int new_w = static_cast<int>(img.cols * ratio);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

    // HWC -> CHW, BGR -> RGB, uint8 -> float32 / 255
    std::vector<float> input_data(3 * input_h * input_w);
    for (int h = 0; h < input_h; h++) {
        for (int w = 0; w < input_w; w++) {
            const cv::Vec3b& pixel = padded.at<cv::Vec3b>(h, w);
            // BGR -> RGB order
            input_data[0 * input_h * input_w + h * input_w + w] = pixel[2] / 255.0f; // R
            input_data[1 * input_h * input_w + h * input_w + w] = pixel[1] / 255.0f; // G
            input_data[2 * input_h * input_w + h * input_w + w] = pixel[0] / 255.0f; // B
        }
    }

    return input_data;
}

std::vector<int> YOLOv8Detector::nms(const std::vector<cv::Rect2f>& boxes,
                                      const std::vector<float>& scores, float nms_thr)
{
    int n = static_cast<int>(boxes.size());
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = boxes[i].width * boxes[i].height;
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);
        if (order.size() == 1) break;

        std::vector<int> remaining(order.begin() + 1, order.end());
        std::vector<int> inds;

        for (int k = 0; k < static_cast<int>(remaining.size()); k++) {
            int j = remaining[k];
            float xx1 = std::max(boxes[i].x, boxes[j].x);
            float yy1 = std::max(boxes[i].y, boxes[j].y);
            float xx2 = std::min(boxes[i].x + boxes[i].width, boxes[j].x + boxes[j].width);
            float yy2 = std::min(boxes[i].y + boxes[i].height, boxes[j].y + boxes[j].height);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[j] - inter);

            if (ovr <= nms_thr) {
                inds.push_back(k);
            }
        }

        std::vector<int> new_order;
        for (int idx : inds) {
            new_order.push_back(remaining[idx]);
        }
        order = std::move(new_order);
    }
    return keep;
}

std::vector<DetectBox> YOLOv8Detector::detect(const cv::Mat& img)
{
    const int input_h = 640, input_w = 640;

    float ratio;
    std::vector<float> input_data = preprocess(img, input_h, input_w, ratio);

    // Inference
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<int64_t, 4> input_shape{1, 3, input_h, input_w};

    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    auto output_tensors = m_session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_ort, 1,
        output_names, 1);

    // Output shape: [1, 84, 8400]
    // 84 = 4 (cx, cy, w, h) + 80 (class scores)
    // Need to transpose to [8400, 84] for processing
    auto& out_tensor = output_tensors[0];
    auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();

    int num_preds = static_cast<int>(out_shape[2]);  // 8400
    int num_attrs = static_cast<int>(out_shape[1]);  // 84
    int num_classes = num_attrs - 4;                  // 80

    const float* output = out_tensor.GetTensorData<float>();

    // Transpose and process
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    const float conf_thr = 0.25f;  // match Python conf filter
    const float nms_thr = 0.45f;
    const float final_thr = 0.3f;  // match Python final filter

    for (int i = 0; i < num_preds; i++) {
        // Output is [84, 8400], so access as output[attr * num_preds + pred]
        float cx = output[0 * num_preds + i];
        float cy = output[1 * num_preds + i];
        float w = output[2 * num_preds + i];
        float h = output[3 * num_preds + i];

        // Find max class score
        int best_class = 0;
        float best_score = output[4 * num_preds + i];
        for (int c = 1; c < num_classes; c++) {
            float score = output[(4 + c) * num_preds + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < conf_thr) continue;

        // Filter: class 0 (person) only
        if (best_class != 0) continue;

        // Convert cx, cy, w, h -> x1, y1, x2, y2 (in input coords)
        float x1 = cx - w / 2;
        float y1 = cy - h / 2;

        boxes.emplace_back(x1, y1, w, h);
        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }

    // NMS
    std::vector<DetectBox> result;
    if (!boxes.empty()) {
        auto keep = nms(boxes, scores, nms_thr);
        for (int idx : keep) {
            // Final filter: score > 0.3 (match Python)
            if (scores[idx] < final_thr) continue;
            // Scale back to original image
            float x1 = boxes[idx].x / ratio;
            float y1 = boxes[idx].y / ratio;
            float x2 = (boxes[idx].x + boxes[idx].width) / ratio;
            float y2 = (boxes[idx].y + boxes[idx].height) / ratio;
            result.push_back({x1, y1, x2, y2});
        }
    }

    return result;
}