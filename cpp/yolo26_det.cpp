#include "yolo26_det.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <iostream>

YOLO26Detector::YOLO26Detector(const std::string& model_path)
    : m_env(ORT_LOGGING_LEVEL_ERROR, "yolo26")
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try CUDA first, fallback to CPU
    OrtCUDAProviderOptions cuda_opts;
    try {
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "  [YOLO26] Using CUDA GPU" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cout << "  [YOLO26] CUDA not available (" << e.what() << "), using CPU" << std::endl;
        OrtSessionOptionsAppendExecutionProvider_CPU(opts, 0);
    }

#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    m_session = std::make_unique<Ort::Session>(m_env, wpath.c_str(), opts);
#else
    m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), opts);
#endif
}

std::vector<float> YOLO26Detector::preprocess(const cv::Mat& img, int input_h, int input_w,
                                               float& ratio, float& pad_x, float& pad_y)
{
    // Letterbox resize with aspect ratio preservation
    float r = std::min(static_cast<float>(input_h) / img.rows,
                       static_cast<float>(input_w) / img.cols);

    int new_h = static_cast<int>(img.rows * r);
    int new_w = static_cast<int>(img.cols * r);

    // Calculate padding
    pad_x = (input_w - new_w) / 2.0f;
    pad_y = (input_h - new_h) / 2.0f;
    ratio = r;

    // Create padded image with gray (114)
    cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));

    // Resize and copy to center
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(padded(cv::Rect(static_cast<int>(pad_x), static_cast<int>(pad_y), new_w, new_h)));

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

std::vector<DetectBox> YOLO26Detector::detect(const cv::Mat& img)
{
    const int input_h = 640, input_w = 640;
    const float conf_thr = 0.25f;
    const int target_class = 0;  // person

    float ratio, pad_x, pad_y;
    std::vector<float> input_data = preprocess(img, input_h, input_w, ratio, pad_x, pad_y);

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

    // Output shape: [1, 300, 6] -> [x1, y1, x2, y2, conf, class]
    auto& out_tensor = output_tensors[0];
    auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();

    int max_det = static_cast<int>(out_shape[1]);  // 300
    int num_fields = static_cast<int>(out_shape[2]);  // 6

    const float* output = out_tensor.GetTensorData<float>();

    std::vector<DetectBox> results;

    for (int i = 0; i < max_det; i++) {
        const float* det = output + i * num_fields;

        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float score = det[4];
        int class_id = static_cast<int>(det[5]);

        // Filter by confidence and class (person = 0)
        if (score < conf_thr || class_id != target_class) {
            continue;
        }

        // Map coordinates back to original image
        // Remove padding and scale by ratio
        float orig_x1 = (x1 - pad_x) / ratio;
        float orig_y1 = (y1 - pad_y) / ratio;
        float orig_x2 = (x2 - pad_x) / ratio;
        float orig_y2 = (y2 - pad_y) / ratio;

        results.push_back({orig_x1, orig_y1, orig_x2, orig_y2});
    }

    return results;
}