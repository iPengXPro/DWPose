#include "yolox_det.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

YOLOXDetector::YOLOXDetector(const std::string& model_path)
    : m_env(ORT_LOGGING_LEVEL_ERROR, "yolox")
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try CUDA first, fallback to CPU
    OrtCUDAProviderOptions cuda_opts;
    try {
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "  [YOLOX] Using CUDA GPU" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cout << "  [YOLOX] CUDA not available (" << e.what() << "), using CPU" << std::endl;
        OrtSessionOptionsAppendExecutionProvider_CPU(opts, 0);
    }

#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    m_session = std::make_unique<Ort::Session>(m_env, wpath.c_str(), opts);
#else
    m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), opts);
#endif
}

cv::Mat YOLOXDetector::preprocess(const cv::Mat& img, int input_h, int input_w, float& ratio)
{
    // matches onnxdet.py::preprocess exactly
    cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));

    ratio = std::min(static_cast<float>(input_h) / img.rows,
                     static_cast<float>(input_w) / img.cols);

    int new_h = static_cast<int>(img.rows * ratio);
    int new_w = static_cast<int>(img.cols * ratio);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

    // HWC -> CHW, uint8 -> float32 (no normalization, no RGB conversion)
    cv::Mat chw;
    cv::dnn::blobFromImage(padded, chw, 1.0, cv::Size(), cv::Scalar(), false, false);

    return chw;
}

void YOLOXDetector::demo_postprocess(float* output, int num_preds, int num_attrs,
                                      int img_h, int img_w)
{
    // matches onnxdet.py::demo_postprocess exactly
    int strides[] = {8, 16, 32};
    int hsizes[] = {img_h / 8, img_h / 16, img_h / 32};
    int wsizes[] = {img_w / 8, img_w / 16, img_w / 32};

    int grid_offset = 0;
    for (int s = 0; s < 3; s++) {
        int hsize = hsizes[s];
        int wsize = wsizes[s];
        int stride = strides[s];
        int count = hsize * wsize;

        for (int idx = 0; idx < count; idx++) {
            int gi = grid_offset + idx;
            int gx = idx % wsize;
            int gy = idx / wsize;

            // outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            output[gi * num_attrs + 0] = (output[gi * num_attrs + 0] + gx) * stride;
            output[gi * num_attrs + 1] = (output[gi * num_attrs + 1] + gy) * stride;

            // outputs[..., 2:4] = exp(outputs[..., 2:4]) * expanded_strides
            output[gi * num_attrs + 2] = std::exp(output[gi * num_attrs + 2]) * stride;
            output[gi * num_attrs + 3] = std::exp(output[gi * num_attrs + 3]) * stride;
        }
        grid_offset += count;
    }
}

std::vector<int> YOLOXDetector::nms(const std::vector<cv::Rect2f>& boxes,
                                     const std::vector<float>& scores, float nms_thr)
{
    // matches onnxdet.py::nms exactly
    int n = static_cast<int>(boxes.size());
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = (boxes[i].width + 1) * (boxes[i].height + 1);
    }

    // argsort descending by score
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

            float w = std::max(0.0f, xx2 - xx1 + 1);
            float h = std::max(0.0f, yy2 - yy1 + 1);
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

std::vector<DetectBox> YOLOXDetector::multiclass_nms(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<std::vector<float>>& scores,
    float nms_thr, float score_thr)
{
    // matches onnxdet.py::multiclass_nms exactly
    int num_boxes = static_cast<int>(boxes.size());
    int num_classes = static_cast<int>(scores.empty() ? 0 : scores[0].size());

    // final_dets: each entry is [x1, y1, x2, y2, score, cls_ind]
    struct Det {
        float x1, y1, x2, y2, score;
        int cls;
    };
    std::vector<Det> final_dets;

    for (int cls_ind = 0; cls_ind < num_classes; cls_ind++) {
        // filter by score_thr for this class
        std::vector<cv::Rect2f> cls_boxes;
        std::vector<float> cls_scores;
        std::vector<int> cls_indices;

        for (int i = 0; i < num_boxes; i++) {
            if (scores[i][cls_ind] > score_thr) {
                cls_boxes.push_back(boxes[i]);
                cls_scores.push_back(scores[i][cls_ind]);
                cls_indices.push_back(i);
            }
        }
        if (cls_boxes.empty()) continue;

        auto keep = nms(cls_boxes, cls_scores, nms_thr);
        for (int k : keep) {
            final_dets.push_back({
                cls_boxes[k].x, cls_boxes[k].y,
                cls_boxes[k].x + cls_boxes[k].width,
                cls_boxes[k].y + cls_boxes[k].height,
                cls_scores[k], cls_ind
            });
        }
    }

    // filter: class==0 AND score>0.3
    std::vector<DetectBox> result;
    for (auto& d : final_dets) {
        if (d.cls == 0 && d.score > 0.3f) {
            result.push_back({d.x1, d.y1, d.x2, d.y2});
        }
    }
    return result;
}

std::vector<DetectBox> YOLOXDetector::detect(const cv::Mat& img)
{
    const int input_h = 640, input_w = 640;

    float ratio;
    cv::Mat input_tensor = preprocess(img, input_h, input_w, ratio);

    // inference
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<int64_t, 4> input_shape{1, 3, input_h, input_w};

    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info, (float*)input_tensor.data, input_tensor.total(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output"};

    auto output_tensors = m_session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_ort, 1,
        output_names, 1);

    auto& out_tensor = output_tensors[0];
    auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    // shape: [1, num_preds, num_attrs]  e.g. [1, 8400, 85]
    int num_preds = static_cast<int>(out_shape[1]);
    int num_attrs = static_cast<int>(out_shape[2]);

    float* output = out_tensor.GetTensorMutableData<float>();

    // postprocess: apply grid + stride decode in-place
    demo_postprocess(output, num_preds, num_attrs, input_h, input_w);

    // extract boxes and scores
    // num_attrs = 5 + num_classes (85 for COCO: cx, cy, w, h, obj, 80 classes)
    int num_classes = num_attrs - 5;

    std::vector<cv::Rect2f> boxes_xyxy;
    std::vector<std::vector<float>> all_scores;

    for (int i = 0; i < num_preds; i++) {
        float* p = output + i * num_attrs;
        float cx = p[0], cy = p[1], w = p[2], h = p[3];

        // convert cxcywh -> xyxy
        float x1 = cx - w / 2;
        float y1 = cy - h / 2;

        boxes_xyxy.emplace_back(x1, y1, w, h);

        // scores = obj_conf * cls_conf
        std::vector<float> cls_scores(num_classes);
        for (int c = 0; c < num_classes; c++) {
            cls_scores[c] = p[4] * p[5 + c];
        }
        all_scores.push_back(std::move(cls_scores));
    }

    // NMS and filter
    auto result = multiclass_nms(boxes_xyxy, all_scores, 0.45f, 0.1f);

    // divide by ratio to map back to original image coordinates
    for (auto& box : result) {
        box.x1 /= ratio;
        box.y1 /= ratio;
        box.x2 /= ratio;
        box.y2 /= ratio;
    }

    return result;
}
