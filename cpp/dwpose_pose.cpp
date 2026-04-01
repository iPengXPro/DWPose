#include "dwpose_pose.h"
#include <algorithm>
#include <cmath>
#include <iostream>

DWPoseEstimator::DWPoseEstimator(const std::string& model_path)
    : m_env(ORT_LOGGING_LEVEL_ERROR, "dwpose"), m_model_w(0), m_model_h(0)
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try CUDA first, fallback to CPU
    OrtCUDAProviderOptions cuda_opts;
    try {
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "  [DWPose] Using CUDA GPU" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cout << "  [DWPose] CUDA not available (" << e.what() << "), using CPU" << std::endl;
        OrtSessionOptionsAppendExecutionProvider_CPU(opts, 0);
    }

#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    m_session = std::make_unique<Ort::Session>(m_env, wpath.c_str(), opts);
#else
    m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), opts);
#endif

    // read model input shape: [1, 3, h, w]
    auto input_shape = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    m_model_h = static_cast<int>(input_shape[2]);
    m_model_w = static_cast<int>(input_shape[3]);
}

void DWPoseEstimator::get_3rd_point(float a0, float a1, float b0, float b1,
                                     float& c0, float& c1)
{
    // matches onnxpose.py::_get_3rd_point
    // direction = a - b
    // c = b + [-direction[1], direction[0]]
    float d0 = a0 - b0;
    float d1 = a1 - b1;
    c0 = b0 + (-d1);
    c1 = b1 + d0;
}

void DWPoseEstimator::bbox_xyxy2cs(float x1, float y1, float x2, float y2,
                                    float padding, float& cx, float& cy, float& sw, float& sh)
{
    // matches onnxpose.py::bbox_xyxy2cs
    cx = (x1 + x2) * 0.5f;
    cy = (y1 + y2) * 0.5f;
    sw = (x2 - x1) * padding;
    sh = (y2 - y1) * padding;
}

void DWPoseEstimator::fix_aspect_ratio(float& sw, float& sh, float aspect_ratio)
{
    // matches onnxpose.py::_fix_aspect_ratio
    // aspect_ratio = w/h (target)
    if (sw > sh * aspect_ratio) {
        sh = sw / aspect_ratio;
    } else {
        sw = sh * aspect_ratio;
    }
}

cv::Mat DWPoseEstimator::get_warp_matrix(float cx, float cy, float sw, float sh,
                                          int dst_w, int dst_h)
{
    // matches onnxpose.py::get_warp_matrix exactly (rot=0)
    // rot_rad = 0, so cos=1, sin=0
    // src_dir = rotate_point([0, sw * -0.5], 0) = [0, sw * -0.5]
    float src_dir0 = 0.0f;
    float src_dir1 = sw * -0.5f;

    float dst_dir0 = 0.0f;
    float dst_dir1 = dst_w * -0.5f;

    // src points
    // src[0] = center + scale * shift  (shift = (0,0))
    float src0_0 = cx;
    float src0_1 = cy;
    // src[1] = center + src_dir + scale * shift
    float src1_0 = cx + src_dir0;
    float src1_1 = cy + src_dir1;
    // src[2] = _get_3rd_point(src[0], src[1])
    float src2_0, src2_1;
    get_3rd_point(src0_0, src0_1, src1_0, src1_1, src2_0, src2_1);

    // dst points
    float dst0_0 = dst_w * 0.5f;
    float dst0_1 = dst_h * 0.5f;
    float dst1_0 = dst_w * 0.5f + dst_dir0;
    float dst1_1 = dst_h * 0.5f + dst_dir1;
    float dst2_0, dst2_1;
    get_3rd_point(dst0_0, dst0_1, dst1_0, dst1_1, dst2_0, dst2_1);

    cv::Point2f src_pts[3] = {
        {src0_0, src0_1}, {src1_0, src1_1}, {src2_0, src2_1}
    };
    cv::Point2f dst_pts[3] = {
        {dst0_0, dst0_1}, {dst1_0, dst1_1}, {dst2_0, dst2_1}
    };

    return cv::getAffineTransform(src_pts, dst_pts);
}

std::vector<std::vector<Keypoint>> DWPoseEstimator::estimate(
    const cv::Mat& img,
    const std::vector<std::vector<float>>& bboxes)
{
    std::vector<std::vector<Keypoint>> all_keypoints;

    if (bboxes.empty()) return all_keypoints;

    // normalization constants (matches onnxpose.py exactly)
    const float mean[3] = {123.675f, 116.28f, 103.53f};
    const float std_val[3] = {58.395f, 57.12f, 57.375f};
    const float simcc_split_ratio = 2.0f;

    for (const auto& bbox : bboxes) {
        float x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];

        // bbox_xyxy2cs with padding=1.25
        float cx, cy, sw, sh;
        bbox_xyxy2cs(x1, y1, x2, y2, 1.25f, cx, cy, sw, sh);

        // fix_aspect_ratio to match model input w/h
        float aspect_ratio = static_cast<float>(m_model_w) / static_cast<float>(m_model_h);
        fix_aspect_ratio(sw, sh, aspect_ratio);

        // get affine matrix and warp
        cv::Mat warp_mat = get_warp_matrix(cx, cy, sw, sh, m_model_w, m_model_h);
        cv::Mat warped;
        cv::warpAffine(img, warped, warp_mat, cv::Size(m_model_w, m_model_h), cv::INTER_LINEAR);

        // normalize: (img - mean) / std, BGR order, HWC -> CHW
        int h = warped.rows, w = warped.cols;
        std::vector<float> input_data(3 * h * w);

        for (int row = 0; row < h; row++) {
            const uchar* ptr = warped.ptr<uchar>(row);
            for (int col = 0; col < w; col++) {
                for (int c = 0; c < 3; c++) {
                    int chw_idx = c * h * w + row * w + col;
                    input_data[chw_idx] = (static_cast<float>(ptr[col * 3 + c]) - mean[c]) / std_val[c];
                }
            }
        }

        // inference
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<int64_t, 4> input_shape{1, 3, h, w};

        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"simcc_x", "simcc_y"};

        auto output_tensors = m_session->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_ort, 1,
            output_names, 2);

        // SIMCC decode
        auto simcc_x_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        auto simcc_y_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

        int num_kpts = static_cast<int>(simcc_x_shape[1]);
        int extend_w = static_cast<int>(simcc_x_shape[2]);
        int extend_h = static_cast<int>(simcc_y_shape[2]);

        const float* simcc_x = output_tensors[0].GetTensorData<float>();
        const float* simcc_y = output_tensors[1].GetTensorData<float>();

        std::vector<Keypoint> keypoints(num_kpts);

        for (int k = 0; k < num_kpts; k++) {
            // argmax on x
            int max_x_pos = 0;
            float max_x_val = simcc_x[k * extend_w];
            for (int i = 1; i < extend_w; i++) {
                if (simcc_x[k * extend_w + i] > max_x_val) {
                    max_x_val = simcc_x[k * extend_w + i];
                    max_x_pos = i;
                }
            }

            // argmax on y
            int max_y_pos = 0;
            float max_y_val = simcc_y[k * extend_h];
            for (int i = 1; i < extend_h; i++) {
                if (simcc_y[k * extend_h + i] > max_y_val) {
                    max_y_val = simcc_y[k * extend_h + i];
                    max_y_pos = i;
                }
            }

            // score = min(max_x_val, max_y_val)  (matches Python exactly)
            float score = std::min(max_x_val, max_y_val);

            float kx = static_cast<float>(max_x_pos);
            float ky = static_cast<float>(max_y_pos);

            // set to -1 if score <= 0
            if (score <= 0.0f) {
                kx = -1.0f;
                ky = -1.0f;
            }

            // divide by simcc_split_ratio
            kx /= simcc_split_ratio;
            ky /= simcc_split_ratio;

            // rescale: keypoints / model_input_size * scale + center - scale / 2
            float fx = kx / m_model_w * sw + cx - sw / 2.0f;
            float fy = ky / m_model_h * sh + cy - sh / 2.0f;

            keypoints[k] = {fx, fy, score};
        }

        all_keypoints.push_back(std::move(keypoints));
    }

    return all_keypoints;
}
