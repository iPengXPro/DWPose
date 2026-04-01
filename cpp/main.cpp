#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <memory>
#include <chrono>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "yolox_det.h"
#include "yolov8_det.h"
#include "yolo26_det.h"
#include "dwpose_pose.h"

namespace fs = std::filesystem;

// Detector type enum
enum class DetectorType { YOLOX, YOLOV8, YOLO26 };

// Timer utility
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// body skeleton (matches quick_run_overlay.py exactly)
static const int skeleton[][2] = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 11}, {6, 12}, {11, 12},
    {11, 13}, {13, 15}, {12, 14}, {14, 16},
};
static const int num_skeleton = 16;

static void draw_pose(cv::Mat& canvas, const std::vector<Keypoint>& kps, float score_thr = 0.3f)
{
    int W = canvas.cols, H = canvas.rows;

    // draw lines
    for (int e = 0; e < num_skeleton; e++) {
        int i = skeleton[e][0], j = skeleton[e][1];
        if (i >= (int)kps.size() || j >= (int)kps.size()) continue;
        if (kps[i].score > score_thr && kps[j].score > score_thr) {
            cv::Point p1(cvRound(kps[i].x), cvRound(kps[i].y));
            cv::Point p2(cvRound(kps[j].x), cvRound(kps[j].y));
            if (p1.x >= 0 && p1.x < W && p1.y >= 0 && p1.y < H &&
                p2.x >= 0 && p2.x < W && p2.y >= 0 && p2.y < H) {
                cv::line(canvas, p1, p2, cv::Scalar(0, 255, 0), 3);
            }
        }
    }

    // draw keypoints
    for (const auto& kp : kps) {
        if (kp.score > score_thr) {
            int x = cvRound(kp.x), y = cvRound(kp.y);
            if (x >= 0 && x < W && y >= 0 && y < H) {
                cv::circle(canvas, cv::Point(x, y), 8, cv::Scalar(0, 255, 0), -1);
                cv::circle(canvas, cv::Point(x, y), 10, cv::Scalar(255, 255, 255), 2);
            }
        }
    }
}

static std::string detector_name(DetectorType type) {
    switch (type) {
        case DetectorType::YOLOX: return "YOLOX-L";
        case DetectorType::YOLOV8: return "YOLOv8n";
        case DetectorType::YOLO26: return "YOLO26s";
    }
    return "Unknown";
}

int main(int argc, char* argv[])
{
    std::string base_dir = fs::current_path().string();
    DetectorType detector_type = DetectorType::YOLOX;  // default

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--yolov8" || arg == "-y8") {
            detector_type = DetectorType::YOLOV8;
        } else if (arg == "--yolo26" || arg == "-y26") {
            detector_type = DetectorType::YOLO26;
        } else if (arg == "--yolox" || arg == "-yx") {
            detector_type = DetectorType::YOLOX;
        } else if (arg.substr(0, 2) != "--") {
            base_dir = arg;
        }
    }

    fs::path model_dir = fs::path(base_dir) / "DWPose-models";
    fs::path image_dir = fs::path(base_dir) / "images";
    fs::path output_dir = fs::path(base_dir) / "results_overlay_cpp";

    fs::create_directories(output_dir);

    std::string det_model;
    switch (detector_type) {
        case DetectorType::YOLOV8:
            det_model = (fs::path(base_dir) / "yolov8n.onnx").string();
            break;
        case DetectorType::YOLO26:
            det_model = (fs::path(base_dir) / "yolo26s.onnx").string();
            break;
        default:
            det_model = (model_dir / "yolox_l.onnx").string();
            break;
    }
    std::string pose_model = (model_dir / "dw-ll_ucoco_384.onnx").string();

    std::cout << "============================================================" << std::endl;
    std::cout << "DWPose C++ Inference" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Detector: " << detector_name(detector_type) << std::endl;
    std::cout << "Pose:     DWPose (dw-ll_ucoco_384)" << std::endl;
    std::cout << "Image dir: " << image_dir << std::endl;

    // collect test images
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        auto ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".png") {
            image_paths.push_back(entry.path().string());
        }
    }
    std::cout << "Found " << image_paths.size() << " images" << std::endl;

    // load models
    std::cout << std::endl << "[Loading models]" << std::endl;

    Timer load_timer;
    std::unique_ptr<YOLOXDetector> yolox_detector;
    std::unique_ptr<YOLOv8Detector> yolov8_detector;
    std::unique_ptr<YOLO26Detector> yolo26_detector;

    switch (detector_type) {
        case DetectorType::YOLOV8:
            yolov8_detector = std::make_unique<YOLOv8Detector>(det_model);
            break;
        case DetectorType::YOLO26:
            yolo26_detector = std::make_unique<YOLO26Detector>(det_model);
            break;
        default:
            yolox_detector = std::make_unique<YOLOXDetector>(det_model);
            break;
    }
    DWPoseEstimator pose_estimator(pose_model);
    double load_time = load_timer.elapsed_ms();
    std::cout << "  Model loading time: " << std::fixed << std::setprecision(1) << load_time << " ms" << std::endl;

    std::cout << std::endl << "[Inference]" << std::endl;

    double total_det_time = 0;
    double total_pose_time = 0;
    double total_draw_time = 0;

    for (const auto& img_path : image_paths) {
        fs::path p(img_path);
        std::cout << std::endl << "Processing: " << p.filename().string() << std::endl;

        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            std::cout << "  Error: Cannot read image" << std::endl;
            continue;
        }

        // detection
        Timer det_timer;
        std::vector<DetectBox> boxes;
        switch (detector_type) {
            case DetectorType::YOLOV8:
                boxes = yolov8_detector->detect(img);
                break;
            case DetectorType::YOLO26:
                boxes = yolo26_detector->detect(img);
                break;
            default:
                boxes = yolox_detector->detect(img);
                break;
        }
        double det_time = det_timer.elapsed_ms();
        total_det_time += det_time;
        std::cout << "  Detection: " << boxes.size() << " persons, " << det_time << " ms" << std::endl;

        if (boxes.empty()) {
            std::cout << "  No person detected, skipping" << std::endl;
            continue;
        }

        // convert DetectBox to vector<vector<float>> for pose estimator
        std::vector<std::vector<float>> bboxes;
        for (const auto& b : boxes) {
            bboxes.push_back({b.x1, b.y1, b.x2, b.y2});
        }

        // pose estimation
        Timer pose_timer;
        auto all_kps = pose_estimator.estimate(img, bboxes);
        double pose_time = pose_timer.elapsed_ms();
        total_pose_time += pose_time;

        std::cout << "  Pose estimation: " << all_kps.size() << " persons, " << pose_time << " ms";
        if (!all_kps.empty()) {
            float max_score = 0;
            for (const auto& kps : all_kps)
                for (const auto& kp : kps)
                    max_score = std::max(max_score, kp.score);
            std::cout << ", max score: " << std::setprecision(3) << max_score;
        }
        std::cout << std::endl;

        // draw on original image
        Timer draw_timer;
        cv::Mat result = img.clone();
        for (const auto& kps : all_kps) {
            draw_pose(result, kps);
        }
        double draw_time = draw_timer.elapsed_ms();
        total_draw_time += draw_time;

        // save
        fs::path out_path = output_dir / (p.stem().string() + ".jpg");
        cv::imwrite(out_path.string(), result);
        std::cout << "  Draw & save: " << draw_time << " ms -> " << out_path.filename().string() << std::endl;
    }

    // Summary
    std::cout << std::endl << "============================================================" << std::endl;
    std::cout << "[Summary]" << std::endl;
    std::cout << "  Detector:            " << detector_name(detector_type) << std::endl;
    std::cout << "  Images processed:    " << image_paths.size() << std::endl;
    std::cout << "  Total detection:     " << std::fixed << std::setprecision(1) << total_det_time << " ms" << std::endl;
    std::cout << "  Total pose:          " << total_pose_time << " ms" << std::endl;
    std::cout << "  Total draw & save:   " << total_draw_time << " ms" << std::endl;
    std::cout << "  Total inference:     " << (total_det_time + total_pose_time + total_draw_time) << " ms" << std::endl;
    std::cout << "  Avg per image:       " << (total_det_time + total_pose_time + total_draw_time) / image_paths.size() << " ms" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Results saved in: " << output_dir << std::endl;

    return 0;
}