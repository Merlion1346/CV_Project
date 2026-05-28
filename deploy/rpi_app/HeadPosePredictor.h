#pragma once
#include <array>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class HeadPosePredictor {
public:
    explicit HeadPosePredictor(const std::string& model_path);

    // Returns {yaw, pitch, roll} in degrees
    std::array<float, 3> predict(const cv::Mat& face_bgr);

private:
    static constexpr int   IMG_SIZE = 224;
    static constexpr float MEAN[3]  = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3]   = {0.229f, 0.224f, 0.225f};

    Ort::Env     env_;
    Ort::Session session_;
    std::string  input_name_;
    std::string  output_name_;

    std::vector<float> preprocess(const cv::Mat& face_bgr);
};
