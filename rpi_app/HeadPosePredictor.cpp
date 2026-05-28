#include "HeadPosePredictor.h"

HeadPosePredictor::HeadPosePredictor(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "HeadPose")
    , session_(env_, model_path.c_str(), Ort::SessionOptions{})
{
    Ort::AllocatorWithDefaultOptions alloc;
    {
        auto name  = session_.GetInputNameAllocated(0, alloc);
        input_name_ = name.get();
    }
    {
        auto name   = session_.GetOutputNameAllocated(0, alloc);
        output_name_ = name.get();
    }
}

std::vector<float> HeadPosePredictor::preprocess(const cv::Mat& face_bgr) {
    cv::Mat rgb, resized;
    cv::cvtColor(face_bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, {IMG_SIZE, IMG_SIZE});
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> ch(3);
    cv::split(resized, ch);
    for (int c = 0; c < 3; c++)
        ch[c] = (ch[c] - MEAN[c]) / STD[c];

    // HWC → CHW
    std::vector<float> data(3 * IMG_SIZE * IMG_SIZE);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < IMG_SIZE; h++)
            for (int w = 0; w < IMG_SIZE; w++)
                data[c * IMG_SIZE * IMG_SIZE + h * IMG_SIZE + w] = ch[c].at<float>(h, w);
    return data;
}

std::array<float, 3> HeadPosePredictor::predict(const cv::Mat& face_bgr) {
    auto data = preprocess(face_bgr);

    const std::array<int64_t, 4> shape = {1, 3, IMG_SIZE, IMG_SIZE};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor   = Ort::Value::CreateTensor<float>(
        mem_info, data.data(), data.size(), shape.data(), shape.size());

    const char* in_names[]  = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

    const float* r = outputs[0].GetTensorData<float>();
    return {r[0], r[1], r[2]};  // yaw, pitch, roll in degrees
}
