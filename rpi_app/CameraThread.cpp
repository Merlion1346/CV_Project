#include "CameraThread.h"
#include "HeadPosePredictor.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

CameraThread::CameraThread(const QString& model_path, int camera_idx,
                           int width, int height,
                           float ema_alpha, QObject* parent)
    : QThread(parent)
    , model_path_(model_path)
    , camera_idx_(camera_idx)
    , width_(width)
    , height_(height)
    , ema_alpha_(ema_alpha)
{}

void CameraThread::stop() { running_ = false; }

// ─── helpers ───────────────────────────────────────────────────────────────

struct DirInfo { const char* label; cv::Scalar color; };

static DirInfo angleToDirection(float yaw, float pitch) {
    if (std::abs(pitch) >= 15.f)
        return pitch > 0 ? DirInfo{"Up",    {0, 220, 220}}
                         : DirInfo{"Down",  {200, 200, 0}};
    if (std::abs(yaw) <= 15.f)
        return {"Front", {0, 220, 0}};
    return yaw < 0 ? DirInfo{"Left",  {255, 80, 0}}
                   : DirInfo{"Right", {0, 80, 255}};
}

void CameraThread::drawAxes(cv::Mat& img, float yaw, float pitch, float roll,
                             int cx, int cy, int size) {
    const float yr = yaw   * static_cast<float>(M_PI) / 180.f;
    const float pr = pitch * static_cast<float>(M_PI) / 180.f;
    const float rr = roll  * static_cast<float>(M_PI) / 180.f;

    cv::Matx33f Rz( std::cos(rr), -std::sin(rr), 0,
                    std::sin(rr),  std::cos(rr), 0,
                    0,             0,             1);
    cv::Matx33f Ry( std::cos(yr),  0,  std::sin(yr),
                    0,             1,  0,
                   -std::sin(yr),  0,  std::cos(yr));
    cv::Matx33f Rx( 1,  0,              0,
                    0,  std::cos(pr), -std::sin(pr),
                    0,  std::sin(pr),  std::cos(pr));
    cv::Matx33f R = Rz * Ry * Rx;

    const cv::Scalar colors[3] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}};
    for (int i = 0; i < 3; i++) {
        cv::Vec3f axis(0.f, 0.f, 0.f);
        axis[i] = static_cast<float>(size);
        cv::Vec3f rot = R * axis;
        cv::arrowedLine(img, {cx, cy},
                        {cx + static_cast<int>(rot[0]), cy - static_cast<int>(rot[1])},
                        colors[i], 2, cv::LINE_AA, 0, 0.3);
    }
}

static std::string findCascade() {
    // Search common OpenCV data directories on Debian/Raspbian
    for (const char* dir : {
            "/usr/share/opencv4/haarcascades/",
            "/usr/share/opencv/haarcascades/",
            "/usr/local/share/opencv4/haarcascades/",
            "/usr/local/share/opencv/haarcascades/"}) {
        std::string path = std::string(dir) + "haarcascade_frontalface_default.xml";
        if (std::filesystem::exists(path))
            return path;
    }
    return {};
}

// ─── main loop ─────────────────────────────────────────────────────────────

void CameraThread::run() {
    HeadPosePredictor predictor(model_path_.toStdString());

    std::string cascade_path = findCascade();
    cv::CascadeClassifier face_cascade;
    if (cascade_path.empty() || !face_cascade.load(cascade_path)) {
        qWarning("CameraThread: Haar cascade not found. Install libopencv-data.");
        return;
    }

    cv::VideoCapture cap(camera_idx_);
    if (!cap.isOpened()) {
        qWarning("CameraThread: cannot open camera %d", camera_idx_);
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    qInfo("CameraThread: resolution %dx%d",
          static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
          static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    std::array<float, 3> ema = {0.f, 0.f, 0.f};
    bool ema_init = false;
    auto t_prev   = std::chrono::steady_clock::now();

    while (running_) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, {60, 60});

        float last_yaw = 0.f, last_pitch = 0.f, last_roll = 0.f;
        bool  face_found = false;

        for (const auto& f : faces) {
            int pad = static_cast<int>(0.15f * std::min(f.width, f.height));
            int x1  = std::max(0, f.x - pad);
            int y1  = std::max(0, f.y - pad);
            int x2  = std::min(frame.cols, f.x + f.width  + pad);
            int y2  = std::min(frame.rows, f.y + f.height + pad);
            cv::Mat face = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            if (face.empty()) continue;

            auto angles = predictor.predict(face);
            if (!ema_init) { ema = angles; ema_init = true; }
            else {
                for (int i = 0; i < 3; i++)
                    ema[i] = ema_alpha_ * angles[i] + (1.f - ema_alpha_) * ema[i];
            }

            float yaw = ema[0], pitch = ema[1], roll = ema[2];
            last_yaw = yaw; last_pitch = pitch; last_roll = roll;
            face_found = true;

            auto [dir, color] = angleToDirection(yaw, pitch);
            char angle_text[64];
            std::snprintf(angle_text, sizeof(angle_text),
                          "Yaw:%+.1f  Pitch:%+.1f  Roll:%+.1f", yaw, pitch, roll);

            cv::rectangle(frame, {x1, y1}, {x2, y2}, color, 2);
            cv::putText(frame, dir, {x1, y1 - 10},
                        cv::FONT_HERSHEY_DUPLEX, 0.75, color, 2);
            cv::putText(frame, angle_text, {x1, y2 + 20},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, {200, 200, 200}, 1);
            drawAxes(frame, yaw, pitch, roll,
                     x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2,
                     std::min(f.width, f.height) / 3);
        }

        // FPS counter
        auto  t_now = std::chrono::steady_clock::now();
        float fps   = 1.f / std::max(
            std::chrono::duration<float>(t_now - t_prev).count(), 1e-6f);
        t_prev = t_now;

        char fps_text[32];
        std::snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
        cv::putText(frame, fps_text, {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 200}, 2);

        // BGR → RGB then wrap in QImage (copy before frame goes out of scope)
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        emit frameReady(QImage(rgb.data, rgb.cols, rgb.rows,
                               static_cast<int>(rgb.step),
                               QImage::Format_RGB888).copy());

        if (face_found)
            emit poseUpdated(last_yaw, last_pitch, last_roll, fps);
    }

    cap.release();
}
