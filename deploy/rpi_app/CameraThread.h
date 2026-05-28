#pragma once
#include <QThread>
#include <QImage>
#include <atomic>
#include <opencv2/opencv.hpp>

class CameraThread : public QThread {
    Q_OBJECT
public:
    explicit CameraThread(const QString& model_path, int camera_idx = 0,
                          int width = 640, int height = 480,
                          float ema_alpha = 0.2f, QObject* parent = nullptr);
    void stop();

signals:
    void frameReady(QImage frame);
    void poseUpdated(float yaw, float pitch, float roll, float fps);

protected:
    void run() override;

private:
    static void drawAxes(cv::Mat& img, float yaw, float pitch, float roll,
                         int cx, int cy, int size);

    QString            model_path_;
    int                camera_idx_;
    int                width_;
    int                height_;
    float              ema_alpha_;
    std::atomic<bool>  running_{true};
};
