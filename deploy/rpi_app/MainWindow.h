#pragma once
#include <QMainWindow>
#include <QLabel>
#include "CameraThread.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(const QString& model_path, int camera_idx, int width, int height,
               float ema_alpha, QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void onFrameReady(QImage frame);
    void onPoseUpdated(float yaw, float pitch, float roll, float fps);

private:
    QLabel*       video_label_;
    QLabel*       pose_label_;
    CameraThread* camera_thread_;
};
