#include "MainWindow.h"
#include <QVBoxLayout>
#include <QWidget>
#include <QFont>
#include <QPixmap>

MainWindow::MainWindow(const QString& model_path, int camera_idx, int width, int height,
                       float ema_alpha, QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("Head Pose Estimation");

    auto* central = new QWidget(this);
    auto* layout  = new QVBoxLayout(central);
    layout->setSpacing(4);
    layout->setContentsMargins(8, 8, 8, 8);

    video_label_ = new QLabel(this);
    video_label_->setFixedSize(width, height);
    video_label_->setAlignment(Qt::AlignCenter);
    video_label_->setStyleSheet("background: #111;");

    pose_label_ = new QLabel("Waiting for face...", this);
    QFont font("Monospace", 11);
    font.setStyleHint(QFont::TypeWriter);
    pose_label_->setFont(font);
    pose_label_->setAlignment(Qt::AlignCenter);

    layout->addWidget(video_label_);
    layout->addWidget(pose_label_);
    setCentralWidget(central);
    adjustSize();

    camera_thread_ = new CameraThread(model_path, camera_idx, width, height, ema_alpha, this);
    connect(camera_thread_, &CameraThread::frameReady,
            this,           &MainWindow::onFrameReady);
    connect(camera_thread_, &CameraThread::poseUpdated,
            this,           &MainWindow::onPoseUpdated);
    camera_thread_->start();
}

MainWindow::~MainWindow() {
    camera_thread_->stop();
    camera_thread_->wait();
}

void MainWindow::onFrameReady(QImage frame) {
    video_label_->setPixmap(
        QPixmap::fromImage(frame).scaled(
            video_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::onPoseUpdated(float yaw, float pitch, float roll, float fps) {
    pose_label_->setText(
        QString("Yaw: %1°   Pitch: %2°   Roll: %3°     FPS: %4")
            .arg(yaw,   0, 'f', 1)
            .arg(pitch, 0, 'f', 1)
            .arg(roll,  0, 'f', 1)
            .arg(fps,   0, 'f', 1));
}
