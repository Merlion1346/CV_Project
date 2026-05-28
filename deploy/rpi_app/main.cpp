#include <QApplication>
#include <QCommandLineParser>
#include <QMessageBox>
#include "MainWindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("HeadPoseRPi");

    QCommandLineParser p;
    p.addHelpOption();
    p.addOption({{"m", "model"},  "Path to .onnx model",    "model"});
    p.addOption({{"c", "camera"}, "Camera device index",    "camera", "0"});
    p.addOption({"width",         "Capture width",           "px",    "640"});
    p.addOption({"height",        "Capture height",          "px",    "480"});
    p.addOption({"ema",           "EMA smoothing (0.1–0.4)", "alpha", "0.2"});
    p.process(app);

    if (!p.isSet("model")) {
        QMessageBox::critical(nullptr, "Error",
            "Usage: HeadPoseRPi --model <path.onnx>");
        return 1;
    }

    MainWindow w(
        p.value("model"),
        p.value("camera").toInt(),
        p.value("width").toInt(),
        p.value("height").toInt(),
        p.value("ema").toFloat()
    );
    w.show();
    return app.exec();
}
