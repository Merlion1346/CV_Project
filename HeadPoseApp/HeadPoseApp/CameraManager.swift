import AVFoundation
import Combine

final class CameraManager: NSObject, ObservableObject,
                           AVCaptureVideoDataOutputSampleBufferDelegate {

    @Published var predictions: [FacePrediction] = []

    let session = AVCaptureSession()
    private let predictor  = HeadPosePredictor()
    private let videoQueue = DispatchQueue(label: "headpose.video", qos: .userInteractive)

    // ── Session setup ─────────────────────────────────────────
    func startSession() {
        guard !session.isRunning else { return }

        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        // 전면 카메라
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                  for: .video, position: .front),
            let input  = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        // 비디오 출력
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)

        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(output)

        // 세로 방향 + 미러링
        if let conn = output.connection(with: .video) {
            if conn.isVideoRotationAngleSupported(90) {
                conn.videoRotationAngle = 90
            }
            conn.isVideoMirrored = true
        }

        session.commitConfiguration()
        DispatchQueue.global(qos: .userInitiated).async { self.session.startRunning() }
    }

    func stopSession() {
        session.stopRunning()
    }

    // ── Frame callback ────────────────────────────────────────
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let preds = predictor.predict(pixelBuffer: pixelBuffer)
        DispatchQueue.main.async { self.predictions = preds }
    }
}
