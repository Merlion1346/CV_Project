import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var permissionDenied = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // 카메라 프리뷰
            CameraPreviewView(session: camera.session)
                .ignoresSafeArea()

            // 자세 오버레이
            GeometryReader { _ in
                HeadPoseOverlay(predictions: camera.predictions)
            }
            .ignoresSafeArea()

            // 하단 HUD
            VStack {
                Spacer()
                AngleHUD(predictions: camera.predictions)
                    .padding(.bottom, 40)
            }

            // 권한 거부 안내
            if permissionDenied {
                permissionDeniedView
            }
        }
        .onAppear {
            checkPermissionAndStart()
        }
        .onDisappear {
            camera.stopSession()
        }
    }

    // ── 권한 확인 ─────────────────────────────────────────────
    private func checkPermissionAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            camera.startSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted { camera.startSession() }
                    else { permissionDenied = true }
                }
            }
        default:
            permissionDenied = true
        }
    }

    private var permissionDeniedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill")
                .font(.system(size: 48))
                .foregroundColor(.gray)
            Text("카메라 권한이 필요합니다")
                .font(.headline)
                .foregroundColor(.white)
            Text("설정 > HeadPoseApp > 카메라 에서 허용해 주세요.")
                .font(.caption)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
        }
        .padding(32)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding()
    }
}

// ── 하단 각도 패널 ────────────────────────────────────────────
struct AngleHUD: View {
    let predictions: [FacePrediction]

    var body: some View {
        if let pred = predictions.first {
            HStack(spacing: 28) {
                AngleIndicator(label: "Yaw",   value: pred.yaw,   color: .red)
                AngleIndicator(label: "Pitch", value: pred.pitch, color: .green)
                AngleIndicator(label: "Roll",  value: pred.roll,  color: Color(red: 0.4, green: 0.7, blue: 1.0))
                Divider()
                    .frame(height: 36)
                    .background(Color.white.opacity(0.3))
                Text(pred.direction)
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(directionColor(pred.direction))
                    .frame(width: 56)
            }
            .padding(.horizontal, 28)
            .padding(.vertical, 14)
            .background(.ultraThinMaterial, in: Capsule())
            .transition(.opacity.combined(with: .move(edge: .bottom)))
        }
    }

    private func directionColor(_ d: String) -> Color {
        switch d {
        case "Front": return .green
        case "Left":  return .orange
        case "Right": return Color(red: 0.2, green: 0.6, blue: 1.0)
        case "Up":    return .yellow
        case "Down":  return .cyan
        default:      return .white
        }
    }
}

struct AngleIndicator: View {
    let label: String
    let value: Float
    let color: Color

    var body: some View {
        VStack(spacing: 3) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(String(format: "%+.1f°", value))
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .foregroundColor(color)
        }
    }
}
