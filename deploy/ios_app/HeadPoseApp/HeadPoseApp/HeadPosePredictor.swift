import Vision
import CoreML

/// Vision 얼굴 검출 + Core ML 헤드포즈 추론 파이프라인
final class HeadPosePredictor {

    private var visionModel: VNCoreMLModel?

    init() {
        // Xcode가 HeadPose.mlpackage → HeadPose.mlmodelc 로 컴파일
        guard let url = Bundle.main.url(forResource: "HeadPose", withExtension: "mlmodelc") else {
            print("[HeadPose] ⚠️  HeadPose.mlpackage 를 Xcode 프로젝트에 추가해 주세요.")
            return
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all          // ANE / GPU 자동 선택
            let ml = try MLModel(contentsOf: url, configuration: config)
            visionModel = try VNCoreMLModel(for: ml)
        } catch {
            print("[HeadPose] 모델 로드 실패: \(error)")
        }
    }

    /// 픽셀버퍼에서 얼굴을 검출하고 각 얼굴의 자세를 추론합니다.
    func predict(pixelBuffer: CVPixelBuffer) -> [FacePrediction] {
        // ── 1. 얼굴 검출 ──────────────────────────────────────
        let faceReq = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        guard (try? handler.perform([faceReq])) != nil,
              let faces = faceReq.results, !faces.isEmpty
        else { return [] }

        // ── 2. 각 얼굴에 대해 헤드포즈 추론 ───────────────────
        guard let visionModel else { return [] }
        var predictions: [FacePrediction] = []

        for face in faces {
            let mlReq = VNCoreMLRequest(model: visionModel)
            mlReq.imageCropAndScaleOption = .scaleFill
            // 15% 패딩을 추가한 얼굴 영역을 ROI로 설정
            mlReq.regionOfInterest = padded(face.boundingBox, by: 0.15)

            guard (try? handler.perform([mlReq])) != nil,
                  let obs = mlReq.results?.first as? VNCoreMLFeatureValueObservation,
                  let arr = obs.featureValue.multiArrayValue
            else { continue }

            predictions.append(FacePrediction(
                boundingBox: face.boundingBox,
                yaw:   Float(arr[0]),
                pitch: Float(arr[1]),
                roll:  Float(arr[2])
            ))
        }
        return predictions
    }

    // ── Helpers ───────────────────────────────────────────────
    private func padded(_ box: CGRect, by ratio: CGFloat) -> CGRect {
        let dx = box.width  * ratio
        let dy = box.height * ratio
        return CGRect(
            x:      max(0,  box.minX - dx),
            y:      max(0,  box.minY - dy),
            width:  min(1 - max(0, box.minX - dx), box.width  + 2 * dx),
            height: min(1 - max(0, box.minY - dy), box.height + 2 * dy)
        )
    }
}
