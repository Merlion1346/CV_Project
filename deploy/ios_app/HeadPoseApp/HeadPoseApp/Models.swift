import Foundation
import CoreGraphics

struct FacePrediction {
    /// Vision 좌표계 (origin: 좌하단, normalized 0–1)
    let boundingBox: CGRect
    let yaw:   Float
    let pitch: Float
    let roll:  Float

    var direction: String {
        if abs(pitch) >= 15 { return pitch > 0 ? "Up" : "Down" }
        if abs(yaw)   <= 15 { return "Front" }
        return yaw < 0 ? "Left" : "Right"
    }
}
