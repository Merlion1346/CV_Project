import SwiftUI

/// 얼굴 검출 박스 + 3D 자세 축 + 각도 텍스트 오버레이
struct HeadPoseOverlay: View {
    let predictions: [FacePrediction]

    var body: some View {
        Canvas { ctx, size in
            for pred in predictions {
                let box = visionToScreen(pred.boundingBox, size: size)
                let color = directionColor(pred.direction)

                // 얼굴 박스
                ctx.stroke(
                    Path(box.insetBy(dx: 0, dy: 0)),
                    with: .color(color),
                    lineWidth: 2.5
                )

                // 방향 레이블
                ctx.draw(
                    Text(pred.direction)
                        .font(.system(size: 18, weight: .bold))
                        .foregroundColor(color),
                    at: CGPoint(x: box.minX, y: box.minY - 26),
                    anchor: .bottomLeading
                )

                // 각도 텍스트
                let angleStr = String(
                    format: "Y %+.1f°  P %+.1f°  R %+.1f°",
                    pred.yaw, pred.pitch, pred.roll
                )
                ctx.draw(
                    Text(angleStr)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.white),
                    at: CGPoint(x: box.minX, y: box.maxY + 6),
                    anchor: .topLeading
                )

                // 3D 자세 축
                let axisLen = min(box.width, box.height) * 0.45
                drawAxes(ctx: ctx,
                         cx: box.midX, cy: box.midY,
                         yaw: Double(pred.yaw),
                         pitch: Double(pred.pitch),
                         roll: Double(pred.roll),
                         length: axisLen)
            }
        }
    }

    // ── 3D Axes ───────────────────────────────────────────────
    private func drawAxes(ctx: GraphicsContext,
                          cx: CGFloat, cy: CGFloat,
                          yaw: Double, pitch: Double, roll: Double,
                          length: CGFloat) {
        let yr = yaw   * .pi / 180
        let pr = pitch * .pi / 180
        let rr = roll  * .pi / 180

        // Rz (roll) * Ry (yaw) * Rx (pitch)
        let cosY = cos(yr), sinY = sin(yr)
        let cosP = cos(pr), sinP = sin(pr)
        let cosR = cos(rr), sinR = sin(rr)

        // 각 축 단위 벡터 회전 후 2D 투영
        // X축 (1,0,0) → 빨강
        let xx = cosR * cosY + sinR * sinP * sinY
        let xy = sinR * cosY - cosR * sinP * sinY

        // Y축 (0,1,0) → 초록
        let yx = -sinR * cosP
        let yy =  cosR * cosP

        // Z축 (0,0,1) → 파랑 (앞/뒤 방향)
        let zx = cosR * sinY - sinR * sinP * cosY
        let zy = sinR * sinY + cosR * sinP * cosY

        let axes: [(Double, Double, Color)] = [
            (xx,  xy, .red),
            (yx,  yy, .green),
            (zx,  zy, .blue),
        ]

        for (ax, ay, color) in axes {
            let end = CGPoint(
                x: cx + CGFloat(ax) * length,
                y: cy - CGFloat(ay) * length   // 화면 Y축 반전
            )
            var path = Path()
            path.move(to: CGPoint(x: cx, y: cy))
            path.addLine(to: end)
            ctx.stroke(path, with: .color(color), lineWidth: 3)

            // 화살촉
            ctx.fill(
                Circle().path(in: CGRect(x: end.x - 4, y: end.y - 4,
                                         width: 8, height: 8)),
                with: .color(color)
            )
        }
    }

    // ── Helpers ───────────────────────────────────────────────

    /// Vision 좌표 (origin 좌하단) → SwiftUI Canvas 좌표 (origin 좌상단)
    private func visionToScreen(_ box: CGRect, size: CGSize) -> CGRect {
        CGRect(
            x:      box.minX * size.width,
            y:      (1 - box.maxY) * size.height,
            width:  box.width  * size.width,
            height: box.height * size.height
        )
    }

    private func directionColor(_ direction: String) -> Color {
        switch direction {
        case "Front": return .green
        case "Left":  return .orange
        case "Right": return Color(red: 0.2, green: 0.6, blue: 1.0)
        case "Up":    return .yellow
        case "Down":  return .cyan
        default:      return .white
        }
    }
}
