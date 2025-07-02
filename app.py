from flask import Flask, request, jsonify
import cv2
import numpy as np
import config
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("[INIT] Loading face detection model...")
face_net = cv2.dnn.readNetFromCaffe(config.RES10_PROTO_PATH, config.RES10_MODEL_PATH)
print("[INIT] Model loaded successfully.")

# ====== CONFIG ======
FACE_COVERAGE_THRESHOLD = 0.05  # Minimum 5% area
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 2.0
EDGE_MARGIN_RATIO = 0.04
MOVEMENT_THRESHOLD = 100

prev_box = None
warning_counts = {
    "no_face": 0,
    "multiple_faces": 0,
    "face_covered": 0,
    "movement": 0
}


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "Video proctoring server is running."})


@app.route("/analyze", methods=["POST"])
def analyze():
    global prev_box

    print("\n========== [REQUEST RECEIVED] ==========")

    image = request.files.get("image")
    if not image:
        print("[ERROR] No image received.")
        return jsonify({"error": "Missing image"}), 400

    frame = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("[ERROR] Could not decode frame.")
        return jsonify({"error": "Invalid image data"}), 400

    h, w = frame.shape[:2]

    # ====== FACE DETECTION ======
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = np.clip(box.astype("int"), 0, [w, h, w, h])
            faces.append(box)

    messages = []

    if len(faces) == 0:
        warning_counts["no_face"] += 1
        messages.append("⚠️ No face detected.")
        print("[WARN] No face detected.")
    elif len(faces) > 1:
        warning_counts["multiple_faces"] += 1
        messages.append("⚠️ Multiple faces detected.")
        print("[WARN] Multiple faces detected.")
    else:
        print("[INFO] One face detected.")
        (startX, startY, endX, endY) = faces[0]
        face_area = (endX - startX) * (endY - startY)
        total_area = h * w
        coverage_ratio = face_area / total_area
        face_width = endX - startX
        face_height = endY - startY
        aspect_ratio = face_height / max(face_width, 1)

        print(f"[DEBUG] Face coverage: {coverage_ratio:.3f}")
        print(f"[DEBUG] Face aspect ratio: {aspect_ratio:.2f}")
        print(f"[DEBUG] Face box: ({startX},{startY})-({endX},{endY})")

        # Coverage Check
        if coverage_ratio < FACE_COVERAGE_THRESHOLD:
            warning_counts["face_covered"] += 1
            messages.append("⚠️ Face is too small or far.")
            print("[WARN] Low face coverage.")

        # Aspect Ratio Check
        elif not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            warning_counts["face_covered"] += 1
            messages.append("⚠️ Face angle/visibility issue.")
            print("[WARN] Unusual face shape or angle.")

        # Edge Position Check
        elif (startX < w * EDGE_MARGIN_RATIO or endX > w * (1 - EDGE_MARGIN_RATIO) or
              startY < h * EDGE_MARGIN_RATIO or endY > h * (1 - EDGE_MARGIN_RATIO)):
            warning_counts["face_covered"] += 1
            messages.append("⚠️ Face too close to frame edge.")
            print("[WARN] Face near edge — possible partial visibility.")

        # Movement Detection
        if prev_box is not None:
            movement = np.sum(np.abs(np.array([startX, startY, endX, endY]) - np.array(prev_box)))
            print(f"[DEBUG] Movement score: {movement} (Threshold: {MOVEMENT_THRESHOLD})")
            if movement > MOVEMENT_THRESHOLD:
                warning_counts["movement"] += 1
                messages.append("⚠️ Excessive movement detected.")
        prev_box = [startX, startY, endX, endY]

    summary = " | ".join(messages) if messages else "✅ All normal"
    print(f"[🔎] Status: {summary}")
    print(f"📈 Warning Log: {warning_counts}")

    return jsonify({
        "warning": summary,
        "log": warning_counts
    })


if __name__ == "__main__":
    print("🚀 Video Proctoring Flask Server Started @ http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
