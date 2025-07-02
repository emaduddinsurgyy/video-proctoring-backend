from flask import Flask, request, jsonify
import cv2
import numpy as np
import config
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ====== CONFIG ======
FACE_COVERAGE_THRESHOLD = 0.05
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 2.0
EDGE_MARGIN_RATIO = 0.04
MOVEMENT_THRESHOLD = 100

prev_box = None

print("[INIT] Loading face detection model...")
face_net = cv2.dnn.readNetFromCaffe(config.RES10_PROTO_PATH, config.RES10_MODEL_PATH)
print("[INIT] Model loaded successfully.")

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "Video proctoring server is running."})


@app.route("/analyze", methods=["POST"])
def analyze():
    global prev_box

    image = request.files.get("image")
    if not image:
        return jsonify({"error": "Missing image"}), 400

    frame = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
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
            faces.append(np.clip(box.astype("int"), 0, [w, h, w, h]))

    messages = []

    if len(faces) == 0:
        messages.append("⚠️ No face detected.")
    elif len(faces) > 1:
        messages.append("⚠️ Multiple faces detected.")
    else:
        (startX, startY, endX, endY) = faces[0]
        face_area = (endX - startX) * (endY - startY)
        coverage_ratio = face_area / (h * w)
        aspect_ratio = (endY - startY) / max((endX - startX), 1)

        if coverage_ratio < FACE_COVERAGE_THRESHOLD:
            messages.append("⚠️ Face is too small or far.")
        elif not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            messages.append("⚠️ Face angle/visibility issue.")
        elif (
            startX < w * EDGE_MARGIN_RATIO or endX > w * (1 - EDGE_MARGIN_RATIO)
            or startY < h * EDGE_MARGIN_RATIO or endY > h * (1 - EDGE_MARGIN_RATIO)
        ):
            messages.append("⚠️ Face too close to frame edge.")

        if prev_box is not None:
            movement = np.sum(np.abs(np.array([startX, startY, endX, endY]) - np.array(prev_box)))
            if movement > MOVEMENT_THRESHOLD:
                messages.append("⚠️ Excessive movement detected.")
        prev_box = [startX, startY, endX, endY]

    result = " | ".join(messages) if messages else "✅ All clear"
    return jsonify({"warning": result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
