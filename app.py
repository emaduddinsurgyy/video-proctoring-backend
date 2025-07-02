from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import config  # Ensure this has RES10_PROTO_PATH and RES10_MODEL_PATH

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ====== CONFIG ======
MIN_COVERAGE = 0.08
PARTIAL_COVERAGE = 0.04
MOVEMENT_THRESHOLD = 150
EDGE_MARGIN_RATIO = 0.04
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 2.0

prev_box = None
repeated_violation_count = 0  # Track face/movement warnings

# ====== LOAD FACE DETECTION MODEL ======
print("[INIT] Loading face detection model...")
face_net = cv2.dnn.readNetFromCaffe(config.RES10_PROTO_PATH, config.RES10_MODEL_PATH)
print("[INIT] Model loaded successfully.")

# ====== ANALYSIS FUNCTION ======
def process_frame(frame, frame_w, frame_h):
    global prev_box, repeated_violation_count
    messages = []
    local_violations = 0

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    try:
        face_net.setInput(blob)
        detections = face_net.forward()
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return ["⚠️ Face detection error"]

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            box = np.clip(box.astype("int"), 0, [frame_w, frame_h, frame_w, frame_h])
            faces.append(box)

    if len(faces) == 0:
        messages.append("⚠️ No face detected.")
        prev_box = None
    elif len(faces) > 1:
        messages.append("⚠️ Multiple faces detected.")  # Excluded from repeat count
        prev_box = None
    else:
        (x1, y1, x2, y2) = faces[0]
        face_area = (x2 - x1) * (y2 - y1)
        coverage = face_area / (frame_w * frame_h)
        aspect_ratio = (y2 - y1) / max((x2 - x1), 1)

        # Face coverage warnings
        if coverage < PARTIAL_COVERAGE:
            messages.append("⚠️ Face not clearly visible. Please face the camera.")
            local_violations += 1
        elif coverage < MIN_COVERAGE:
            messages.append("⚠️ Face partially visible. Adjust your position.")
            local_violations += 1

        # Angle
        if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            messages.append("⚠️ Unusual face angle. Look straight at the screen.")

        # Frame edge
        margin_x = frame_w * EDGE_MARGIN_RATIO
        margin_y = frame_h * EDGE_MARGIN_RATIO
        if (x1 < margin_x or x2 > frame_w - margin_x or
            y1 < margin_y or y2 > frame_h - margin_y):
            messages.append("⚠️ Face is too close to screen edge.")

        # Movement detection
        current_box = [x1, y1, x2, y2]
        if prev_box is not None:
            movement = np.sum(np.abs(np.array(current_box) - np.array(prev_box)))
            if movement > MOVEMENT_THRESHOLD:
                messages.append("⚠️ Excessive movement detected.")
                local_violations += 1
        prev_box = current_box

    # Increment global count for only relevant issues
    repeated_violation_count += local_violations

    # Condensed warning every 4 issues
    if repeated_violation_count >= 4:
        repeated_violation_count = 0
        return ["⚠️ Repeated face/movement violation detected. Please stay steady and visible."]

    return messages if messages else ["✅ All clear"]

# ====== HTTP ROUTES ======
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Video proctoring server is running."})

@app.route("/analyze", methods=["POST"])
def analyze():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "Missing image file"}), 400

    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    h, w = frame.shape[:2]
    messages = process_frame(frame, w, h)
    return jsonify({"warning": " | ".join(messages)})

# ====== WEBSOCKET EVENTS ======
@socketio.on("connect")
def on_connect():
    print("[INFO] Client connected")
    emit("connected", {"message": "WebSocket connected successfully"})

@socketio.on("frame")
def on_frame(data):
    try:
        encoded = data["image"].split(",")[1] if "," in data["image"] else data["image"]
        img_bytes = base64.b64decode(encoded)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            emit("warning", {"message": "⚠️ Invalid image format"})
            return

        h, w = frame.shape[:2]
        messages = process_frame(frame, w, h)
        emit("warning", {"message": " | ".join(messages)})

    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        emit("warning", {"message": "⚠️ Frame processing error"})

@socketio.on("disconnect")
def on_disconnect():
    print("[INFO] Client disconnected")

# ====== MAIN ======
if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
