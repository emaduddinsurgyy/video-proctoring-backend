import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import config

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ========== CONFIGURABLE PROCTORING PARAMETERS ==========

# 📷 Minimum percentage of face area (of total frame) that must be visible to be considered clear
MIN_COVERAGE = 0.06              # 6% = stricter full visibility

# ⚠️ Acceptable partial face threshold, below which it's considered insufficient visibility
PARTIAL_COVERAGE = 0.03          # 3%

# 🔁 Movement difference (in pixels) allowed between frames
MOVEMENT_THRESHOLD = 200         # Lower = stricter movement detection

# 🔲 How far the face can be from edges (ignored now but left as option)
EDGE_MARGIN_RATIO = 0.04         # Disable this by commenting logic below

# 📐 Acceptable aspect ratio of detected face (height/width)
ASPECT_RATIO_MIN = 0.6
ASPECT_RATIO_MAX = 1.8           # Adjust if tilted faces are common

# 🔄 Warnings required before triggering a violation
VIOLATION_TRIGGER = 2            # Every 2 local violations = 1 actual warning

# ========================================================

prev_box = None
repeated_violation_count = 0

# ====== LOAD FACE DETECTION MODEL ======
print("[INIT] Loading face detection model...")
face_net = cv2.dnn.readNetFromCaffe(config.RES10_PROTO_PATH, config.RES10_MODEL_PATH)
print("[INIT] Model loaded successfully.")

# ====== FRAME PROCESSING FUNCTION ======
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
        messages.append("⚠️ Multiple faces detected.")
        prev_box = None
    else:
        (x1, y1, x2, y2) = faces[0]
        face_area = (x2 - x1) * (y2 - y1)
        coverage = face_area / (frame_w * frame_h)
        aspect_ratio = (y2 - y1) / max((x2 - x1), 1)

        # ✅ Coverage check
        if coverage < PARTIAL_COVERAGE:
            messages.append("⚠️ Face not clearly visible.")
            local_violations += 1
        elif coverage < MIN_COVERAGE:
            messages.append("⚠️ Face partially visible.")
            local_violations += 1

        # ✅ Angle check
        if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            messages.append("⚠️ Face angle is unusual. Look straight.")

        # ✅ Movement detection
        current_box = [x1, y1, x2, y2]
        if prev_box is not None:
            movement = np.sum(np.abs(np.array(current_box) - np.array(prev_box)))
            if movement > MOVEMENT_THRESHOLD:
                messages.append("⚠️ Excessive movement detected.")
                local_violations += 1
        prev_box = current_box

    # ✅ Final warning logic
    repeated_violation_count += local_violations
    if repeated_violation_count >= VIOLATION_TRIGGER:
        repeated_violation_count = 0
        return ["⚠️ Repeated face/movement violation."]

    return messages if messages else ["✅ All clear"]

# ====== ROUTES ======
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

# ====== SOCKET EVENTS ======
@socketio.on("connect")
def on_connect():
    print("[INFO] WebSocket client connected")
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
    print("[INFO] WebSocket client disconnected")

# ====== RUN SERVER ======
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
# To run the server, use the command:
# python app.py 