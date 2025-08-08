from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time

RES10_PROTO_PATH = "deploy.prototxt.txt"
RES10_MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

MIN_COVERAGE = 0.07         # 7% of frame area
PARTIAL_COVERAGE = 0.04
ASPECT_RATIO_MIN = 0.6
ASPECT_RATIO_MAX = 1.8
MOVEMENT_THRESHOLD = 200
WARNINGS_PER_VIOLATION = 1
LAST_N_WARNINGS = 3

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

face_net = cv2.dnn.readNetFromCaffe(RES10_PROTO_PATH, RES10_MODEL_PATH)
session_stats = {}  # session_id: { "count": int, "history": [], "prev_box": None }

def analyze_frame(frame, session_id):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    warnings = []
    face_boxes = []
    max_coverage = 0
    best_box = None

    # Stricter thresholds
    CONF_THRESHOLD = 0.7
    TOO_CLOSE_COVERAGE = 0.8    # Face too close if coverage exceeds 80%
    TOO_FAR_COVERAGE = 0.07     # Face too far if coverage below 7%
    MOVEMENT_THRESHOLD_STRICT = 100  # More strict movement threshold

    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > CONF_THRESHOLD:
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            box = box.astype("int").tolist()
            face_boxes.append(box)
            x1, y1, x2, y2 = box
            face_area = (x2 - x1) * (y2 - y1)
            coverage = face_area / (w * h)
            if coverage > max_coverage:
                max_coverage = coverage
                best_box = box

    if len(face_boxes) == 0:
        warnings.append("⚠️ No face detected")
        session_stats[session_id]["prev_box"] = None

    elif len(face_boxes) > 1:
        warnings.append("⚠️ Multiple faces detected")
        session_stats[session_id]["prev_box"] = None

    else:
        x1, y1, x2, y2 = best_box
        coverage = (x2 - x1) * (y2 - y1) / (w * h)
        aspect_ratio = (y2 - y1) / max((x2 - x1), 1)

        if coverage > TOO_CLOSE_COVERAGE:
            warnings.append("⚠️ Face too close to camera")
        elif coverage < TOO_FAR_COVERAGE:
            warnings.append("⚠️ Face too far from camera")
        elif coverage < PARTIAL_COVERAGE:
            warnings.append("⚠️ Face not clearly visible")
        elif coverage < MIN_COVERAGE:
            warnings.append("⚠️ Face partially visible")

        if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            warnings.append("⚠️ Face angle is unusual. Look straight.")

        prev = session_stats[session_id]["prev_box"]
        if prev is not None:
            movement = sum(abs(np.array([x1, y1, x2, y2]) - np.array(prev)))
            if movement > MOVEMENT_THRESHOLD_STRICT:
                warnings.append("⚠️ Excessive movement detected")
        session_stats[session_id]["prev_box"] = [x1, y1, x2, y2]

    return warnings

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Video proctoring backend running."})

# REST polling endpoint
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    frame_b64 = data.get("frame")
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",")[1]
    frame_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if session_id not in session_stats:
        session_stats[session_id] = {"count": 0, "history": [], "prev_box": None}

    warnings = analyze_frame(frame, session_id)

    if warnings:
        session_stats[session_id]["count"] += len(warnings) * WARNINGS_PER_VIOLATION
        session_stats[session_id]["history"].extend(warnings)

    session_stats[session_id]["history"] = session_stats[session_id]["history"][-LAST_N_WARNINGS:]

    resp = {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "warning_count": session_stats[session_id]["count"],
        "last_warnings": session_stats[session_id]["history"][:],
        "timestamp": time.time()
    }
    return jsonify(resp)

# WebSocket real-time events
@socketio.on("frame")
def on_frame(data):
    session_id = data.get("session_id", "default")
    frame_b64 = data.get("frame") or data.get("image")
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",")[1]
    frame_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if session_id not in session_stats:
        session_stats[session_id] = {"count": 0, "history": [], "prev_box": None}

    warnings = analyze_frame(frame, session_id)

    if warnings:
        session_stats[session_id]["count"] += len(warnings) * WARNINGS_PER_VIOLATION
        session_stats[session_id]["history"].extend(warnings)

    session_stats[session_id]["history"] = session_stats[session_id]["history"][-LAST_N_WARNINGS:]

    emit("warning", {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "warning_count": session_stats[session_id]["count"],
        "last_warnings": session_stats[session_id]["history"][:],
        "timestamp": time.time()
    })

@socketio.on("connect")
def on_connect():
    emit("connected", {"message": "WebSocket connected."})

@socketio.on("disconnect")
def on_disconnect():
    print("[INFO] WebSocket client disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
