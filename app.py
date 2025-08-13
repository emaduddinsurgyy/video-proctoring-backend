from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time
import os

# Model paths - ensure these files exist
RES10_PROTO_PATH = "models/deploy.prototxt.txt"
RES10_MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Proctoring parameters
MIN_COVERAGE = 0.06         # 6% of frame area
PARTIAL_COVERAGE = 0.03
ASPECT_RATIO_MIN = 0.6
ASPECT_RATIO_MAX = 1.8
WARNINGS_PER_VIOLATION = 1
LAST_N_WARNINGS = 3

# Strict thresholds for better detection
CONF_THRESHOLD = 0.7
TOO_CLOSE_COVERAGE = 0.35   # Face too close if coverage exceeds 35%
TOO_FAR_COVERAGE = 0.02     # Face too far if coverage below 2%
MOVEMENT_THRESHOLD_STRICT = 100  # Strict movement threshold (pixels)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Check if model files exist before loading
try:
    if not os.path.exists(RES10_PROTO_PATH):
        print(f"Error: Model file not found: {RES10_PROTO_PATH}")
        print("Please ensure you have the model files in the 'models' directory")
        exit(1)
    if not os.path.exists(RES10_MODEL_PATH):
        print(f"Error: Model file not found: {RES10_MODEL_PATH}")
        print("Please ensure you have the model files in the 'models' directory")
        exit(1)
    
    print("[INIT] Loading face detection model...")
    face_net = cv2.dnn.readNetFromCaffe(RES10_PROTO_PATH, RES10_MODEL_PATH)
    print("[INIT] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load face detection model: {e}")
    exit(1)

session_stats = {}  # session_id: { "count": int, "history": [], "prev_box": None, "baseline_coverage": None }

def analyze_frame(frame, session_id):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    warnings = []
    face_boxes = []
    max_coverage = 0
    best_box = None

    # Detect faces
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

    # No face detected
    if len(face_boxes) == 0:
        warnings.append("⚠️ No face detected")
        session_stats[session_id]["prev_box"] = None
        return warnings

    # Multiple faces detected
    elif len(face_boxes) > 1:
        warnings.append("⚠️ Multiple faces detected")
        session_stats[session_id]["prev_box"] = None
        return warnings

    # Single face analysis
    else:
        x1, y1, x2, y2 = best_box
        coverage = (x2 - x1) * (y2 - y1) / (w * h)
        aspect_ratio = (y2 - y1) / max((x2 - x1), 1)

        # Initialize baseline coverage for first detection
        if session_stats[session_id]["prev_box"] is None:
            session_stats[session_id]["baseline_coverage"] = coverage

        baseline = session_stats[session_id].get("baseline_coverage", coverage)

        # Distance warnings based on coverage
        if coverage > TOO_CLOSE_COVERAGE:
            warnings.append("⚠️ Face too close to camera")
        elif coverage < TOO_FAR_COVERAGE:
            warnings.append("⚠️ Face too far from camera")
        elif coverage < baseline * 0.5:  # Face moved significantly farther
            warnings.append("⚠️ Please maintain your distance from camera")
        elif coverage > baseline * 2.0:  # Face moved significantly closer
            warnings.append("⚠️ You are too close to the camera")

        # Face visibility warnings
        if coverage < PARTIAL_COVERAGE:
            warnings.append("⚠️ Face not clearly visible")
        elif coverage < MIN_COVERAGE:
            warnings.append("⚠️ Face partially visible")

        # Face angle check
        if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            warnings.append("⚠️ Face angle is unusual. Look straight.")

        # Movement detection with improved algorithm
        prev = session_stats[session_id]["prev_box"]
        if prev is not None:
            # Calculate center movement (more accurate than box corner movement)
            prev_center_x = (prev[0] + prev[2]) / 2
            prev_center_y = (prev[1] + prev[3]) / 2
            curr_center_x = (x1 + x2) / 2
            curr_center_y = (y1 + y2) / 2
            
            center_movement = np.sqrt((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)
            
            # Also check box size change (scaling movement)
            prev_size = (prev[2] - prev[0]) * (prev[3] - prev[1])
            curr_size = (x2 - x1) * (y2 - y1)
            size_change_ratio = abs(curr_size - prev_size) / prev_size if prev_size > 0 else 0
            
            # Combined movement detection
            if center_movement > MOVEMENT_THRESHOLD_STRICT or size_change_ratio > 0.3:
                warnings.append("⚠️ Excessive movement detected")

        # Update previous box
        session_stats[session_id]["prev_box"] = [x1, y1, x2, y2]

    return warnings

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Video proctoring backend running.", "port": 5000})

# REST polling endpoint for 150ms frame analysis
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        frame_b64 = data.get("frame")
        
        if not frame_b64:
            return jsonify({"error": "No frame data provided"}), 400
            
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]
        
        frame_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400

        # Initialize session if new
        if session_id not in session_stats:
            session_stats[session_id] = {
                "count": 0, 
                "history": [], 
                "prev_box": None,
                "baseline_coverage": None
            }

        warnings = analyze_frame(frame, session_id)

        if warnings:
            session_stats[session_id]["count"] += len(warnings) * WARNINGS_PER_VIOLATION
            session_stats[session_id]["history"].extend(warnings)

        # Keep only last N warnings
        session_stats[session_id]["history"] = session_stats[session_id]["history"][-LAST_N_WARNINGS:]

        resp = {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
            "warning_count": session_stats[session_id]["count"],
            "last_warnings": session_stats[session_id]["history"][:],
            "timestamp": time.time()
        }
        return jsonify(resp)
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return jsonify({"error": "Frame analysis failed"}), 500

# WebSocket real-time events
@socketio.on("frame")
def on_frame(data):
    try:
        session_id = data.get("session_id", "default")
        frame_b64 = data.get("frame") or data.get("image")
        
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]
        
        frame_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if session_id not in session_stats:
            session_stats[session_id] = {
                "count": 0, 
                "history": [], 
                "prev_box": None,
                "baseline_coverage": None
            }

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
        
    except Exception as e:
        print(f"[ERROR] WebSocket frame processing failed: {e}")
        emit("error", {"message": "Frame processing failed"})

@socketio.on("connect")
def on_connect():
    print("[INFO] WebSocket client connected")
    emit("connected", {"message": "WebSocket connected successfully"})

@socketio.on("disconnect")
def on_disconnect():
    print("[INFO] WebSocket client disconnected")

if __name__ == "__main__":
    print("Starting Video Proctoring Backend Server...")
    print("REST API: http://localhost:5000/analyze")
    print("WebSocket: ws://localhost:5000")
    print("Optimized for 150ms frame polling")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)