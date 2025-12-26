"""
test_improved.py

FINAL SCRIPT for Real-Time Emotion Recognition.
Updated for the High-Accuracy Model (160x160 RGB).
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base_dir, 'facial_expression_model.keras')

# UPDATED: Match the new training size
IMG_SIZE = (160, 160)

# Labels
EMOTION_LABELS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_LABELS)

# Detection settings
MIN_DETECTION_CONFIDENCE = 0.5 

# Smoothing parameters
PREDICTION_QUEUE_LENGTH = 5 
EMA_ALPHA = 0.6 
MIN_CONFIDENCE_THRESHOLD = 0.40 # Increased slightly since model is more confident now

# Colors & Fonts
BOX_COLOR_DEFAULT = (200, 200, 200) 
TEXT_COLOR = (255, 255, 255) 
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
RECT_THICKNESS = 2

EMOTION_COLORS = {
    'angry': (0, 0, 255),       # Red
    'contempt': (255, 140, 0),  # Orange
    'disgust': (0, 100, 0),     # Dark Green
    'fear': (128, 0, 128),      # Purple
    'happy': (0, 255, 0),       # Green
    'neutral': (169, 169, 169), # Gray
    'sad': (255, 0, 0),         # Blue
    'surprise': (0, 255, 255)   # Yellow
}

# --- Helper Functions ---
def preprocess_face(face_roi, target_size):
    """
    Preprocesses the face for the NEW RGB Model (160x160).
    """
    if face_roi.size == 0:
        return None
    
    # 1. Resize directly (Keep color information)
    face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
    
    # 2. Convert BGR (OpenCV standard) to RGB (Model standard)
    # We DO NOT convert to grayscale anymore!
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # 3. MobileNetV2 Preprocessing (Values -1 to 1)
    face_array = face_rgb.astype("float32")
    face_preprocessed = preprocess_input(face_array) 
    
    # Expand dimensions for batch
    face_batch = np.expand_dims(face_preprocessed, axis=0)
    return face_batch

# --- Main Execution ---
def main():
    # Disable GPU logs
    try:
        tf.debugging.set_log_device_placement(False)
    except:
        pass

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize MediaPipe
    print("Initializing face detection...")
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE)

    # Video Capture
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Tracking Variables
    face_predictions = {}
    last_seen = {}
    face_id_counter = 0
    MAX_FACES_TO_TRACK = 5 
    FRAME_TIMEOUT = 10 

    frame_count = 0
    start_time = time.time()

    print("Starting detection... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get dimensions
        frame_height, frame_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Face Detection
        results = face_detection.process(frame_rgb)

        current_frame_face_centers = []
        detected_faces_data = []

        if results.detections:
            for detection in results.detections:
                try:
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * frame_width)
                    y_min = int(bboxC.ymin * frame_height)
                    width = int(bboxC.width * frame_width)
                    height = int(bboxC.height * frame_height)

                    # Add padding (Catch hair/chin for context)
                    padding_w = int(width * 0.15) # Increased padding slightly
                    padding_h = int(height * 0.15)
                    
                    x = max(0, x_min - padding_w)
                    y = max(0, y_min - padding_h)
                    w = min(frame_width - x, width + 2 * padding_w)
                    h = min(frame_height - y, height + 2 * padding_h)

                    if w > 0 and h > 0:
                        face_roi = frame[y:y+h, x:x+w]
                        center_x = x + w // 2
                        center_y = y + h // 2
                        current_frame_face_centers.append((center_x, center_y))
                        detected_faces_data.append(((x, y, w, h), face_roi))
                except:
                    continue 

        # 2. Tracking & Prediction
        processed_indices = set()

        # A. Update existing faces
        for face_id, data in list(face_predictions.items()):
            min_dist = float('inf')
            best_match_idx = -1

            for idx, (center_x, center_y) in enumerate(current_frame_face_centers):
                if idx in processed_indices: continue
                dist = np.sqrt((center_x - data['center'][0])**2 + (center_y - data['center'][1])**2)
                if dist < min_dist and dist < 100:
                    min_dist = dist
                    best_match_idx = idx

            if best_match_idx != -1:
                # Match found
                bbox, face_roi = detected_faces_data[best_match_idx]
                center = current_frame_face_centers[best_match_idx]

                # Preprocess & Predict
                face_preprocessed = preprocess_face(face_roi, IMG_SIZE)
                if face_preprocessed is not None:
                    preds = model.predict(face_preprocessed, verbose=0)[0]

                    # Smooth predictions
                    data['queue'].append(preds)
                    data['smoothed'] = EMA_ALPHA * preds + (1 - EMA_ALPHA) * data['smoothed']
                    
                    label_idx = np.argmax(data['smoothed'])
                    label = EMOTION_LABELS[label_idx]
                    confidence = data['smoothed'][label_idx]

                    data['bbox'] = bbox
                    data['center'] = center
                    last_seen[face_id] = frame_count
                    processed_indices.add(best_match_idx)

                    # Draw
                    x, y, w, h = bbox
                    if confidence >= MIN_CONFIDENCE_THRESHOLD:
                        color = EMOTION_COLORS.get(label, BOX_COLOR_DEFAULT)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, RECT_THICKNESS)
                        
                        # Text Label
                        label_text = f"{label} {confidence:.0%}"
                        (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
                        cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
                        cv2.putText(frame, label_text, (x, y - 5), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLOR_DEFAULT, RECT_THICKNESS)
            else:
                # Face missing
                if frame_count - last_seen[face_id] > FRAME_TIMEOUT:
                    del face_predictions[face_id]
                    if face_id in last_seen: del last_seen[face_id]

        # B. Register new faces
        for idx, ((bbox, face_roi), center) in enumerate(zip(detected_faces_data, current_frame_face_centers)):
            if idx not in processed_indices and len(face_predictions) < MAX_FACES_TO_TRACK:
                face_preprocessed = preprocess_face(face_roi, IMG_SIZE)
                if face_preprocessed is not None:
                    preds = model.predict(face_preprocessed, verbose=0)[0]
                    
                    new_id = face_id_counter
                    face_id_counter += 1

                    face_predictions[new_id] = {
                        'queue': deque([preds] * 3, maxlen=PREDICTION_QUEUE_LENGTH),
                        'smoothed': preds,
                        'bbox': bbox,
                        'center': center
                    }
                    last_seen[new_id] = frame_count
                    
                    # Draw immediately
                    x, y, w, h = bbox
                    label_idx = np.argmax(preds)
                    label = EMOTION_LABELS[label_idx]
                    color = EMOTION_COLORS.get(label, BOX_COLOR_DEFAULT)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, RECT_THICKNESS)
                    cv2.putText(frame, label, (x, y - 10), FONT, FONT_SCALE, color, FONT_THICKNESS)

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1:
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.6, (0, 255, 0), 2)
            start_time = time.time()
            frame_count = 0

        cv2.imshow("Advanced Emotion Recognition (RGB 160x160)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()