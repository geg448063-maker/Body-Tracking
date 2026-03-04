import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# download model if not present
model_path = Path(__file__).parent / "pose_landmarker_lite.task"
if not model_path.exists():
    import urllib.request

    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        model_path,
    )
    print("Download complete!")

# options similar to face tracker but using PoseLandmarker
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

frame_count = 0

# states for pushup detection: 0=up,1=down
state = 0
count = 0

# we will use nose y-coordinate for simplicity
def get_nose_y(landmarks):
    # landmark index 0 = nose
    return landmarks[0].y if landmarks else None

# initial baseline
baseline_y = None

def annotate(frame, text):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_pose(frame, landmarks, h, w):
    """Draw pose skeleton on frame."""
    # connection pairs for the body
    connections = [
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # shoulders to hips
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]
    
    # draw circles at each landmark
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # draw lines connecting joints
    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_pos = (int(start.x * w), int(start.y * h))
        end_pos = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_pos, end_pos, (255, 0, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp = int(frame_count * 1000 / 30)

    results = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_count += 1

    if results.pose_landmarks:
        y = get_nose_y(results.pose_landmarks[0])
        if baseline_y is None:
            baseline_y = y
        # threshold: 0.10 of height
        thresh = baseline_y + 0.10
        if state == 0 and y > thresh:
            state = 1
        elif state == 1 and y <= thresh:
            state = 0
            count += 1

        draw_pose(frame, results.pose_landmarks[0], h, w)
        annotate(frame, f"Pushups: {count}")

    cv2.imshow("Pushup Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
