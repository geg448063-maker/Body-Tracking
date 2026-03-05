import math
import ctypes
import sys
from importlib import import_module
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# we will use elbow angle for pushup detection
def calculate_angle(shoulder, elbow, wrist):
    """Calculate angle between upper arm and forearm at the elbow."""
    # vectors from elbow to shoulder and elbow to wrist
    v1 = (shoulder.x - elbow.x, shoulder.y - elbow.y)
    v2 = (wrist.x - elbow.x, wrist.y - elbow.y)
    
    # dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    # cosine of angle
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))  # clamp to [-1, 1]
    
    # angle in degrees
    angle = math.degrees(math.acos(cos_angle))
    return angle

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

# states for pushup detection: 0=up (extended), 1=down (bent)
state = 0
count = 0

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
        landmarks = results.pose_landmarks[0]
        
        # calculate right arm elbow angle (shoulder 12, elbow 14, wrist 16)
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # state: 0 = up (extended, angle < 90), 1 = down (bent, angle >= 90)
        if state == 0 and angle >= 90:
            state = 1
        elif state == 1 and angle < 90:
            state = 0
            count += 1

        draw_pose(frame, landmarks, h, w)
        annotate(frame, f"Pushups: {count} | Angle: {angle:.1f}°")

    cv2.imshow("Pushup Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
