import ctypes
import sys
from importlib import import_module

if sys.platform == "win32":
    _b = import_module("mediapipe.tasks.python.core.mediapipe_c_bindings")

    def _patched(signatures=()):
        import os
        import platform
        from importlib import resources
        if _b._shared_lib is None:
            fn = "libmediapipe.dll" if os.name != "posix" else ("libmediapipe.dylib" if platform.system() == "Darwin" else "libmediapipe.so")
            _b._shared_lib = ctypes.CDLL(str(resources.files("mediapipe.tasks.c") / fn))
        for s in signatures:
            c = getattr(_b._shared_lib, s.func_name)
            c.argtypes, c.restype = s.argtypes, s.restype
        try:
            _b._shared_lib.free.argtypes = [ctypes.c_void_p]
            _b._shared_lib.free.restype = None
        except AttributeError:
            for crt in ("ucrtbase", "msvcrt", "api-ms-win-crt-heap-l1-1-0"):
                try:
                    _b._shared_lib.free = ctypes.CDLL(crt).free
                    break
                except OSError:
                    continue
            _b._shared_lib.free.argtypes = [ctypes.c_void_p]
            _b._shared_lib.free.restype = None
        return _b._shared_lib

    _b.load_raw_library = _patched

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

model_path = Path(__file__).parent / "face_landmarker.task"
if not model_path.exists():
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path,
    )

base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
frame_count = 0

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

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]
        nose_tip = landmarks[1]
        x = int(nose_tip.x * w)
        y = int(nose_tip.y * h)
        cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)

    cv2.imshow("Face Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
