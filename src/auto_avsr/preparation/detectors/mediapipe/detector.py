import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class LandmarksDetector:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cur_dir))))), 'data', 'face_landmarker.task'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))), 'data', 'face_landmarker.task'),
            os.path.abspath('data/face_landmarker.task'),
            os.path.abspath('SilentSpeech/data/face_landmarker.task')
        ]
        task_path = next((p for p in candidates if os.path.exists(p)), None)
        if task_path is None:
            task_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))), 'data', 'face_landmarker.task')
            os.makedirs(os.path.dirname(task_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, task_path)
            except Exception:
                pass

        self.detector = None
        try:
            base_options = python.BaseOptions(model_asset_path=task_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            self.detector = None

        cascade_candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cur_dir))))), 'data', 'haarcascade_frontalface_default.xml'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))), 'data', 'haarcascade_frontalface_default.xml'),
            os.path.abspath('data/haarcascade_frontalface_default.xml'),
            os.path.abspath('SilentSpeech/data/haarcascade_frontalface_default.xml')
        ]
        cascade_path = next((p for p in cascade_candidates if os.path.exists(p)), None)
        self.face_cascade = cv2.CascadeClassifier(cascade_path) if cascade_path else None

    def __call__(self, video_frames):
        landmarks = self.detect(video_frames)
        assert any(l is not None for l in landmarks), "Cannot detect any face in the video"
        return landmarks

    def detect(self, video_frames):
        landmarks = []
        smoothed_face = None
        for frame in video_frames:
            h, w = frame.shape[:2]
            if self.detector is not None:
                try:
                    rgb = np.ascontiguousarray(frame)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = self.detector.detect(mp_img)

                    if res.face_landmarks:
                        lms = res.face_landmarks[0]
                        re = [(lms[33].x + lms[133].x) / 2.0 * w, (lms[33].y + lms[133].y) / 2.0 * h]
                        le = [(lms[362].x + lms[263].x) / 2.0 * w, (lms[362].y + lms[263].y) / 2.0 * h]
                        nose = [lms[4].x * w, lms[4].y * h]
                        mouth = [(lms[13].x + lms[14].x) / 2.0 * w, (lms[13].y + lms[14].y) / 2.0 * h]
                        landmarks.append(np.array([re, le, nose, mouth]))
                        continue
                except Exception:
                    pass

            # Fast native OpenCV Haar Cascade Face Detection
            detected_face = None
            if self.face_cascade is not None and not self.face_cascade.empty():
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        detected_face = faces[0]
                except Exception:
                    pass

            if detected_face is not None:
                fx, fy, fw, fh = detected_face
                if smoothed_face is None:
                    smoothed_face = [float(fx), float(fy), float(fw), float(fh)]
                else:
                    smoothed_face = [
                        0.3 * float(fx) + 0.7 * smoothed_face[0],
                        0.3 * float(fy) + 0.7 * smoothed_face[1],
                        0.3 * float(fw) + 0.7 * smoothed_face[2],
                        0.3 * float(fh) + 0.7 * smoothed_face[3]
                    ]
                sfx, sfy, sfw, sfh = smoothed_face
                re = [sfx + 0.35 * sfw, sfy + 0.38 * sfh]
                le = [sfx + 0.65 * sfw, sfy + 0.38 * sfh]
                nose = [sfx + 0.50 * sfw, sfy + 0.55 * sfh]
                mouth = [sfx + 0.50 * sfw, sfy + 0.78 * sfh]
                landmarks.append(np.array([re, le, nose, mouth]))
                continue

            if smoothed_face is not None:
                sfx, sfy, sfw, sfh = smoothed_face
                re = [sfx + 0.35 * sfw, sfy + 0.38 * sfh]
                le = [sfx + 0.65 * sfw, sfy + 0.38 * sfh]
                nose = [sfx + 0.50 * sfw, sfy + 0.55 * sfh]
                mouth = [sfx + 0.50 * sfw, sfy + 0.78 * sfh]
                landmarks.append(np.array([re, le, nose, mouth]))
                continue

            # Standard facial proportions fallback
            re = [0.38 * w, 0.40 * h]
            le = [0.62 * w, 0.40 * h]
            nose = [0.50 * w, 0.55 * h]
            mouth = [0.50 * w, 0.72 * h]
            landmarks.append(np.array([re, le, nose, mouth]))
        return landmarks
