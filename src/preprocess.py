import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request

def load_and_preprocess_video(video_path, target_size=(96, 96)):
    """
    Standard unified video preprocessing for SilentSpeech:
    - Reads frames with cv2.VideoCapture
    - Converts each frame to grayscale
    - Crops fixed mouth region: y from 50% to 90%, x from 30% to 70%
    - Resizes to target_size (default 96x96)
    - Normalizes to [0.0, 1.0] as float32
    - Returns numpy array of shape (Frames, 1, 96, 96)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop_y1 = int(h * 0.5)
        crop_y2 = int(h * 0.9)
        crop_x1 = int(w * 0.3)
        crop_x2 = int(w * 0.7)
        
        crop = gray[crop_y1:crop_y2, crop_x1:crop_x2]
        resized = cv2.resize(crop, target_size)
        norm = resized.astype(np.float32) / 255.0
        frames.append(np.expand_dims(norm, axis=0)) # (1, H, W)
        
    cap.release()
    
    if len(frames) > 0:
        return np.stack(frames) # (Frames, 1, H, W)
    else:
        return np.zeros((1, 1, target_size[0], target_size[1]), dtype=np.float32)

def dynamic_preprocess_video(video_path, target_size=(96, 96)):
    """
    Robust dynamic video preprocessing for custom user videos (webcam, phone, or downloaded):
    - Uses MediaPipe FaceLandmarker for continuous lip tracking
    - Applies Exponential Moving Average (EMA) smoothing to eliminate camera/lip jitter
    - Handles head motion and tilt across all video frames
    - Applies CLAHE adaptive contrast normalization for varying lighting
    - Resamples temporally to 25 FPS (matching model training distribution)
    """
    model_path = 'data/face_landmarker.task'
    if not os.path.exists(model_path):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(os.path.dirname(cur_dir), 'data', 'face_landmarker.task')
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            import urllib.request
            os.makedirs('data', exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

    detector = None
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        detector = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"[PREPROCESS] MediaPipe FaceLandmarker unavailable ({e}). Using robust geometric mouth tracking fallback.", flush=True)
        detector = None

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps is None or video_fps <= 0 or np.isnan(video_fps):
        video_fps = 25.0

    frames = []
    smoothed_bbox = None
    first_bgr_frame = None
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        if first_bgr_frame is None:
            first_bgr_frame = frame.copy()

        # Run MediaPipe face mesh if available
        if detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)

                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    min_x, min_y, max_x, max_y = get_lip_bounding_box(landmarks, w, h)

                    lip_w = max_x - min_x
                    lip_cx = (min_x + max_x) // 2
                    lip_cy = (min_y + max_y) // 2

                    box_w = max(int(lip_w * 3.79), 60)
                    box_h = int(box_w / 1.252)

                    crop_y1 = int(lip_cy - box_h * 0.682)
                    crop_y2 = crop_y1 + box_h
                    crop_x1 = int(lip_cx - box_w // 2)
                    crop_x2 = crop_x1 + box_w

                    crop_y1 = max(0, min(h - 10, crop_y1))
                    crop_y2 = max(crop_y1 + 10, min(h, crop_y2))
                    crop_x1 = max(0, min(w - 10, crop_x1))
                    crop_x2 = max(crop_x1 + 10, min(w, crop_x2))

                    current_bbox = [crop_x1, crop_y1, crop_x2, crop_y2]

                    # EMA smoothing (alpha=0.35)
                    if smoothed_bbox is None:
                        smoothed_bbox = current_bbox
                    else:
                        smoothed_bbox = [
                            int(0.35 * cur + 0.65 * sm)
                            for cur, sm in zip(current_bbox, smoothed_bbox)
                        ]
            except Exception:
                pass

        # Crop using smoothed bounding box (or fallback center if face not detected yet)
        if smoothed_bbox is not None:
            bx1, by1, bx2, by2 = smoothed_bbox
            lip_crop = frame[by1:by2, bx1:bx2]
        else:
            # Fallback to lower face center
            cy1, cy2 = int(h * 0.5), int(h * 0.9)
            cx1, cx2 = int(w * 0.3), int(w * 0.7)
            lip_crop = frame[cy1:cy2, cx1:cx2]

        if lip_crop.size > 0:
            gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
            # Enhance local lip contrast for uneven lighting
            enhanced = clahe.apply(gray)
            resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
            norm = resized.astype(np.float32) / 255.0
            frames.append(np.expand_dims(norm, axis=0))

    cap.release()

    if len(frames) > 0:
        # Standardize temporal rate to 25 FPS
        target_fps = 25.0
        if abs(video_fps - target_fps) > 0.5:
            duration = len(frames) / video_fps
            target_frame_count = max(int(duration * target_fps), 1)
            indices = np.linspace(0, len(frames) - 1, target_frame_count).astype(int)
            frames = [frames[i] for i in indices]

        frames_np = np.stack(frames) # (Frames, 1, H, W)
        return frames_np, tuple(smoothed_bbox) if smoothed_bbox else None, first_bgr_frame
    else:
        return np.zeros((1, 1, target_size[0], target_size[1]), dtype=np.float32), None, None


def get_lip_bounding_box(landmarks, width, height):
    # MediaPipe Face Mesh lip indices (outer and inner)
    lip_indices = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        191, 80, 81, 82, 13, 312, 311, 310, 415
    ]
    
    x_coords = [landmarks[i].x for i in lip_indices]
    y_coords = [landmarks[i].y for i in lip_indices]
    
    min_x = int(min(x_coords) * width)
    max_x = int(max(x_coords) * width)
    min_y = int(min(y_coords) * height)
    max_y = int(max(y_coords) * height)
    
    return min_x, min_y, max_x, max_y

def process_video(video_path, output_dir, artifact_img_path):
    model_path = 'data/face_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading Face Landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    
    detector = vision.FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    os.makedirs(output_dir, exist_ok=True)
    
    first_frame = None
    first_crop = None
    bx1, by1, bx2, by2 = 0, 0, 0, 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            min_x, min_y, max_x, max_y = get_lip_bounding_box(landmarks, w, h)
            
            # Add padding
            pad_x = int((max_x - min_x) * 0.2)
            pad_y = int((max_y - min_y) * 0.2)
            
            min_x = max(0, min_x - pad_x)
            max_x = min(w, max_x + pad_x)
            min_y = max(0, min_y - pad_y)
            max_y = min(h, max_y + pad_y)
            
            # Crop and process
            lip_crop = frame[min_y:max_y, min_x:max_x]
            if lip_crop.size == 0:
                continue
                
            lip_gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
            lip_resized = cv2.resize(lip_gray, (96, 96))
            
            if frame_idx == 0:
                first_frame = frame.copy()
                first_crop = lip_resized.copy()
                bx1, by1, bx2, by2 = min_x, min_y, max_x, max_y
                
            # Save frame
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), lip_resized)
            
        frame_idx += 1
        
    cap.release()
    
    # Save the comparison visualization for the artifact
    if first_frame is not None and first_crop is not None:
        h, w, _ = first_frame.shape
        # Add labels
        cv2.putText(first_frame, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(first_frame, (bx1, by1), (bx2, by2), (0,0,255), 2)
        
        crop_bgr = cv2.cvtColor(first_crop, cv2.COLOR_GRAY2BGR)
        crop_resized = cv2.resize(crop_bgr, (h, h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop_resized, 'Cropped 96x96', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        combined = cv2.hconcat([first_frame, crop_resized])
        
        os.makedirs(os.path.dirname(artifact_img_path), exist_ok=True)
        cv2.imwrite(artifact_img_path, combined)
        print(f"Visualization saved to {artifact_img_path}")
        print(f"Processed {frame_idx} frames.")
    else:
        print("No faces found in the video.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='data/sample.mp4')
    parser.add_argument('--output', type=str, default='data/processed_lips')
    parser.add_argument('--artifact-img', type=str, default='C:/Users/anshi/.gemini/antigravity/brain/b744a2bc-4e6e-4cf1-b70f-983ebf732aea/comparison.png')
    args = parser.parse_args()
    process_video(args.video, args.output, args.artifact_img)
