"""
SilentSpeech: Visual Preprocessing & ROI Frame Pipeline Utilities
Provides robust frame extraction, face/mouth landmark tracking annotation,
CLAHE dynamic range enhancement, and multi-stage visual artifacts for the frontend.
"""

import os
import cv2
import base64
import numpy as np


def enhance_display(img, is_gray=False):
    """
    Ensure the frame has rich contrast, vibrant dynamic range,
    and is clearly legible even on dark UI themes.
    """
    if img is None or img.size == 0:
        return img
    if img.dtype in (np.float32, np.float64):
        img = (img * 255).clip(0, 255).astype(np.uint8)

    if is_gray or len(img.shape) == 2:
        gray = img.squeeze() if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        res = clahe.apply(gray)
        if res.mean() < 80:
            p2, p98 = np.percentile(res, 2), np.percentile(res, 98)
            if p98 > p2:
                res = np.clip((res.astype(np.float32) - p2) / (p98 - p2) * 220 + 25, 0, 255).astype(np.uint8)
        return res
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_enh = clahe.apply(l)
        if l_enh.mean() < 80:
            p2, p98 = np.percentile(l_enh, 2), np.percentile(l_enh, 98)
            if p98 > p2:
                l_enh = np.clip((l_enh.astype(np.float32) - p2) / (p98 - p2) * 220 + 25, 0, 255).astype(np.uint8)
        lab = cv2.merge([l_enh, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def frame_to_base64(img, is_grayscale=False):
    """Encode a numpy frame array as a base64 JPEG string."""
    if is_grayscale and len(img.shape) == 2:
        arr = img
    elif is_grayscale and len(img.shape) == 3:
        arr = img.squeeze()
    else:
        arr = img
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode('utf-8')


def draw_hud_box(img, x1, y1, x2, y2, color=(241, 160, 99), label='LIP TRACKER'):
    """Draw a futuristic HUD bounding box with corner brackets and a status badge."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    length = max(int((x2 - x1) * 0.18), 8)
    t = 3
    # Top-left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, t)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, t)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, t)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, t)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 6, ty + 2), (15, 23, 42), -1)
        cv2.putText(img, label, (x1 + 3, ty), font, font_scale, color, thickness, cv2.LINE_AA)


def find_best_representative_frame(video_path, max_scan=35):
    """
    Scans the beginning of the video to avoid dark lead-ins / fade-from-black.
    Picks the sharpest frame with visible features (mean >= 20).
    Returns (best_index, best_bgr_frame).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while idx < max_scan:
        ret, f = cap.read()
        if not ret:
            break
        frames.append((idx, f))
        idx += 1
    cap.release()

    if not frames:
        return 0, None

    visible = [(i, f) for i, f in frames if f.mean() >= 20]
    if visible:
        def score(item):
            gray = cv2.cvtColor(item[1], cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        return max(visible, key=score)
    else:
        # All frames are dark, return brightest
        return max(frames, key=lambda item: item[1].mean())


def get_face_cascade():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(cur_dir), 'data', 'haarcascade_frontalface_default.xml'),
        os.path.join(cur_dir, '..', 'data', 'haarcascade_frontalface_default.xml'),
        os.path.abspath('data/haarcascade_frontalface_default.xml')
    ]
    for p in candidates:
        if os.path.exists(p):
            return cv2.CascadeClassifier(p)
    return None


def extract_visual_pipeline_stages(video_path, roi=None, preprocessed_frames=None, bbox=None):
    """
    Produce the three synchronized stages for the UI Visual Lip Tracking modal:
      - Stage 1: Face Landmark / Speech Tracking View ('annotated')
      - Stage 2: Stabilized Mouth Crop ('cropped', 96x96)
      - Stage 3: Normalized Neural Input ('processed', 88x88 CLAHE)
    """
    frames_data = {}
    best_idx, best_bgr = find_best_representative_frame(video_path)
    if best_bgr is None:
        return frames_data

    h, w = best_bgr.shape[:2]
    display_w = 320
    display_h = 320

    # Ensure good brightness for Stage 1 display
    base_display = enhance_display(best_bgr)
    resized_display = cv2.resize(base_display, (display_w, display_h), interpolation=cv2.INTER_CUBIC)
    frames_data['original'] = frame_to_base64(resized_display)

    annotated = resized_display.copy()
    mouth_crop_fallback = None

    if h <= 128 and w <= 128:
        # Pre-cropped mouth video (like LRS3 96x96 clips)
        mx1 = int(display_w * 0.16)
        mx2 = int(display_w * 0.84)
        my1 = int(display_h * 0.28)
        my2 = int(display_h * 0.74)
        draw_hud_box(annotated, mx1, my1, mx2, my2, color=(241, 160, 99), label='LIP TRACKER [96x96]')
        mouth_crop_fallback = cv2.resize(best_bgr, (96, 96))
    else:
        # Full-frame video
        scale_x = display_w / w
        scale_y = display_h / h

        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            mx1 = int(bx1 * scale_x)
            my1 = int(by1 * scale_y)
            mx2 = int(bx2 * scale_x)
            my2 = int(by2 * scale_y)
            draw_hud_box(annotated, mx1, my1, mx2, my2, color=(241, 160, 99), label='MOUTH ROI [96x96]')
            mouth_crop_fallback = best_bgr[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
        else:
            cascade = get_face_cascade()
            faces = []
            if cascade is not None:
                gray = cv2.cvtColor(best_bgr, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                # Subtle face contour
                cv2.rectangle(annotated, (int(fx * scale_x), int(fy * scale_y)),
                              (int((fx + fw) * scale_x), int((fy + fh) * scale_y)), (60, 80, 110), 1)
                mx1 = int((fx + fw * 0.20) * scale_x)
                mx2 = int((fx + fw * 0.80) * scale_x)
                my1 = int((fy + fh * 0.62) * scale_y)
                my2 = int((fy + fh * 0.92) * scale_y)
                draw_hud_box(annotated, mx1, my1, mx2, my2, color=(241, 160, 99), label='MOUTH ROI [96x96]')
                mouth_crop_fallback = best_bgr[int(fy + fh * 0.62):int(fy + fh * 0.92), int(fx + fw * 0.20):int(fx + fw * 0.80)]
            else:
                mx1 = int(display_w * 0.25)
                mx2 = int(display_w * 0.75)
                my1 = int(display_h * 0.52)
                my2 = int(display_h * 0.88)
                draw_hud_box(annotated, mx1, my1, mx2, my2, color=(241, 160, 99), label='SPEECH ROI')
                mouth_crop_fallback = best_bgr[int(h * 0.52):int(h * 0.88), int(w * 0.25):int(w * 0.75)]

    frames_data['annotated'] = frame_to_base64(annotated)

    # Stage 2: Stabilized Mouth Crop
    crop_display = None
    if roi is not None and len(roi) > 0:
        f_idx = min(best_idx, len(roi) - 1)
        r_frame = roi[f_idx]
        if len(r_frame.shape) == 3 and r_frame.shape[2] == 3:
            crop_bgr = cv2.cvtColor(r_frame, cv2.COLOR_RGB2BGR)
        else:
            crop_bgr = cv2.cvtColor(r_frame, cv2.COLOR_GRAY2BGR)
        crop_enh = enhance_display(crop_bgr)
        crop_display = cv2.resize(crop_enh, (160, 160), interpolation=cv2.INTER_CUBIC)
    elif mouth_crop_fallback is not None and mouth_crop_fallback.size > 0:
        crop_enh = enhance_display(mouth_crop_fallback)
        crop_display = cv2.resize(crop_enh, (160, 160), interpolation=cv2.INTER_CUBIC)
    else:
        crop_display = cv2.resize(base_display, (160, 160))

    frames_data['cropped'] = frame_to_base64(crop_display)

    # Stage 3: Normalized 88x88 Input
    if preprocessed_frames is not None and len(preprocessed_frames) > 0:
        f_idx = min(best_idx, len(preprocessed_frames) - 1)
        proc_raw = preprocessed_frames[f_idx].squeeze()
        c_h, c_w = proc_raw.shape[:2]
        sy = max(0, (c_h - 88) // 2)
        sx = max(0, (c_w - 88) // 2)
        proc_88 = proc_raw[sy:sy + 88, sx:sx + 88]
        proc_enh = enhance_display(proc_88, is_gray=True)
        proc_display = cv2.resize(proc_enh, (160, 160), interpolation=cv2.INTER_NEAREST)
    else:
        gray_crop = cv2.cvtColor(crop_display, cv2.COLOR_BGR2GRAY)
        c_h, c_w = gray_crop.shape[:2]
        sy = max(0, (c_h - 88) // 2)
        sx = max(0, (c_w - 88) // 2)
        proc_88 = gray_crop[sy:sy + 88, sx:sx + 88]
        proc_enh = enhance_display(proc_88, is_gray=True)
        proc_display = cv2.resize(proc_enh, (160, 160), interpolation=cv2.INTER_NEAREST)

    frames_data['processed'] = frame_to_base64(proc_display, is_grayscale=True)
    return frames_data
