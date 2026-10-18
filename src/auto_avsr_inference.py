"""
SilentSpeech - Auto-AVSR Open-Vocabulary Lip-Reading Inference Interface
Wraps the pre-trained LRS3 Auto-AVSR model for end-to-end open-vocabulary transcription.
"""

import os
import sys
import time
import base64
import cv2
import numpy as np
import torch

# Ensure auto_avsr directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
auto_avsr_dir = os.path.join(current_dir, 'auto_avsr')
if auto_avsr_dir not in sys.path:
    sys.path.insert(0, auto_avsr_dir)

from pipeline import LipReader, normalize_video
from visual_utils import extract_visual_pipeline_stages, frame_to_base64


def extract_visual_frames_from_roi(video_path, roi):
    """
    Extract high-contrast, landmark-annotated visualization frames for the frontend:
    - Stage 1: Face Landmark / Speech Tracking View ('annotated')
    - Stage 2: Stabilized Mouth Crop ('cropped', 96x96)
    - Stage 3: Normalized Neural Input ('processed', 88x88 CLAHE)
    """
    return extract_visual_pipeline_stages(video_path, roi=roi)



class AutoAVSRModelWrapper:
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"[AutoAVSR] Initializing LipReader on device: {self.device}...", flush=True)
        self.reader = LipReader(device=self.device, detector="mediapipe", enhance=True)
        print("[AutoAVSR] Auto-AVSR model initialized successfully.", flush=True)

    def predict(self, video_path, nbest=5, hflip=False):
        """
        Run open-vocabulary lip reading inference on the given video.
        Returns a dictionary with transcript, segments, stats, and visual breakdown frames.
        """
        t_start = time.time()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        # Run transcription with LipReader
        # LipReader internally normalizes to 25 FPS and extracts mouth ROIs
        norm_path = normalize_video(video_path, hflip=hflip, fps=25)
        
        try:
            try:
                roi = self.reader.vsr.crop_sequence(norm_path)
            except Exception as e:
                return {
                    'transcript': '',
                    'error': f'Could not detect face in video: {str(e)}',
                    'num_frames': total_frames,
                    'processing_time_ms': round((time.time() - t_start) * 1000),
                    'video_duration_s': round(duration, 2),
                    'fps': round(fps, 1),
                    'frames': {}
                }

            from pipeline import lip_motion, segment_motion
            tensor = self.reader.vsr.roi_to_tensor(roi)
            # For clips up to 12 seconds, decode whole sequence continuously to preserve full sentence context
            if tensor.shape[0] <= int(12.0 * self.reader.fps):
                segs = [(0, tensor.shape[0])]
            else:
                motion = lip_motion(roi)
                segs = segment_motion(motion, fps=self.reader.fps)
            results = []
            for (a, b) in segs:
                sub = tensor[a:b]
                if sub.shape[0] < 4:
                    continue
                hyps = self.reader.vsr.transcribe_tensor(sub, nbest=nbest)
                results.append({"seg_frames": [a, b],
                                "seg_time": [a / self.reader.fps, b / self.reader.fps],
                                "text": hyps[0]["text"].lower().strip(),
                                "nbest": hyps})
            transcript = " ".join(r["text"] for r in results).strip()
            
            # If transcript is all uppercase or lowercase, format nicely
            if transcript:
                transcript = transcript.capitalize()
                
            visual_frames = extract_visual_frames_from_roi(video_path, roi)
            t_end = time.time()
            
            return {
                'transcript': transcript,
                'segments': results,
                'num_frames': len(roi) if roi is not None else total_frames,
                'processing_time_ms': round((t_end - t_start) * 1000),
                'video_duration_s': round(duration, 2),
                'fps': round(fps, 1),
                'frames': visual_frames,
                'model_name': 'Auto-AVSR (LRS3 Open-Vocabulary)',
                'vocab_size': 5000
            }
        finally:
            try:
                if os.path.exists(norm_path) and norm_path != video_path:
                    os.remove(norm_path)
            except Exception:
                pass
