"""
SilentSpeech: Model Router & Inference Dispatcher
Manages active model selection ('auto_avsr' vs 'grid'), lazy loading,
GPU memory caching, and unified prediction formatting.
"""

import os
import sys
import time
import base64
import traceback
import cv2
import numpy as np
import torch

# Ensure src/ is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import VisualSpeechModel
from dataset import CharTokenizer
from decoder import OpenVocabularyDecoder
from preprocess import dynamic_preprocess_video
from auto_avsr_inference import AutoAVSRModelWrapper, extract_visual_frames_from_roi
from visual_utils import extract_visual_pipeline_stages, frame_to_base64

# Global model cache to prevent reloading models on every request
_MODEL_CACHE = {}
DEFAULT_MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "auto_avsr").strip().lower()


def extract_grid_visual_frames(video_path, preprocessed_frames, bbox=None, first_bgr_frame=None):
    """Extract visual analysis frames for the GRID baseline pipeline."""
    return extract_visual_pipeline_stages(
        video_path,
        preprocessed_frames=preprocessed_frames,
        bbox=bbox
    )



class GridModelWrapper:
    """Wrapper for in-house 3D-CNN + ResNet-18 + BiGRU trained on GRID."""

    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = CharTokenizer()
        self.decoder = OpenVocabularyDecoder(tokenizer=self.tokenizer, beam_width=12, lm_weight=0.6)
        self.model = VisualSpeechModel(vocab_size=self.tokenizer.vocab_size)

        base_dir = os.path.dirname(current_dir)
        candidates = [
            os.path.join(base_dir, 'models', 'last_checkpoint.pth'),
            os.path.join(base_dir, 'models', 'best_wer.pth'),
            os.path.join(base_dir, 'models', 'best_cer.pth'),
            os.path.join(base_dir, 'models', 'best_model.pth'),
        ]
        checkpoint_path = next((c for c in candidates if os.path.exists(c)), None)

        if not checkpoint_path:
            raise FileNotFoundError("No trained GRID checkpoint found in models/ directory.")

        print(f"[ModelRouter] Loading GRID baseline from: {checkpoint_path} on {device}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    def predict(self, video_path):
        t_start = time.time()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames_raw / fps if fps > 0 else 0.0
        cap.release()

        frames_np, bbox, first_bgr_frame = dynamic_preprocess_video(video_path)
        num_frames = frames_np.shape[0]

        if num_frames == 0 or (num_frames == 1 and np.all(frames_np == 0)):
            return {
                'transcript': '',
                'model': 'grid',
                'model_name': 'GRID Baseline (51-word Vocabulary)',
                'time_seconds': round(time.time() - t_start, 2),
                'processing_time_ms': round((time.time() - t_start) * 1000),
                'num_frames': 0,
                'video_duration_s': round(duration, 2),
                'fps': round(fps, 1),
                'frames': {},
                'error': 'No face or lip movements could be detected in the video.'
            }

        frames_tensor = torch.tensor(frames_np, dtype=torch.float32).unsqueeze(0)
        frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4).to(self.device)

        with torch.no_grad():
            logits = self.model(frames_tensor)

        decoded_list = self.decoder.decode(logits, use_lexicon=True)
        raw_text = decoded_list[0] if decoded_list else ''
        transcript = raw_text.capitalize() if raw_text else ''

        visual_frames = extract_grid_visual_frames(video_path, frames_np, bbox, first_bgr_frame)
        elapsed = time.time() - t_start

        return {
            'transcript': transcript,
            'model': 'grid',
            'model_name': 'GRID Baseline (51-word Vocabulary)',
            'time_seconds': round(elapsed, 2),
            'processing_time_ms': round(elapsed * 1000),
            'num_frames': num_frames,
            'video_duration_s': round(duration, 2),
            'fps': round(fps, 1),
            'frames': visual_frames,
            'vocab_size': self.tokenizer.vocab_size,
            'error': None
        }


def get_model(model_name=None, device=None):
    """
    Lazy loader for models. Loads on demand and caches in memory.
    Falls back gracefully from CUDA to CPU if out-of-memory occurs.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_CHOICE

    model_name = model_name.strip().lower()
    if model_name not in ('auto_avsr', 'grid'):
        model_name = 'auto_avsr'

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cache_key = f"{model_name}_{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    print(f"[ModelRouter] Initializing '{model_name}' on device '{device}'...", flush=True)

    try:
        if model_name == 'auto_avsr':
            model_instance = AutoAVSRModelWrapper(device=device)
        else:
            model_instance = GridModelWrapper(device=device)

        _MODEL_CACHE[cache_key] = model_instance
        return model_instance

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        error_msg = str(e).lower()
        if 'out of memory' in error_msg and device == 'cuda':
            print("[ModelRouter] GPU Out of Memory! Freeing cache and falling back to CPU...", flush=True)
            torch.cuda.empty_cache()
            return get_model(model_name=model_name, device='cpu')
        raise


def predict_single(video_path, model_name='auto_avsr', device=None):
    """Run prediction on a single model."""
    model_name = (model_name or DEFAULT_MODEL_CHOICE).strip().lower()
    model = get_model(model_name=model_name, device=device)

    try:
        res = model.predict(video_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower():
            print("[ModelRouter] Prediction OOM. Clearing CUDA cache and retrying on CPU...", flush=True)
            torch.cuda.empty_cache()
            cpu_model = get_model(model_name=model_name, device='cpu')
            res = cpu_model.predict(video_path)
        else:
            raise

    # Ensure consistent keys
    if 'time_seconds' not in res:
        ms = res.get('processing_time_ms', 0)
        res['time_seconds'] = round(ms / 1000.0, 2)

    if 'model' not in res:
        res['model'] = model_name

    if 'model_name' not in res:
        res['model_name'] = 'Auto-AVSR (LRS3 Open Vocabulary)' if model_name == 'auto_avsr' else 'GRID Baseline (51-word Vocabulary)'

    # Clean up PyTorch GPU memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def predict(video_path, model_name=None, device=None):
    """
    Unified router endpoint.
    Supports:
      - 'auto_avsr': Uses Oxford Auto-AVSR open vocabulary model.
      - 'grid': Uses in-house GRID baseline model.
      - 'compare': Runs both models on the video and returns side-by-side transcripts.
    """
    choice = (model_name or DEFAULT_MODEL_CHOICE).strip().lower()

    if choice == 'compare':
        print(f"[ModelRouter] Running comparison on: {video_path}", flush=True)
        t_start = time.time()

        res_auto = predict_single(video_path, model_name='auto_avsr', device=device)
        res_grid = predict_single(video_path, model_name='grid', device=device)

        elapsed = time.time() - t_start
        primary_transcript = res_auto.get('transcript') or res_grid.get('transcript', '')

        return {
            'compare': True,
            'transcript': primary_transcript,
            'model': 'compare',
            'model_name': 'Model Comparison (Auto-AVSR vs GRID)',
            'time_seconds': round(elapsed, 2),
            'processing_time_ms': round(elapsed * 1000),
            'auto_avsr': {
                'transcript': res_auto.get('transcript', ''),
                'time_seconds': res_auto.get('time_seconds', 0),
                'model_name': 'Auto-AVSR (LRS3 Open Vocabulary)',
                'vocab': '5,000 SentencePiece Units (LRS3-TED)',
                'error': res_auto.get('error')
            },
            'grid': {
                'transcript': res_grid.get('transcript', ''),
                'time_seconds': res_grid.get('time_seconds', 0),
                'model_name': 'GRID Baseline (51-word Vocabulary)',
                'vocab': '51 Words (Fixed Grammar Formula)',
                'error': res_grid.get('error')
            },
            'num_frames': res_auto.get('num_frames') or res_grid.get('num_frames', 0),
            'video_duration_s': res_auto.get('video_duration_s') or res_grid.get('video_duration_s', 0),
            'fps': res_auto.get('fps') or res_grid.get('fps', 25.0),
            'frames': res_auto.get('frames') or res_grid.get('frames', {}),
            'error': res_auto.get('error') or res_grid.get('error')
        }

    return predict_single(video_path, model_name=choice, device=device)
