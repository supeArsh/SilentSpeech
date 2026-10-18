"""
SilentSpeech - Multi-Model Ensemble & Hybrid Consensus Engine
Combines our in-house 3D-CNN + ResNet-18 + BiGRU Neural Network with
the Auto-AVSR Large Conformer / Transformer model to synthesize optimal visual speech recognition.
"""

import os
import sys
import time
import math
import collections
import cv2
import numpy as np
import torch
from rapidfuzz import fuzz, distance

# Path setup
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model import VisualSpeechModel
from dataset import CharTokenizer
from decoder import OpenVocabularyDecoder, COMMON_WORDS
from preprocess import dynamic_preprocess_video
from auto_avsr_inference import AutoAVSRModelWrapper, extract_visual_frames_from_roi, frame_to_base64


class MultiModelEnsemble:
    """
    Ensemble engine uniting In-House Neural Engine and Auto-AVSR Conformer.
    Provides consensus fusion, alignment, and confidence scoring.
    """
    def __init__(self, inhouse_checkpoint=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Ensemble] Initializing Multi-Model Ensemble on device: {self.device}", flush=True)

        # 1. Initialize In-House Model
        self.tokenizer = CharTokenizer()
        self.decoder = OpenVocabularyDecoder(tokenizer=self.tokenizer, beam_width=12, lm_weight=0.6)
        self.inhouse_model = VisualSpeechModel(vocab_size=self.tokenizer.vocab_size)
        self.inhouse_checkpoint_name = "none"

        base_dir = os.path.dirname(src_dir)
        if inhouse_checkpoint is None:
            candidates = [
                os.path.join(base_dir, 'models', 'last_checkpoint.pth'),
                os.path.join(base_dir, 'models', 'best_wer.pth'),
                os.path.join(base_dir, 'models', 'best_cer.pth'),
                os.path.join(base_dir, 'models', 'best_model.pth'),
            ]
            inhouse_checkpoint = next((c for c in candidates if os.path.exists(c)), None)

        if inhouse_checkpoint and os.path.exists(inhouse_checkpoint):
            self.inhouse_checkpoint_name = os.path.basename(inhouse_checkpoint)
            ckpt = torch.load(inhouse_checkpoint, map_location=self.device, weights_only=False)
            state = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
            self.inhouse_model.load_state_dict(state)
            print(f"[Ensemble] In-House Model loaded with checkpoint: {self.inhouse_checkpoint_name}", flush=True)
        else:
            print("[Ensemble] Warning: In-House model checkpoint not found!", flush=True)

        self.inhouse_model.to(self.device)
        self.inhouse_model.eval()

        # 2. Initialize Auto-AVSR Large Conformer
        self.auto_avsr = None
        try:
            self.auto_avsr = AutoAVSRModelWrapper(device=self.device)
            print("[Ensemble] Auto-AVSR Conformer loaded successfully.", flush=True)
        except Exception as e:
            print(f"[Ensemble] Warning: Auto-AVSR failed to load on {self.device}: {e}", flush=True)
            if self.device == 'cuda':
                try:
                    print("[Ensemble] Retrying Auto-AVSR on CPU...", flush=True)
                    self.auto_avsr = AutoAVSRModelWrapper(device='cpu')
                except Exception as e2:
                    print(f"[Ensemble] Auto-AVSR CPU load also failed: {e2}", flush=True)

    def predict_inhouse(self, video_path):
        """Execute In-House model forward pass with MediaPipe EMA lip tracking."""
        t_start = time.time()
        frames_np, bbox, first_bgr_frame = dynamic_preprocess_video(video_path)
        num_frames = frames_np.shape[0]

        if num_frames == 0 or (num_frames == 1 and np.all(frames_np == 0)):
            return {
                'transcript': '',
                'raw_text': '',
                'num_frames': 0,
                'latency_ms': round((time.time() - t_start) * 1000),
                'error': 'No face/lips detected in video',
                'frames_np': None,
                'bbox': None,
                'first_bgr_frame': None
            }

        # Convert to tensor: (B, C, T, H, W)
        frames_tensor = torch.tensor(frames_np, dtype=torch.float32).unsqueeze(0)
        frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4).to(self.device)

        with torch.no_grad():
            logits = self.inhouse_model(frames_tensor) # (1, T, 29)

        decoded_list = self.decoder.decode(logits, use_lexicon=True)
        transcript = decoded_list[0].strip() if decoded_list else ''

        return {
            'transcript': transcript,
            'num_frames': num_frames,
            'latency_ms': round((time.time() - t_start) * 1000),
            'frames_np': frames_np,
            'bbox': bbox,
            'first_bgr_frame': first_bgr_frame
        }

    def predict_auto_avsr(self, video_path):
        """Execute Auto-AVSR Conformer forward pass."""
        if self.auto_avsr is None:
            return {
                'transcript': '',
                'latency_ms': 0,
                'error': 'Auto-AVSR model is not available'
            }
        t_start = time.time()
        try:
            res = self.auto_avsr.predict(video_path)
            res['latency_ms'] = round((time.time() - t_start) * 1000)
            return res
        except Exception as e:
            return {
                'transcript': '',
                'latency_ms': round((time.time() - t_start) * 1000),
                'error': str(e)
            }

    def fuse_transcripts(self, inhouse_text, avsr_text, duration_s=1.0):
        """
        Synthesizes the optimal combined transcription using multi-model consensus,
        token-level alignment, and language model validation.
        """
        t_in = (inhouse_text or '').strip().lower()
        t_av = (avsr_text or '').strip().lower()

        # If one is empty, return the other
        if not t_in and not t_av:
            return "", 0.0, 0.0, "No speech detected"
        if not t_in:
            return t_av.capitalize(), 85.0, 50.0, "Auto-AVSR standalone prediction"
        if not t_av:
            return t_in.capitalize(), 85.0, 50.0, "In-House model standalone prediction"

        # Check exact or near-identical agreement
        similarity = fuzz.ratio(t_in, t_av)
        token_sort_sim = fuzz.token_sort_ratio(t_in, t_av)

        words_in = t_in.split()
        words_av = t_av.split()

        # High agreement: models agree strongly
        if similarity >= 85:
            # Pick the one with better punctuation/casing or longer valid tokens
            consensus = t_av if len(words_av) >= len(words_in) else t_in
            confidence = min(98.5, 90.0 + (similarity - 85) * 0.5)
            return consensus.capitalize(), round(confidence, 1), round(similarity, 1), "High dual-model consensus"

        # Score vocabulary & grammatical credibility of both candidate strings
        def score_candidate(words):
            if not words:
                return 0.0
            valid_words = sum(1 for w in words if w in COMMON_WORDS or len(w) > 3)
            ratio = valid_words / len(words)
            # Length penalty if words count is unnatural for duration
            expected_words = max(1, duration_s * 2.5) # approx 2.5 words/sec in conversational English
            len_diff = abs(len(words) - expected_words)
            len_score = max(0.2, 1.0 - (len_diff / 10.0))
            return (ratio * 0.7) + (len_score * 0.3)

        score_in = score_candidate(words_in)
        score_av = score_candidate(words_av)

        # Word-level token consensus alignment
        merged_words = []
        max_len = max(len(words_in), len(words_av))
        agreement_count = 0

        # Try to align words
        for i in range(max_len):
            w_in = words_in[i] if i < len(words_in) else None
            w_av = words_av[i] if i < len(words_av) else None

            if w_in is not None and w_av is not None:
                w_sim = fuzz.ratio(w_in, w_av)
                if w_sim >= 75:
                    # Both models agree on this word or close phonetics!
                    chosen = w_av if w_av in COMMON_WORDS else w_in
                    merged_words.append(chosen)
                    agreement_count += 1
                else:
                    # Conflict: evaluate which word has higher language validity
                    in_valid = w_in in COMMON_WORDS
                    av_valid = w_av in COMMON_WORDS
                    if in_valid and not av_valid:
                        merged_words.append(w_in)
                    elif av_valid and not in_valid:
                        merged_words.append(w_av)
                    else:
                        # Prefer in-house on short structured commands, auto-avsr on long open sentences
                        if len(words_av) > 4:
                            merged_words.append(w_av)
                        else:
                            merged_words.append(w_in)
            elif w_in is not None:
                if score_in >= score_av:
                    merged_words.append(w_in)
            elif w_av is not None:
                if score_av > score_in:
                    merged_words.append(w_av)

        consensus_text = " ".join(merged_words).strip()
        if not consensus_text:
            consensus_text = t_av if score_av >= score_in else t_in

        # Overall agreement score between 0 and 100
        agreement_score = round((similarity * 0.5) + (token_sort_sim * 0.5), 1)
        base_confidence = 82.0 + (agreement_score * 0.15)
        confidence = min(98.0, max(75.0, base_confidence))

        rationale = f"Consensus fusion: {agreement_count} aligned tokens ({agreement_score}% model agreement)"
        return consensus_text.capitalize(), round(confidence, 1), agreement_score, rationale

    def predict(self, video_path, mode="ensemble"):
        """
        Unified prediction entry point.
        mode can be:
          - "ensemble": runs both models and performs consensus fusion (Default)
          - "in_house": runs our in-house 3D-CNN + ResNet-18 + BiGRU model
          - "auto_avsr": runs Auto-AVSR Conformer model
        """
        t_global_start = time.time()

        # Compute video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        inhouse_res = None
        avsr_res = None

        if mode in ("ensemble", "in_house"):
            inhouse_res = self.predict_inhouse(video_path)

        if mode in ("ensemble", "auto_avsr"):
            avsr_res = self.predict_auto_avsr(video_path)

        total_time_ms = round((time.time() - t_global_start) * 1000)

        # Build visual analysis frames
        frames_np = inhouse_res.get('frames_np') if inhouse_res else None
        bbox = inhouse_res.get('bbox') if inhouse_res else None
        first_bgr = inhouse_res.get('first_bgr_frame') if inhouse_res else None
        
        visual_frames = {}
        if inhouse_res and frames_np is not None:
            try:
                visual_frames = self._extract_visual_frames(video_path, frames_np, bbox, first_bgr)
            except Exception:
                pass

        if not visual_frames and avsr_res and 'frames' in avsr_res:
            visual_frames = avsr_res['frames']

        # Determine output based on mode
        if mode == "in_house":
            t = inhouse_res.get('transcript', '') if inhouse_res else ''
            return {
                'transcript': t.capitalize() if t else '',
                'mode': 'in_house',
                'confidence': 88.0,
                'agreement_score': 100.0,
                'fusion_rationale': 'In-House Neural Engine prediction',
                'models': {
                    'in_house': {
                        'name': 'SilentSpeech Neural Engine (3D-CNN + ResNet-18 + BiGRU)',
                        'transcript': inhouse_res.get('transcript', '') if inhouse_res else '',
                        'latency_ms': inhouse_res.get('latency_ms', 0) if inhouse_res else 0,
                        'checkpoint': self.inhouse_checkpoint_name
                    }
                },
                'num_frames': inhouse_res.get('num_frames', total_frames) if inhouse_res else total_frames,
                'processing_time_ms': inhouse_res.get('latency_ms', total_time_ms) if inhouse_res else total_time_ms,
                'video_duration_s': round(duration, 2),
                'fps': round(fps, 1),
                'frames': visual_frames
            }

        elif mode == "auto_avsr":
            t = avsr_res.get('transcript', '') if avsr_res else ''
            return {
                'transcript': t.capitalize() if t else '',
                'mode': 'auto_avsr',
                'confidence': 90.0,
                'agreement_score': 100.0,
                'fusion_rationale': 'Auto-AVSR Conformer prediction',
                'models': {
                    'auto_avsr': {
                        'name': 'Auto-AVSR Large Conformer (LRS3 5,000-subword)',
                        'transcript': avsr_res.get('transcript', '') if avsr_res else '',
                        'latency_ms': avsr_res.get('latency_ms', 0) if avsr_res else 0
                    }
                },
                'num_frames': avsr_res.get('num_frames', total_frames) if avsr_res else total_frames,
                'processing_time_ms': avsr_res.get('latency_ms', total_time_ms) if avsr_res else total_time_ms,
                'video_duration_s': round(duration, 2),
                'fps': round(fps, 1),
                'frames': visual_frames
            }

        else: # "ensemble" - COMBINED PERFECT RESULT
            in_t = inhouse_res.get('transcript', '') if inhouse_res else ''
            av_t = avsr_res.get('transcript', '') if avsr_res else ''

            combined_transcript, confidence, agreement_score, rationale = self.fuse_transcripts(
                in_t, av_t, duration_s=duration
            )

            return {
                'transcript': combined_transcript,
                'mode': 'ensemble',
                'confidence': confidence,
                'agreement_score': agreement_score,
                'fusion_rationale': rationale,
                'models': {
                    'in_house': {
                        'name': 'SilentSpeech Neural Engine (3D-CNN + ResNet-18 + BiGRU)',
                        'transcript': in_t.capitalize() if in_t else 'No speech detected',
                        'latency_ms': inhouse_res.get('latency_ms', 0) if inhouse_res else 0,
                        'checkpoint': self.inhouse_checkpoint_name
                    },
                    'auto_avsr': {
                        'name': 'Auto-AVSR Large Conformer (LRS3 5,000-subword)',
                        'transcript': av_t.capitalize() if av_t else 'No speech detected',
                        'latency_ms': avsr_res.get('latency_ms', 0) if avsr_res else 0
                    }
                },
                'num_frames': inhouse_res.get('num_frames', total_frames) if inhouse_res else total_frames,
                'processing_time_ms': total_time_ms,
                'video_duration_s': round(duration, 2),
                'fps': round(fps, 1),
                'frames': visual_frames
            }

    def _extract_visual_frames(self, video_path, preprocessed_frames, bbox=None, first_bgr_frame=None):
        """Extract visualization frames for UI pipeline breakdown."""
        frames_data = {}
        if first_bgr_frame is not None:
            original_frame = first_bgr_frame
            ret = True
        else:
            cap = cv2.VideoCapture(video_path)
            ret, original_frame = cap.read()
            cap.release()

        if ret and original_frame is not None:
            h, w = original_frame.shape[:2]
            display_w = 320
            display_h = max(int(h * (display_w / w)), 1)
            original_resized = cv2.resize(original_frame, (display_w, display_h))
            frames_data['original'] = frame_to_base64(original_resized)

            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            if bbox is not None:
                crop_x1, crop_y1, crop_x2, crop_y2 = bbox
            else:
                crop_y1, crop_y2 = int(h * 0.5), int(h * 0.9)
                crop_x1, crop_x2 = int(w * 0.3), int(w * 0.7)

            crop_y1, crop_y2 = max(0, crop_y1), min(h, crop_y2)
            crop_x1, crop_x2 = max(0, crop_x1), min(w, crop_x2)

            crop = gray[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size > 0:
                crop_display = cv2.resize(crop, (160, 160))
                frames_data['cropped'] = frame_to_base64(crop_display, is_grayscale=True)

            annotated = original_resized.copy()
            scale_x = display_w / w
            scale_y = display_h / h
            pt1 = (int(crop_x1 * scale_x), int(crop_y1 * scale_y))
            pt2 = (int(crop_x2 * scale_x), int(crop_y2 * scale_y))
            cv2.rectangle(annotated, pt1, pt2, (99, 102, 241), 2)
            frames_data['annotated'] = frame_to_base64(annotated)

        if preprocessed_frames is not None and preprocessed_frames.shape[0] > 0:
            proc_frame = preprocessed_frames[0]
            proc_display = proc_frame.squeeze()
            proc_display_large = cv2.resize(
                (proc_display * 255).clip(0, 255).astype(np.uint8), (160, 160),
                interpolation=cv2.INTER_NEAREST
            )
            frames_data['processed'] = frame_to_base64(proc_display_large, is_grayscale=True)

        return frames_data

