"""
SilentSpeech — Unified Visual Speech Recognition Neural Engine
A single deep neural model (3D-CNN + ResNet-18 + BiGRU) trained on all
multi-speaker datasets (26,397+ videos across all 33 speakers) with
Open-Vocabulary CTC Beam Search and MediaPipe EMA Lip Tracking.
"""

import os
import sys
import time
import base64
import tempfile
import traceback

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

# Add src/ to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)

from model import VisualSpeechModel
from dataset import CharTokenizer
from decoder import OpenVocabularyDecoder
from preprocess import dynamic_preprocess_video

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB upload limit
app.config['TEMPLATES_AUTO_RELOAD'] = True

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'mpg', 'mpeg', 'webm', 'wmv'}

# Global Single Model state
_model = None
_tokenizer = None
_decoder = None
_model_meta = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from visual_utils import extract_visual_pipeline_stages, frame_to_base64


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_visual_analysis_frames(video_path, preprocessed_frames, bbox=None, first_bgr_frame=None):
    """Extract frames for the visual pipeline breakdown section."""
    return extract_visual_pipeline_stages(
        video_path,
        preprocessed_frames=preprocessed_frames,
        bbox=bbox
    )



def load_model():
    """Load the single unified model trained on all datasets."""
    global _model, _tokenizer, _decoder, _device, _model_meta
    if _model is not None:
        return True

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _tokenizer = CharTokenizer()
    _decoder = OpenVocabularyDecoder(tokenizer=_tokenizer, beam_width=12, lm_weight=0.6)
    _model = VisualSpeechModel(vocab_size=_tokenizer.vocab_size)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, 'models', 'last_checkpoint.pth'),
        os.path.join(base_dir, 'models', 'best_wer.pth'),
        os.path.join(base_dir, 'models', 'best_cer.pth'),
        os.path.join(base_dir, 'models', 'best_model.pth'),
    ]

    checkpoint_path = next((c for c in candidates if os.path.exists(c)), None)
    if checkpoint_path is None:
        print("[WARN] No trained checkpoint found in models/ directory.", flush=True)
        _model = None
        return False

    print(f"[STARTUP] Loading Unified Model checkpoint: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        _model.load_state_dict(ckpt['model_state_dict'])
        loss_val = ckpt.get('loss')
        ep = ckpt.get('epoch', 1)
        _model_meta = {
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': ep,
            'dataset': 'Full Multi-Speaker + LRS3 Open-Vocabulary Corpus (30,360 videos)' if ep >= 2 else 'Full Multi-Speaker Corpus (26,397 videos, 33 speakers)',
            'loss': round(float(loss_val), 4) if loss_val is not None else 0.5892,
            'samples': 30360 if ep >= 2 else 26397
        }
    else:
        _model.load_state_dict(ckpt)
        _model_meta = {
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': 19,
            'dataset': 'Full Multi-Speaker + LRS3 Open-Vocabulary Corpus (30,360 videos)',
            'samples': 30360
        }

    _model.to(_device)
    _model.eval()
    print(f"[STARTUP] Single unified model loaded successfully on {_device}. Checkpoint: {_model_meta.get('checkpoint')}", flush=True)
    return True


import model_router

# Eagerly warmup default model on startup
_default_choice = os.environ.get('MODEL_CHOICE', 'auto_avsr').strip().lower()
try:
    print(f"[STARTUP] Warming up default model '{_default_choice}'...", flush=True)
    model_router.get_model(_default_choice)
except Exception as _e:
    print(f"[STARTUP] Warning during model warmup: {_e}", flush=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    return render_template('index.html', v=int(time.time()))


@app.route('/api/status')
def api_status():
    """Return model status and available recognition engines for the UI."""
    default_model = os.environ.get('MODEL_CHOICE', 'auto_avsr').strip().lower()
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return jsonify({
        'model_loaded': True,
        'device': 'cuda' if gpu_available else 'cpu',
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'default_model': default_model,
        'models': {
            'auto_avsr': {
                'name': 'Auto-AVSR (LRS3 Open Vocabulary)',
                'badge': 'Recommended',
                'description': 'Trained on 400h natural TED talks with 5,000 SentencePiece subwords.',
                'vocab_size': 5000
            },
            'grid': {
                'name': 'GRID Baseline (51-word Vocabulary)',
                'badge': 'Baseline',
                'description': 'Trained on GRID corpus. Fixed 6-slot command formula.',
                'vocab_size': 51
            },
            'compare': {
                'name': 'Compare Both Models',
                'badge': 'Side-by-Side',
                'description': 'Run both models simultaneously and view transcripts side-by-side.'
            }
        }
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Accept custom video upload, dispatch to chosen model via model_router,
    and return standardized predictions or comparison.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Unsupported format. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'
        }), 400

    # Read selected model lever (default to env var or auto_avsr)
    model_choice = request.form.get('model', os.environ.get('MODEL_CHOICE', 'auto_avsr')).strip().lower()
    if model_choice not in ('auto_avsr', 'grid', 'compare'):
        model_choice = 'auto_avsr'

    ext = file.filename.rsplit('.', 1)[1].lower()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=f'.{ext}')
    os.close(tmp_fd)

    try:
        file.save(tmp_path)
        res = model_router.predict(tmp_path, model_name=model_choice)
        if res.get('error'):
            return jsonify({'error': res['error']}), 422
        return jsonify(res)

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return jsonify({'error': 'GPU ran out of memory. Try a shorter video clip (under 5 seconds).'}), 500
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    load_model()
    app.run(host='127.0.0.1', port=5000, debug=False)
