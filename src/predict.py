"""
SilentSpeech - In-House Neural Model CLI Inference Script
Uses VisualSpeechModel (3D-CNN + ResNet-18 + BiGRU) + OpenVocabularyDecoder
with adaptive MediaPipe EMA lip tracking for custom and benchmark videos.
"""

import os
import sys
import argparse
import time
import torch

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model import VisualSpeechModel
from dataset import CharTokenizer
from decoder import OpenVocabularyDecoder
from preprocess import dynamic_preprocess_video


def predict_our_model(video_path, checkpoint_path=None, device=None, beam_width=12):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_dir = os.path.dirname(src_dir)
    if checkpoint_path is None:
        candidates = [
            os.path.join(base_dir, 'models', 'best_wer.pth'),
            os.path.join(base_dir, 'models', 'best_cer.pth'),
            os.path.join(base_dir, 'models', 'best_model.pth'),
            os.path.join(base_dir, 'models', 'last_checkpoint.pth'),
        ]
        checkpoint_path = next((c for c in candidates if os.path.exists(c)), None)

    print(f"\n[SilentSpeech] Processing video: {video_path}...")
    t_start = time.time()

    frames_np, bbox, _ = dynamic_preprocess_video(video_path)
    if frames_np.shape[0] == 0 or (frames_np.shape[0] == 1 and frames_np.max() == 0):
        print("[SilentSpeech] Error: No face or lip movement could be detected.")
        return None

    num_frames = frames_np.shape[0]
    print(f"[SilentSpeech] Extracted {num_frames} normalized lip frames.")

    tokenizer = CharTokenizer()
    decoder = OpenVocabularyDecoder(tokenizer=tokenizer, beam_width=beam_width)
    model = VisualSpeechModel(vocab_size=tokenizer.vocab_size)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[SilentSpeech] Loading in-house checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"[SilentSpeech] Warning: Checkpoint not found at {checkpoint_path}")

    model.to(device)
    model.eval()

    frames_tensor = torch.tensor(frames_np, dtype=torch.float32).unsqueeze(0)
    frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4).to(device)

    with torch.no_grad():
        logits = model(frames_tensor)

    # Open-Vocabulary Beam Search Decoding
    preds = decoder.decode(logits, use_lexicon=True)
    result_text = preds[0] if preds else ""
    if result_text:
        result_text = result_text.capitalize()

    elapsed_ms = round((time.time() - t_start) * 1000)

    print("\n" + "="*60)
    print("SILENTSPEECH NEURAL PREDICTION RESULT:")
    print(f"Transcript: {result_text}")
    print(f"Frames:     {num_frames}")
    print(f"Inference:  {elapsed_ms} ms")
    print("="*60 + "\n")

    return {
        'transcript': result_text,
        'num_frames': num_frames,
        'processing_time_ms': elapsed_ms
    }


def main():
    parser = argparse.ArgumentParser(description="SilentSpeech In-House Neural Lip Reading CLI Inference")
    parser.add_argument('--video', type=str, required=True, help="Path to video file")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to .pth checkpoint file")
    parser.add_argument('--device', type=str, default=None, help="Computation device (cuda/cpu)")
    parser.add_argument('--beam_width', type=int, default=12, help="Beam width for CTC beam search")
    args = parser.parse_args()

    predict_our_model(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        device=args.device,
        beam_width=args.beam_width
    )


if __name__ == "__main__":
    main()
