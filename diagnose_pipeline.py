import os
import sys
import torch
import cv2
import numpy as np
import time

# Add src and current dir to path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))

from src.dataset import LipReadingDataset, CharTokenizer, parse_align_file
from src.model import VisualSpeechModel
from src.train import greedy_ctc_decode
from src.preprocess import load_and_preprocess_video as eval_preprocess
import app
import jiwer

def main():
    print("=== UI Pipeline Diagnostics ===")
    
    # 1. Load the UI model and print its metadata
    success = app.load_model()
    if not success:
        print("Failed to load UI model.")
        return
        
    print("\n--- UI Checkpoint Info ---")
    print(f"Checkpoint Path: models/{app._checkpoint_meta.get('checkpoint')}")
    print(f"Epoch: {app._checkpoint_meta.get('epoch')}")
    print(f"Recorded Metrics: {app._checkpoint_meta.get('metrics')}")
    print(f"Vocabulary Size: {app._tokenizer.vocab_size}")
    
    # 2. Select 10 validation videos
    val_txt = 'splits/val.txt'
    if not os.path.exists(val_txt):
        print(f"Validation list {val_txt} not found.")
        return
        
    with open(val_txt, 'r') as f:
        val_files = [line.strip() for line in f.readlines() if line.strip()]
        
    sample_files = val_files[:10]
    
    data_samples = []
    for line in sample_files:
        parts = line.split('\t')
        video_path = parts[0].strip()
        align_path = parts[1].strip() if len(parts) > 1 else video_path.replace('video', 'align').replace('.mpg', '.align')
        # Handle the fact that paths might have SilentSpeech/ prefix because the script runs inside SilentSpeech directory
        if video_path.startswith("SilentSpeech/"):
            video_path = video_path[13:]
        if align_path.startswith("SilentSpeech/"):
            align_path = align_path[13:]
        data_samples.append((video_path, align_path))
    
    # 3. Setup Evaluation Pipeline exactly as in training/evaluation
    val_dataset = LipReadingDataset(data_samples)
    
    print("\n--- Pipeline Comparison on 10 Validation Videos ---")
    
    eval_preds = []
    ui_preds = []
    targets = []
    
    for i, (video_path, align_path) in enumerate(data_samples):
        # Target
        frames_eval, target_eval = val_dataset[i]
        
        target_str = app._tokenizer.decode(target_eval.tolist())
        targets.append(target_str)
        
        # --- Evaluation Pipeline ---
        frames_eval_batch = frames_eval.permute(1, 0, 2, 3).unsqueeze(0).to(app._device)
        
        with torch.no_grad():
            eval_logits = app._model(frames_eval_batch)
            eval_pred_list = greedy_ctc_decode(eval_logits, app._tokenizer)
            eval_pred_str = eval_pred_list[0] if eval_pred_list else ""
        eval_preds.append(eval_pred_str)
        
        # --- UI Pipeline ---
        frames_ui_np, _ = app.dynamic_preprocess_video(video_path)
        if frames_ui_np.shape[0] == 0:
            ui_pred_str = "<NO FRAMES>"
        else:
            frames_ui_tensor = torch.tensor(frames_ui_np, dtype=torch.float32).unsqueeze(0)
            frames_ui_tensor = frames_ui_tensor.permute(0, 2, 1, 3, 4).to(app._device)
            
            with torch.no_grad():
                ui_logits = app._model(frames_ui_tensor)
                ui_pred_list = greedy_ctc_decode(ui_logits, app._tokenizer)
                ui_pred_str = ui_pred_list[0] if ui_pred_list else ""
        
        ui_preds.append(ui_pred_str)
        
        print(f"\nVideo {i+1}: {os.path.basename(video_path)}")
        print(f"Target : {target_str}")
        print(f"Eval   : {eval_pred_str}")
        print(f"UI     : {ui_pred_str}")
        
    print("\n--- Error Metrics ---")
    eval_wer = jiwer.wer(targets, eval_preds)
    eval_cer = jiwer.cer(targets, eval_preds)
    ui_wer = jiwer.wer(targets, ui_preds)
    ui_cer = jiwer.cer(targets, ui_preds)
    
    print(f"Evaluation Pipeline - WER: {eval_wer:.4f}, CER: {eval_cer:.4f}")
    print(f"UI Inference Pipeline - WER: {ui_wer:.4f}, CER: {ui_cer:.4f}")

if __name__ == '__main__':
    main()
