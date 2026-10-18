"""
SilentSpeech - Open-Vocabulary Training & Fine-Tuning Pipeline
Trains our in-house VisualSpeechModel on open-vocabulary conversational datasets
and custom user video recordings with Mixed Precision (AMP) and SpecAugment.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import jiwer

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model import VisualSpeechModel
from dataset import CharTokenizer
from decoder import OpenVocabularyDecoder
from preprocess import dynamic_preprocess_video, load_and_preprocess_video


class OpenVocabLipDataset(Dataset):
    """
    Dataset loader for open-vocabulary visual speech recognition:
    Supports:
      1. Standard TSV/TXT split files: <video_path>\t<transcript_or_align_path>
      2. Custom recording directory: pairs of .mp4 and .txt files
    """
    def __init__(self, samples, transform=None, augment=True):
        self.samples = samples
        self.tokenizer = CharTokenizer()
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Handle LRS3 open-vocabulary parquet sample
        if isinstance(sample, tuple) and len(sample) == 4 and sample[0] == 'lrs3_parquet':
            _, p_path, r_idx, transcript = sample
            if not hasattr(self, '_parquet_cache'):
                self._parquet_cache = {}
            if p_path not in self._parquet_cache:
                import pyarrow.parquet as pq
                self._parquet_cache[p_path] = pq.read_table(p_path)
            table = self._parquet_cache[p_path]
            raw_v = np.array(table['video'][r_idx].as_py(), dtype=np.float32) / 255.0
            # Normalize to standard visual speech distribution
            raw_v = (raw_v - 0.421) / 0.165
            video_tensor = torch.tensor(raw_v, dtype=torch.float32).unsqueeze(0)  # (1, T, 96, 96)

        else:
            video_path, text = sample

            # If text is a path to an alignment or transcript file, load it
            if os.path.isfile(text):
                if text.endswith('.align'):
                    from dataset import parse_align_file
                    transcript = parse_align_file(text)
                else:
                    with open(text, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
            else:
                transcript = str(text).strip()

            # Load video frames
            # Use dynamic_preprocess_video for custom videos or standard preprocessor
            if 'grid_corpus' in video_path:
                frames_np = load_and_preprocess_video(video_path)
            else:
                frames_np, _, _ = dynamic_preprocess_video(video_path)

            # Convert to float tensor: (T, 1, 96, 96)
            video_tensor = torch.tensor(frames_np, dtype=torch.float32)

            # Permute to (C, T, H, W) = (1, T, 96, 96)
            video_tensor = video_tensor.permute(1, 0, 2, 3)

        # Data augmentation during training: Temporal Masking
        if self.augment and video_tensor.shape[1] > 10:
            T = video_tensor.shape[1]
            mask_len = np.random.randint(1, max(2, int(T * 0.15)))
            t0 = np.random.randint(0, T - mask_len)
            video_tensor[:, t0:t0+mask_len, :, :] = 0.0

            # Horizontal flip augmentation (50% chance)
            if np.random.rand() > 0.5:
                video_tensor = torch.flip(video_tensor, dims=[3])

        # Encode transcript into character indices
        target_indices = self.tokenizer.encode(transcript)
        if len(target_indices) == 0:
            target_indices = [1]
        target_tensor = torch.tensor(target_indices, dtype=torch.long)

        input_len = video_tensor.shape[1]
        target_len = len(target_indices)

        return video_tensor, target_tensor, input_len, target_len


def collate_open_vocab(batch):
    videos, targets, input_lens, target_lens = zip(*batch)

    # Pad videos along time dimension T
    max_t = max(v.shape[1] for v in videos)
    C, _, H, W = videos[0].shape
    B = len(videos)

    padded_videos = torch.zeros(B, C, max_t, H, W, dtype=torch.float32)
    for i, v in enumerate(videos):
        t = v.shape[1]
        padded_videos[i, :, :t, :, :] = v

    # Concatenate targets into 1D tensor for PyTorch CTCLoss
    concatenated_targets = torch.cat(targets, dim=0)

    input_lengths = torch.tensor(input_lens, dtype=torch.long)
    target_lengths = torch.tensor(target_lens, dtype=torch.long)

    return padded_videos, concatenated_targets, input_lengths, target_lengths


def train_open_vocab(
    checkpoint_path=None,
    epochs=15,
    batch_size=8,
    lr=3e-5,
    max_samples=None,
    custom_data_dir=None,
    device='cuda'
):
    """
    Fine-tunes VisualSpeechModel for open vocabulary and custom video adaptation.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"SILENTSPEECH OPEN-VOCABULARY NEURAL TRAINING")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Target Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"{'='*65}\n")

    tokenizer = CharTokenizer()
    model = VisualSpeechModel(vocab_size=tokenizer.vocab_size)
    decoder = OpenVocabularyDecoder(tokenizer=tokenizer, beam_width=8)

    # Load existing best checkpoint as base
    base_dir = os.path.dirname(src_dir)
    if checkpoint_path is None:
        candidates = [
            os.path.join(base_dir, 'models', 'last_checkpoint.pth'),
            os.path.join(base_dir, 'models', 'best_wer.pth'),
            os.path.join(base_dir, 'models', 'best_cer.pth'),
            os.path.join(base_dir, 'models', 'best_model.pth'),
        ]
        checkpoint_path = next((c for c in candidates if os.path.exists(c)), None)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[STARTUP] Initializing from pretrained checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"[STARTUP] Loaded weights (previous epoch: {ckpt.get('epoch', 'N/A')})")
        else:
            model.load_state_dict(ckpt)
    else:
        print("[STARTUP] No previous checkpoint specified; training from initialization.")

    model.to(device)

    # Prepare dataset samples
    train_samples = []
    val_samples = []

    # 1. Custom dataset if provided
    if custom_data_dir and os.path.exists(custom_data_dir):
        print(f"[DATA] Scanning custom video dataset in {custom_data_dir}...")
        for root, _, files in os.walk(custom_data_dir):
            for f in files:
                if f.endswith(('.mp4', '.avi', '.mov', '.webm')):
                    v_path = os.path.join(root, f)
                    t_path = os.path.splitext(v_path)[0] + '.txt'
                    if os.path.exists(t_path):
                        train_samples.append((v_path, t_path))

    # 2. Existing split dataset (GRID 33 speakers)
    split_dir = os.path.join(base_dir, 'splits')
    train_txt = os.path.join(split_dir, 'train.txt')
    val_txt = os.path.join(split_dir, 'val.txt')

    def resolve_sample_path(rel_path):
        p = rel_path.strip().replace('\\', '/')
        if p.startswith('SilentSpeech/'):
            p = p[13:]
        # Try relative to base_dir (SilentSpeech/)
        cand1 = os.path.normpath(os.path.join(base_dir, p))
        if os.path.exists(cand1):
            return cand1
        # Try relative to workspace root
        cand2 = os.path.normpath(os.path.join(os.path.dirname(base_dir), p))
        if os.path.exists(cand2):
            return cand2
        # Try relative to workspace root with SilentSpeech prepended
        cand3 = os.path.normpath(os.path.join(os.path.dirname(base_dir), 'SilentSpeech', p))
        if os.path.exists(cand3):
            return cand3
        return None

    if os.path.exists(train_txt):
        with open(train_txt, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    v_p = resolve_sample_path(parts[0])
                    a_p = resolve_sample_path(parts[1])
                    if v_p and a_p:
                        train_samples.append((v_p, a_p))

    if os.path.exists(val_txt):
        with open(val_txt, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    v_p = resolve_sample_path(parts[0])
                    a_p = resolve_sample_path(parts[1])
                    if v_p and a_p:
                        val_samples.append((v_p, a_p))

    # 3. LRS3 Open-Vocabulary Dataset (Conversational TED talks)
    lrs3_dir = os.path.join(base_dir, 'data', 'lrs3')
    lrs3_samples = []
    if os.path.exists(lrs3_dir):
        import glob
        import pyarrow.parquet as pq
        p_files = sorted(glob.glob(os.path.join(lrs3_dir, "*.parquet")))
        for p_file in p_files:
            try:
                table = pq.read_table(p_file, columns=['label'])
                for r_idx in range(table.num_rows):
                    lbl = table['label'][r_idx].as_py()
                    if lbl and len(str(lbl).strip()) > 0:
                        lrs3_samples.append(('lrs3_parquet', p_file, r_idx, str(lbl).strip()))
            except Exception as e:
                print(f"[WARN] Failed scanning {p_file}: {e}")
        print(f"[DATA] Ingested {len(lrs3_samples)} unique LRS3 open-vocabulary sentence samples.")

    # Interleave and balance datasets:
    # Upsample LRS3 conversational data 3x to ensure strong open-vocabulary representation
    if lrs3_samples:
        train_samples.extend(lrs3_samples * 3)

    if max_samples and max_samples < len(train_samples):
        import random
        # Always include all LRS3 samples
        always_lrs3 = lrs3_samples * 2
        remaining_budget = max(0, max_samples - len(always_lrs3))
        grid_only = [s for s in train_samples if not (isinstance(s, tuple) and len(s) == 4 and s[0] == 'lrs3_parquet')]
        sampled_grid = random.sample(grid_only, min(remaining_budget, len(grid_only)))
        train_samples = always_lrs3 + sampled_grid
        random.shuffle(train_samples)

    print(f"[DATA] Loaded {len(train_samples)} training samples total, {len(val_samples)} validation samples.")

    if not train_samples:
        print("[ERROR] No valid training samples found.")
        return

    train_dataset = OpenVocabLipDataset(train_samples, augment=True)
    val_dataset = OpenVocabLipDataset(val_samples[:200], augment=False) if val_samples else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_open_vocab,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_meta = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt_meta, dict) and 'epoch' in ckpt_meta:
            start_epoch = ckpt_meta['epoch'] + 1

    best_loss = float('inf')

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_idx, (videos, targets, input_lens, target_lens) in enumerate(train_loader):
            videos = videos.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(videos) # (B, T, V)
                logits_t = logits.permute(1, 0, 2) # (T, B, V)
                log_probs = nn.functional.log_softmax(logits_t, dim=2)
                loss = criterion(log_probs, targets, input_lens, target_lens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                avg_l = total_loss / (batch_idx + 1)
                print(f"  [Epoch {epoch:02d}/{start_epoch+epochs-1:02d}] Step {batch_idx+1:04d}/{len(train_loader):04d} - Loss: {loss.item():.4f} (Avg: {avg_l:.4f})", flush=True)

            # Periodic checkpoint saving every 300 steps
            if (batch_idx + 1) % 300 == 0:
                p_save = os.path.join(base_dir, 'models', 'last_checkpoint.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / (batch_idx + 1),
                }, p_save)

        scheduler.step()
        epoch_loss = total_loss / len(train_loader)
        elapsed = round(time.time() - t0, 1)

        print(f"\n--> Epoch {epoch:02d} Completed in {elapsed}s | Train Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save latest checkpoint
        save_path = os.path.join(base_dir, 'models', 'last_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, save_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            b_path = os.path.join(base_dir, 'models', 'best_wer.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': epoch_loss,
            }, b_path)
            print(f"[CHECKPOINT] Saved new best model to {b_path} (Loss: {epoch_loss:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SilentSpeech Open-Vocabulary Model Training")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--custom_dir', type=str, default=None)
    args = parser.parse_args()

    train_open_vocab(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        custom_data_dir=args.custom_dir
    )
