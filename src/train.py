import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader, CharTokenizer
from model import VisualSpeechModel
import os
import sys
import jiwer

# Enable line buffering on stdout for immediate logging
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

def greedy_ctc_decode(logits, tokenizer):
    """
    Greedy decoding for CTC.
    logits: (Batch, Frames, Vocab Size)
    """
    pred_indices = torch.argmax(logits, dim=-1) # (B, T)
    decoded_preds = []
    
    for i in range(pred_indices.size(0)):
        pred = pred_indices[i]
        # Collapse repeated tokens and remove blanks
        collapsed = []
        prev_idx = -1
        for idx in pred:
            idx = idx.item()
            if idx != prev_idx:
                if idx != 0:  # 0 is the CTC blank token
                    collapsed.append(idx)
            prev_idx = idx
        
        decoded_str = tokenizer.decode(collapsed)
        decoded_preds.append(decoded_str)
        
    return decoded_preds

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1, num_epochs=30):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (videos, targets, input_lens, target_lens) in enumerate(dataloader):
        videos = videos.to(device)
        targets = targets.to(device)
        
        # PyTorch CTCLoss expects logits of shape (T, B, vocab_size)
        logits = model(videos) # (B, T, V)
        logits = logits.permute(1, 0, 2) # (T, B, V)
        
        # Log softmax is required before passing to CTCLoss
        log_probs = nn.functional.log_softmax(logits, dim=2)
        
        loss = criterion(log_probs, targets, input_lens, target_lens)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping with maximum norm 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
            avg_so_far = total_loss / (batch_idx + 1)
            print(f"  [Epoch {epoch:02d}/{num_epochs:02d}] Batch {batch_idx + 1:03d}/{num_batches:03d} - Batch Loss: {loss.item():.4f} (Avg: {avg_so_far:.4f})", flush=True)
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for videos, targets, input_lens, target_lens in dataloader:
            videos = videos.to(device)
            targets = targets.to(device)
            
            logits = model(videos) # (B, T, V)
            
            # For loss computation
            logits_t = logits.permute(1, 0, 2)
            log_probs = nn.functional.log_softmax(logits_t, dim=2)
            loss = criterion(log_probs, targets, input_lens, target_lens)
            total_loss += loss.item()
            
            # Decode predictions
            preds = greedy_ctc_decode(logits, tokenizer)
            all_preds.extend(preds)
            
            # Reconstruct string targets from flattened 1D target tensor
            start = 0
            for length in target_lens:
                target_slice = targets[start:start+length].tolist()
                target_str = tokenizer.decode(target_slice)
                all_targets.append(target_str)
                start += length
                
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    # Calculate CER and WER using JiWER
    safe_preds = [p if p.strip() else "<empty>" for p in all_preds]
    safe_targets = [t if t.strip() else "<empty>" for t in all_targets]
    
    try:
        cer = jiwer.cer(safe_targets, safe_preds)
        wer = jiwer.wer(safe_targets, safe_preds)
    except Exception as e:
        print(f"Error calculating CER/WER: {e}", flush=True)
        cer, wer = 1.0, 1.0
        
    return avg_loss, cer, wer, all_preds, all_targets

def save_checkpoint(filepath, epoch, model, optimizer, scheduler, tokenizer, metrics, project_root):
    ckpt_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'tokenizer_config': {
            'chars': tokenizer.chars,
            'vocab_size': tokenizer.vocab_size
        },
        'preprocessing_config': {
            'crop_y': [0.5, 0.9],
            'crop_x': [0.3, 0.7],
            'target_size': [96, 96],
            'grayscale': True,
            'normalization': '[0.0, 1.0]'
        },
        'metrics': metrics
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(ckpt_dict, filepath)
    
    # Mirror checkpoint
    try:
        if "SilentSpeech" in os.path.abspath(filepath):
            alt_path = os.path.abspath(filepath).replace(os.path.join("SilentSpeech", "models"), "models")
        else:
            alt_path = os.path.join(project_root, "SilentSpeech", "models", os.path.basename(filepath))
        if os.path.abspath(alt_path) != os.path.abspath(filepath):
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            torch.save(ckpt_dict, alt_path)
    except Exception:
        pass

def main():
    import argparse
    import random
    from dataset import get_grid_corpus_data
    
    # Locate project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    parent_root = os.path.dirname(project_root) if os.path.basename(project_root) == "SilentSpeech" else project_root
    
    # Default resume checkpoint search
    default_resume = None
    candidate_checkpoints = [
        os.path.join(project_root, "models", "last_checkpoint.pth"),
        os.path.join(parent_root, "models", "last_checkpoint.pth"),
        os.path.join(project_root, "models", "best_model_epoch10_baseline.pth"),
        os.path.join(parent_root, "models", "best_model_epoch10_baseline.pth"),
    ]
    for cp in candidate_checkpoints:
        if os.path.exists(cp):
            default_resume = cp
            break
    if default_resume is None:
        default_resume = 'models/last_checkpoint.pth'
    
    parser = argparse.ArgumentParser(description="Train Visual Speech Recognition Model (Phase 2)")
    parser.add_argument('--start-epoch', type=int, default=None, help="Starting epoch number (auto-detected if None)")
    parser.add_argument('--max-epochs', type=int, default=20, help="Maximum total epochs to train until")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate (resumes from checkpoint if None)")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience on validation CER")
    parser.add_argument('--resume', type=str, default=default_resume, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    splits_train = os.path.join(project_root, "splits", "train.txt")
    splits_val = os.path.join(project_root, "splits", "val.txt")
    if not (os.path.exists(splits_train) and os.path.exists(splits_val)):
        splits_train = os.path.join(parent_root, "splits", "train.txt")
        splits_val = os.path.join(parent_root, "splits", "val.txt")
    
    if os.path.exists(splits_train) and os.path.exists(splits_val):
        print(f"Loading split files from {splits_train} and {splits_val}...", flush=True)
        with open(splits_train, 'r', encoding='utf-8') as f:
            train_data = [line.strip().split('\t') for line in f if line.strip()]
        with open(splits_val, 'r', encoding='utf-8') as f:
            val_data = [line.strip().split('\t') for line in f if line.strip()]
        print(f"Loaded {len(train_data)} train samples and {len(val_data)} val samples.", flush=True)
    else:
        grid_dir = os.path.join(project_root, "data", "grid_corpus")
        samples = get_grid_corpus_data(base_dir=grid_dir)
        random.seed(42)
        random.shuffle(samples)
        split_idx = int(len(samples) * 0.8)
        train_data = samples[:split_idx]
        val_data = samples[split_idx:]
        print(f"Dataset split: {len(train_data)} train, {len(val_data)} validation samples.", flush=True)
        
    train_loader = get_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Fixed 20 validation clips for tracking predictions across epochs
    fixed_val_samples = val_data[:20]
    fixed_val_loader = get_dataloader(fixed_val_samples, batch_size=args.batch_size, shuffle=False)
    
    tokenizer = CharTokenizer()
    model = VisualSpeechModel(vocab_size=tokenizer.vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    initial_lr = args.lr if args.lr is not None else 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # ReduceLROnPlateau based on validation CER with factor 0.5 and patience 2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Check baseline checkpoint resume
    resume_path = args.resume
    if not os.path.exists(resume_path):
        for candidate in [os.path.join(project_root, args.resume), os.path.join(parent_root, args.resume)]:
            if os.path.exists(candidate):
                resume_path = candidate
                break

    print(f"Loading checkpoint to resume: {resume_path}", flush=True)
    assert os.path.exists(resume_path), f"Checkpoint not found at {resume_path}!"
    checkpoint = torch.load(resume_path, map_location=device)
    
    has_opt = False
    ckpt_epoch = None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_epoch = checkpoint.get("epoch", None)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                has_opt = True
            except Exception:
                has_opt = False
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("Loaded scheduler state from checkpoint.", flush=True)
            except Exception as e:
                print(f"Note: Could not restore scheduler state: {e}", flush=True)
    else:
        model.load_state_dict(checkpoint)
        has_opt = False
        
    if not has_opt:
        print(f"Optimizer state not present in checkpoint. Initializing fresh AdamW optimizer (lr: {initial_lr}).", flush=True)
    else:
        current_opt_lr = optimizer.param_groups[0]['lr']
        print(f"Loaded existing optimizer state from checkpoint (lr: {current_opt_lr:.6e}).", flush=True)
        if args.lr is not None:
            for g in optimizer.param_groups:
                g['lr'] = args.lr
            print(f"Overrode optimizer learning rate with CLI value: {args.lr:.6e}", flush=True)

    # Determine start epoch
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif ckpt_epoch is not None:
        start_epoch = ckpt_epoch + 1
    else:
        start_epoch = 11

    # Baseline validation evaluation
    print(f"\nRunning validation evaluation on resumed checkpoint (Epoch {ckpt_epoch if ckpt_epoch else 'initial'})...", flush=True)
    base_val_loss, base_cer, base_wer, _, _ = validate(model, val_loader, criterion, device, tokenizer)
    print(f"--> Resumed Checkpoint: Val Loss: {base_val_loss:.4f} | CER: {base_cer:.4f} | WER: {base_wer:.4f}\n", flush=True)
    
    best_cer = base_cer
    best_wer = base_wer
    
    # Check existing best_cer and best_wer checkpoints to prevent overwriting true records
    for cer_candidate in [os.path.join(project_root, "models", "best_cer.pth"), os.path.join(parent_root, "models", "best_cer.pth")]:
        if os.path.exists(cer_candidate):
            try:
                c = torch.load(cer_candidate, map_location='cpu')
                if isinstance(c, dict) and 'metrics' in c and 'cer' in c['metrics']:
                    cand_cer = float(c['metrics']['cer'])
                    if cand_cer < best_cer:
                        best_cer = cand_cer
                        print(f"Preserving existing recorded best CER: {best_cer:.4f} from {cer_candidate}", flush=True)
                    break
            except Exception:
                pass

    for wer_candidate in [os.path.join(project_root, "models", "best_wer.pth"), os.path.join(parent_root, "models", "best_wer.pth")]:
        if os.path.exists(wer_candidate):
            try:
                w = torch.load(wer_candidate, map_location='cpu')
                if isinstance(w, dict) and 'metrics' in w and 'wer' in w['metrics']:
                    cand_wer = float(w['metrics']['wer'])
                    if cand_wer < best_wer:
                        best_wer = cand_wer
                        print(f"Preserving existing recorded best WER: {best_wer:.4f} from {wer_candidate}", flush=True)
                    break
            except Exception:
                pass

    patience_counter = 0
    
    # Initialize CSV log file
    csv_file = os.path.join(project_root, "training_log.csv")
    csv_mirror = os.path.join(parent_root, "training_log.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("epoch,train_loss,val_loss,cer,wer,learning_rate,best_cer,best_wer\n")
            
    print(f"Starting training from Epoch {start_epoch} to {args.max_epochs} ({args.max_epochs - start_epoch + 1} epochs)...", flush=True)
    print(f"Early stopping patience: {args.patience} epochs on CER | Gradient clipping: 1.0\n", flush=True)
    
    for epoch in range(start_epoch, args.max_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n==================================================", flush=True)
        print(f"=== EPOCH {epoch:02d}/{args.max_epochs:02d} (Learning Rate: {current_lr:.6e}) ===", flush=True)
        print(f"==================================================", flush=True)
        
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_epochs=args.max_epochs
        )
        val_loss, cer, wer, preds, targets = validate(
            model, val_loader, criterion, device, tokenizer
        )
        
        # Step LR scheduler based on validation CER
        scheduler.step(cer)
        
        # Check improvements
        cer_improved = False
        wer_improved = False
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'cer': cer,
            'wer': wer,
            'learning_rate': current_lr
        }
        
        if cer < best_cer:
            best_cer = cer
            cer_improved = True
            patience_counter = 0
            cer_ckpt = os.path.join(project_root, "models", "best_cer.pth")
            save_checkpoint(cer_ckpt, epoch, model, optimizer, scheduler, tokenizer, metrics, project_root)
            print(f"    >> [NEW BEST CER] Saved checkpoint to models/best_cer.pth (CER: {best_cer:.4f})", flush=True)
        else:
            patience_counter += 1
            
        if wer < best_wer:
            best_wer = wer
            wer_improved = True
            wer_ckpt = os.path.join(project_root, "models", "best_wer.pth")
            save_checkpoint(wer_ckpt, epoch, model, optimizer, scheduler, tokenizer, metrics, project_root)
            print(f"    >> [NEW BEST WER] Saved checkpoint to models/best_wer.pth (WER: {best_wer:.4f})", flush=True)
            
        # Always save last checkpoint
        last_ckpt = os.path.join(project_root, "models", "last_checkpoint.pth")
        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, tokenizer, metrics, project_root)
        
        # Summary log
        print(f"\n--> Epoch {epoch:02d}/{args.max_epochs:02d} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | CER: {cer:.4f} | WER: {wer:.4f}", flush=True)
        print(f"    Current Best CER: {best_cer:.4f} | Current Best WER: {best_wer:.4f} | Patience: {patience_counter}/{args.patience}", flush=True)
        
        # Append to CSV
        log_line = f"{epoch},{train_loss:.4f},{val_loss:.4f},{cer:.4f},{wer:.4f},{current_lr:.6e},{best_cer:.4f},{best_wer:.4f}\n"
        with open(csv_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
        try:
            with open(csv_mirror, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception:
            pass
            
        # Print fixed 20 validation predictions
        print(f"\n--- Fixed 20 Validation Predictions (Epoch {epoch:02d}) ---", flush=True)
        _, _, _, f_preds, f_targets = validate(model, fixed_val_loader, criterion, device, tokenizer)
        for i, (fp, ft) in enumerate(zip(f_preds, f_targets), start=1):
            print(f"  [{i:02d}] Target: '{ft}'\n       Pred:   '{fp}'", flush=True)
        print("----------------------------------------------------------\n", flush=True)
        
        # Early stopping check
        if patience_counter >= args.patience:
            print(f"\n[EARLY STOPPING TRIGGERED] Validation CER did not improve for {args.patience} consecutive epochs. Stopping at epoch {epoch}.", flush=True)
            break

    print(f"\nPhase 2 training finished. Best CER: {best_cer:.4f}, Best WER: {best_wer:.4f}", flush=True)

if __name__ == "__main__":
    main()
