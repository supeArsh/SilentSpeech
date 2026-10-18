import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from preprocess import load_and_preprocess_video

class CharTokenizer:
    def __init__(self, chars="abcdefghijklmnopqrstuvwxyz '"):
        # Basic character set for CTC: lowercase letters + space + apostrophe (vocab size 29)
        self.chars = chars
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.char_to_idx['<blank>'] = 0  # CTC blank token is usually 0
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text):
        text = text.lower()
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        
    def decode(self, indices):
        return "".join([self.idx_to_char[idx] for idx in indices if idx != 0])

def parse_align_file(filepath):
    """
    Parses a GRID corpus .align file into a string of words.
    Ignores 'sil' and 'sp' tokens.
    """
    words = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    word = parts[2]
                    if word not in ['sil', 'sp']:
                        words.append(word)
    return " ".join(words)

class LipReadingDataset(Dataset):
    def __init__(self, data_samples, transform=None):
        """
        data_samples: list of tuples (video_path, align_path)
        """
        self.data_samples = data_samples
        self.tokenizer = CharTokenizer()
        self.transform = transform
        
    def __len__(self):
        return len(self.data_samples)
        
    def __getitem__(self, idx):
        video_path, align_path = self.data_samples[idx]
        
        # Load transcript
        transcript = parse_align_file(align_path)
        
        # Load and preprocess frames using shared preprocessor (Frames, 1, 96, 96)
        frames_np = load_and_preprocess_video(video_path)
        video_tensor = torch.tensor(frames_np, dtype=torch.float32)
            
        # Encode transcript
        encoded_text = self.tokenizer.encode(transcript)
        text_tensor = torch.tensor(encoded_text, dtype=torch.long)
        
        return video_tensor, text_tensor

def collate_fn(batch):
    """
    Pads short sequences and stacks them into (Batch, Channels, Frames, Height, Width).
    Also prepares targets for CTC loss.
    """
    videos = []
    targets = []
    target_lengths = []
    input_lengths = []
    
    # Find max frames in the batch
    max_frames = max(v.size(0) for v, _ in batch)
    
    for video, target in batch:
        # video shape: (Frames, Channels, Height, Width)
        f, c, h, w = video.shape
        
        # Pad video tensor along the Frames dimension
        pad_size = max_frames - f
        if pad_size > 0:
            padding = torch.zeros((pad_size, c, h, w), dtype=torch.float32)
            padded_video = torch.cat((video, padding), dim=0)
        else:
            padded_video = video
            
        # Permute to (Channels, Frames, Height, Width) for 3D CNNs
        padded_video = padded_video.permute(1, 0, 2, 3)
        videos.append(padded_video)
        
        targets.append(target)
        input_lengths.append(f)
        target_lengths.append(len(target))
        
    # Stack to (Batch, Channels, Frames, Height, Width)
    videos_tensor = torch.stack(videos)
    
    # Concatenate targets into a single 1D tensor for CTCLoss
    targets_tensor = torch.cat(targets)
    
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return videos_tensor, targets_tensor, input_lengths, target_lengths

def get_dataloader(data_samples, batch_size=4, shuffle=True, num_workers=0):
    dataset = LipReadingDataset(data_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader

def get_grid_corpus_data(base_dir="data/grid_corpus", speakers=None):
    """
    Scans the data directory and pairs .mpg video files with their .align text files.
    Searches recursively to handle nested directories.
    If speakers is None, it discovers all subdirectories in base_dir.
    """
    data_samples = []
    
    if speakers is None:
        if os.path.exists(base_dir):
            speakers = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        else:
            speakers = []
            
    for speaker in speakers:
        video_dir = os.path.join(base_dir, speaker, "video")
        align_dir = os.path.join(base_dir, speaker, "align")
        
        if not os.path.exists(video_dir) or not os.path.exists(align_dir):
            continue
            
        align_files = {}
        for root, _, files in os.walk(align_dir):
            for f in files:
                if f.endswith('.align'):
                    align_files[os.path.splitext(f)[0]] = os.path.join(root, f)
                    
        for root, _, files in os.walk(video_dir):
            for f in files:
                if f.endswith('.mpg'):
                    video_name = os.path.splitext(f)[0]
                    video_path = os.path.join(root, f)
                    
                    if video_name in align_files:
                        data_samples.append((video_path, align_files[video_name]))
                        
    return data_samples

if __name__ == "__main__":
    # Test with real GRID data
    print("Gathering dataset files...")
    samples = get_grid_corpus_data()
    print(f"Found {len(samples)} valid video-alignment pairs.")
    
    if len(samples) > 0:
        loader = get_dataloader(samples, batch_size=2)
        for batch in loader:
            videos, targets, input_lens, target_lens = batch
            print(f"Videos shape (B, C, F, H, W): {videos.shape}")
            print(f"Targets shape (Total Chars): {targets.shape}")
            print(f"Input lengths: {input_lens}")
            print(f"Target lengths: {target_lens}")
            break
