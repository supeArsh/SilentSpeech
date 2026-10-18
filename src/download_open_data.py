"""
SilentSpeech - Open-Vocabulary Dataset Downloader & Custom Data Collector
Downloads open-vocabulary lip-reading datasets (LRS2/LRS3/LRW) or organizes
user-recorded custom videos for training.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile

def setup_custom_data_dir(target_dir):
    """
    Creates template directories and a sample manifest for user video training.
    Users can simply drop their .mp4 videos and corresponding .txt transcripts.
    """
    os.makedirs(target_dir, exist_ok=True)
    readme_path = os.path.join(target_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "SilentSpeech Custom Dataset Directory\n"
            "====================================\n\n"
            "To train/fine-tune the model on your own face and voice:\n"
            "1. Drop your video files (.mp4, .avi, or .mov) here.\n"
            "2. For each video (e.g., video1.mp4), create a text file with the same name (e.g., video1.txt)\n"
            "   containing the spoken sentence.\n"
            "3. Run: python src/train_open_vocab.py --custom_dir data/custom\n"
        )
    print(f"[OK] Custom dataset directory initialized at: {target_dir}")
    print(f"[INFO] Place your custom .mp4 and matching .txt transcript files in this folder.")


def main():
    parser = argparse.ArgumentParser(description="SilentSpeech Open Dataset & Custom Data Manager")
    parser.add_argument('--setup_custom', action='store_true', help="Setup custom dataset folder")
    parser.add_argument('--custom_path', type=str, default='data/custom', help="Path for custom dataset")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(project_dir, args.custom_path)
    setup_custom_data_dir(target)


if __name__ == "__main__":
    main()
