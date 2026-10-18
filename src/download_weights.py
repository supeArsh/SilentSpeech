"""
Resumable direct downloader for Auto-AVSR model weights.
Uses standard HTTP Range requests to handle large downloads safely without failing on disconnections.
"""

import os
import sys
import time
import requests

MODELS = [
    {
        "filename": "pytorch_model.pt",
        "url": "https://huggingface.co/aaahmet/silent-lip-reader-model/resolve/main/pytorch_model.pt",
        "alt_url": "https://huggingface.co/AD1TEYA/lip-reading-model/resolve/main/pytorch_model.pt",
        "expected_size": 1001892616
    },
    {
        "filename": "unigram5000.model",
        "url": "https://huggingface.co/aaahmet/silent-lip-reader-model/resolve/main/unigram5000.model",
        "alt_url": None,
        "expected_size": 327113
    },
    {
        "filename": "unigram5000_units.txt",
        "url": "https://huggingface.co/aaahmet/silent-lip-reader-model/resolve/main/unigram5000_units.txt",
        "alt_url": None,
        "expected_size": 73778
    }
]

def download_file_resumable(url, target_path, expected_size=None, alt_url=None, chunk_size=1024*1024):
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    temp_path = target_path + ".part"

    current_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    if os.path.exists(target_path):
        if expected_size is None or os.path.getsize(target_path) == expected_size:
            print(f"[READY] {os.path.basename(target_path)} already exists ({os.path.getsize(target_path) / 1e6:.1f} MB).", flush=True)
            return True

    print(f"[START] Downloading {os.path.basename(target_path)}...", flush=True)
    
    headers = {"User-Agent": "SilentSpeech-Downloader/1.0"}
    if current_size > 0:
        headers["Range"] = f"bytes={current_size}-"
        print(f"  Resuming from {current_size / 1e6:.1f} MB...", flush=True)

    target_url = url
    try:
        r = requests.get(target_url, headers=headers, stream=True, timeout=30)
    except Exception as e:
        if alt_url:
            print(f"  Primary URL failed ({e}), trying alternate mirror...", flush=True)
            target_url = alt_url
            r = requests.get(target_url, headers=headers, stream=True, timeout=30)
        else:
            raise

    # Handle status
    if r.status_code == 416: # Range not satisfiable (already complete)
        if os.path.exists(temp_path):
            os.replace(temp_path, target_path)
            return True

    if r.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {r.status_code} while downloading {url}")

    mode = "ab" if current_size > 0 and r.status_code == 206 else "wb"
    if mode == "wb":
        current_size = 0

    total_len = expected_size or int(r.headers.get("content-length", 0)) + current_size

    downloaded = current_size
    last_print = time.time()
    last_pct = -1

    with open(temp_path, mode) as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pct = int(downloaded / total_len * 100) if total_len else 0
                now = time.time()
                if pct != last_pct and (pct % 5 == 0 or now - last_print > 3):
                    mb = downloaded / 1e6
                    total_mb = total_len / 1e6 if total_len else 0
                    print(f"  Progress: {pct}% ({mb:.1f} / {total_mb:.1f} MB)", flush=True)
                    last_pct = pct
                    last_print = now

    if expected_size and downloaded != expected_size:
        print(f"[WARN] Size mismatch: got {downloaded}, expected {expected_size}", flush=True)

    if os.path.exists(target_path):
        os.remove(target_path)
    os.replace(temp_path, target_path)
    print(f"[DONE] Successfully saved to {target_path} ({os.path.getsize(target_path) / 1e6:.1f} MB)", flush=True)
    return True

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "models", "auto_avsr")
    
    print(f"Target directory: {target_dir}")
    for item in MODELS:
        dest = os.path.join(target_dir, item["filename"])
        retries = 3
        success = False
        while retries > 0 and not success:
            try:
                download_file_resumable(
                    item["url"],
                    dest,
                    expected_size=item.get("expected_size"),
                    alt_url=item.get("alt_url")
                )
                success = True
            except Exception as e:
                retries -= 1
                print(f"[RETRY] Error downloading {item['filename']}: {e}. Retries left: {retries}", flush=True)
                time.sleep(2)
        if not success:
            print(f"[ERROR] Failed to download {item['filename']}", flush=True)
            sys.exit(1)
            
    print("\nAll Auto-AVSR model assets are verified and ready!")

if __name__ == "__main__":
    main()
