"""
SilentSpeech - Fast Resumable LRS3 Parquet Downloader
Downloads and verifies all LRS3 open-vocabulary sentence video partitions for neural training.
"""

import os
import sys
import requests
import pyarrow.parquet as pq

FILES = [
    {
        "name": "train-00000.parquet",
        "url": "https://huggingface.co/datasets/mattymchen/lrs3-test/resolve/main/data/train-00000-of-00002-6d235766a16b101a.parquet",
        "cached": r"C:\Users\sparsh chauhan\.cache\huggingface\hub\datasets--mattymchen--lrs3-test\blobs\83ed27da72318d23ac316fcbbdb7834818fe90e095cce036b5c69d3d5a0e71cb.1a08b683.incomplete"
    },
    {
        "name": "train-00001.parquet",
        "url": "https://huggingface.co/datasets/mattymchen/lrs3-test/resolve/main/data/train-00001-of-00002-a89b175a3b0aa629.parquet",
        "cached": None
    }
]

def download_file(item, target_dir):
    target_file = os.path.join(target_dir, item["name"])
    url = item["url"]

    head_resp = requests.head(url, allow_redirects=True, timeout=30)
    total_size = int(head_resp.headers.get("content-length", 0))
    print(f"\n[LRS3] Downloading {item['name']} (Total: {total_size / (1024*1024):.2f} MB)...")

    # If cached file exists and target doesn't
    if not os.path.exists(target_file) and item.get("cached") and os.path.exists(item["cached"]):
        print(f"[LRS3] Reusing existing cached blob ({os.path.getsize(item['cached']) / (1024*1024):.2f} MB)...")
        with open(item["cached"], "rb") as fin, open(target_file, "wb") as fout:
            fout.write(fin.read())

    # Download with automatic retry and byte range resumption
    max_retries = 15
    for attempt in range(max_retries):
        current_size = os.path.getsize(target_file) if os.path.exists(target_file) else 0
        if current_size >= total_size and total_size > 0:
            print(f"[LRS3] File complete ({current_size / (1024*1024):.2f} MB).")
            break

        print(f"[LRS3] Attempt {attempt+1}/{max_retries} - Resuming from {current_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB...")
        headers = {"Range": f"bytes={current_size}-{total_size - 1}"}
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=60)
            if resp.status_code not in (200, 206):
                print(f"[WARN] HTTP {resp.status_code}, retrying...")
                time.sleep(2)
                continue

            with open(target_file, "ab") as fout:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fout.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (5 * 1024 * 1024) == 0:
                            pct = ((current_size + downloaded) / total_size) * 100
                            print(f"[{pct:.1f}%]", end=" ", flush=True)
            print()
            break
        except Exception as e:
            print(f"\n[LRS3] Connection hiccup ({e}). Retrying in 2s...")
            import time
            time.sleep(2)

    # Verify parquet integrity
    table = pq.read_table(target_file)
    print(f"[OK] {item['name']}: {table.num_rows} valid samples. Columns: {table.column_names}")
    return target_file, table.num_rows

def main():
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lrs3")
    os.makedirs(target_dir, exist_ok=True)

    total_samples = 0
    for item in FILES:
        _, rows = download_file(item, target_dir)
        total_samples += rows

    print(f"\n[LRS3 DATASET COMPLETE] Total open-vocabulary samples ready: {total_samples}")

if __name__ == "__main__":
    main()
