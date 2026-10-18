import os
import urllib.request
import zipfile
import tarfile
import sys

def report_hook(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        # Cap at 100% since sometimes it can overshoot slightly depending on block boundaries
        percent = min(percent, 100)
        sys.stdout.write(f"\rDownloading: {percent}%")
        sys.stdout.flush()

def count_files_with_ext(directory, ext):
    if not os.path.exists(directory):
        return 0
    count = 0
    for _, _, files in os.walk(directory):
        for f in files:
            if f.endswith(ext):
                count += 1
    return count

def download_and_extract(url, extract_to, is_tar=False):
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)
    
    print(f"Downloading {url} to {filepath}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            total_size = int(response.headers.get('content-length', 0))
            count = 0
            block_size = 8192
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                count += 1
                report_hook(count, block_size, total_size)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return
        
    print(f"Extracting {filename}...")
    try:
        if is_tar:
            with tarfile.open(filepath, "r:*") as tar:
                tar.extractall(path=extract_to)
        else:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        print(f"Extraction complete for {filename}.")
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        
    print(f"Removing archive {filename}...\n")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

def main():
    # Resolve base_dir relative to this script's directory for reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "data", "grid_corpus")
    os.makedirs(base_dir, exist_ok=True)
    
    # GRID Corpus endpoints for normal resolution videos and alignments
    speakers = [i for i in range(1, 35) if i != 21]
    for speaker_id in speakers:
        speaker_str = f"s{speaker_id}"
        speaker_dir = os.path.join(base_dir, speaker_str)
        
        video_dir = os.path.join(speaker_dir, "video")
        align_dir = os.path.join(speaker_dir, "align")
        
        # Check if videos already exist
        video_count = count_files_with_ext(video_dir, ".mpg")
        if video_count >= 50:
            print(f"[{speaker_str}] Found {video_count} existing video files in {video_dir}. Skipping video download.")
        else:
            video_url = f"http://spandh.dcs.shef.ac.uk/gridcorpus/{speaker_str}/video/{speaker_str}.mpg_vcd.zip"
            download_and_extract(video_url, video_dir, is_tar=False)
        
        # Check if alignments already exist
        align_count = count_files_with_ext(align_dir, ".align")
        if align_count >= 50:
            print(f"[{speaker_str}] Found {align_count} existing alignment files in {align_dir}. Skipping alignment download.")
        else:
            # Note: Alignments archive in GRID corpus contains an 'align/' folder at its root,
            # so extracting into speaker_dir places files directly into speaker_dir/align/
            align_url = f"http://spandh.dcs.shef.ac.uk/gridcorpus/{speaker_str}/align/{speaker_str}.tar"
            download_and_extract(align_url, speaker_dir, is_tar=True)
        
    print("\nAll downloads and extractions check complete!")

if __name__ == "__main__":
    main()
