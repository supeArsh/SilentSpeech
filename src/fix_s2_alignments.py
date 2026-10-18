import os
import requests
import zipfile
import shutil

def download_and_extract(url, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    filepath = os.path.join(extract_to, 'alignments.zip')
    
    print(f"Downloading {url} to {filepath}...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return
        
    print(f"Extracting {filepath}...")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        
    print("Removing archive...")
    if os.path.exists(filepath):
        os.remove(filepath)

def main():
    url = "https://zenodo.org/records/3625687/files/alignments.zip?download=1"
    tmp_dir = "data/tmp_alignments"
    download_and_extract(url, tmp_dir)
    
    # Search for s2 alignments folder in the extracted zip
    s2_src = None
    for root, dirs, files in os.walk(tmp_dir):
        if 's2' in dirs:
            s2_cand = os.path.join(root, 's2')
            cand_files = os.listdir(s2_cand)
            if any(f.endswith('.align') for f in cand_files):
                s2_src = s2_cand
                break

    if s2_src:
        s2_dest = "data/grid_corpus/s2/align"
        os.makedirs(s2_dest, exist_ok=True)
        print(f"Moving alignments from {s2_src} to {s2_dest}...")
        for f in os.listdir(s2_src):
            if f.endswith('.align'):
                shutil.move(os.path.join(s2_src, f), os.path.join(s2_dest, f))
        print("Moved Speaker 2 alignments successfully!")
    else:
        print("Could not find s2 alignments directory in the extracted zip.")
        
    # Clean up
    print(f"Cleaning up {tmp_dir}...")
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
