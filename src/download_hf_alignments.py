import os
import tarfile
import shutil
from huggingface_hub import hf_hub_download

def main():
    s2_align_dir = 'data/grid_corpus/s2/align'
    
    # Clean up dummy files
    if os.path.exists(s2_align_dir):
        print(f"Deleting placeholder files in {s2_align_dir}...")
        for f in os.listdir(s2_align_dir):
            file_path = os.path.join(s2_align_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    os.makedirs(s2_align_dir, exist_ok=True)
    
    print("Downloading GRID_alignments.tar from Hugging Face SilentSpeak/EGCLLC...")
    
    tar_path = hf_hub_download(
        repo_id="SilentSpeak/EGCLLC", 
        repo_type="dataset", 
        filename="GRID_alignments.tar"
    )
    
    print(f"Downloaded to {tar_path}. Extracting...")
    
    tmp_extract_dir = "data/hf_tar_tmp"
    os.makedirs(tmp_extract_dir, exist_ok=True)
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=tmp_extract_dir)
        
    print("Extraction complete. Moving s2 alignments...")
    
    # Locate the s2 align directory within the extracted contents
    s2_src = None
    for root, dirs, files in os.walk(tmp_extract_dir):
        if 's2' in dirs:
            s2_cand = os.path.join(root, 's2')
            cand_files = os.listdir(s2_cand)
            if any(f.endswith('.align') for f in cand_files):
                s2_src = s2_cand
                break
            
            # Check if it's nested under s2/align/
            s2_align_cand = os.path.join(root, 's2', 'align')
            if os.path.exists(s2_align_cand) and any(f.endswith('.align') for f in os.listdir(s2_align_cand)):
                s2_src = s2_align_cand
                break

    if s2_src:
        count = 0
        for f in os.listdir(s2_src):
            if f.endswith('.align'):
                shutil.copy(os.path.join(s2_src, f), os.path.join(s2_align_dir, f))
                count += 1
        print(f"Successfully moved {count} alignment files to {s2_align_dir}.")
    else:
        print("Could not find s2 alignments directory in the extracted tar.")
        
    # Clean up the temporary extract folder
    shutil.rmtree(tmp_extract_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
