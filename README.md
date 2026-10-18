# SilentSpeech: AI Lip Reading & Silent Speech Recognition

SilentSpeech is an AI-powered visual speech recognition (lip-reading) application. It decodes spoken words directly from silent video by analyzing lip movements and mouth dynamics.

The system includes a runtime **Model Switch Lever** allowing users to switch between an open-vocabulary deep learning model and a closed-vocabulary baseline, or compare both side by side.

---

## Models Included

SilentSpeech supports two distinct visual speech recognition engines:

### 1. Auto-AVSR (Recommended - Default)
* **Architecture:** Visual frontend (ResNet-3D + Conformer encoder-decoder) with SentencePiece Unigram subword tokenization (5,000 subword units).
* **Vocabulary:** Open Vocabulary. Can transcribe unconstrained, natural English sentences and arbitrary phrases.
* **Pre-training:** Trained on 400+ hours of TED talks from the LRS3-TED dataset.
* **Weights:** `models/auto_avsr/pytorch_model.pt` + `models/auto_avsr/unigram5000.model`.
* **Execution:** Fast GPU inference (~1.5s per video clip) with automatic CPU fallback.

### 2. GRID Baseline
* **Architecture:** 3D CNN + 2D ResNet-18 + Bidirectional GRU + CTC Decoder.
* **Vocabulary:** Closed 51-word vocabulary following the fixed GRID corpus grammar formula:
  `<command> <color> <preposition> <letter> <digit> <adverb>`
  (e.g., *"bin blue at F 2 soon"*).
* **Purpose:** Serves as an academic baseline to demonstrate the difference between closed-vocabulary pattern matching and modern open-vocabulary lip reading.
* **Weights:** `models/last_checkpoint.pth`.

### 3. Compare Both (Side-by-Side)
* Runs the same uploaded video through both Auto-AVSR and the GRID baseline sequentially.
* Shows both outputs side by side with execution times and vocabulary badges, making it ideal for college demonstrations and comparative analysis.

---

## Quick Start (How to Run)

### Prerequisites
* Windows 10/11 with Python 3.10+
* NVIDIA GPU with CUDA support recommended (CPU fallback is automatically supported)
* Virtual environment with PyTorch and dependencies installed

### 1. Activate Environment
Open PowerShell or Command Prompt in the project root directory:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Run the Web Application
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## How to Switch Models

You can switch models in two ways:

### Method A: Interactive UI Lever (Recommended)
On the web interface, use the **Model Selector Lever** located above the upload zone:
* **Auto-AVSR (Open Vocabulary)**: Recommended for all general, natural sentences.
* **GRID Model (Baseline)**: For fixed 51-word vocabulary GRID sentences.
* **Compare Both (Side-by-Side)**: Runs both models on the video and displays both transcripts side by side.

### Method B: Environment Variable
You can configure the server default model at startup by setting the `MODEL_CHOICE` environment variable:
```powershell
# Set default to Auto-AVSR
$env:MODEL_CHOICE="auto_avsr"
python app.py

# Set default to GRID Baseline
$env:MODEL_CHOICE="grid"
python app.py
```

---

## API Endpoints

### 1. `GET /api/status`
Returns system status, active model, GPU availability, and model registry.
```json
{
  "active_model": "auto_avsr",
  "available_models": ["auto_avsr", "grid", "compare"],
  "device": "cuda",
  "model_loaded": true,
  "status": "ready"
}
```

### 2. `POST /api/predict`
Uploads a video and returns transcription.
* **Form data:**
  * `video`: Video file (`.mp4`, `.avi`, `.mov`, `.webm`)
  * `model`: Optional choice (`auto_avsr`, `grid`, or `compare`). Defaults to server active model.
* **Response (Single Model):**
```json
{
  "model": "auto_avsr",
  "model_name": "Auto-AVSR (Open Vocabulary)",
  "transcript": "hello world",
  "time_seconds": 1.48,
  "vocab": "Open Vocabulary (SentencePiece 5,000 units)"
}
```
* **Response (Compare Mode):**
```json
{
  "model": "compare",
  "model_name": "Model Comparison (Auto-AVSR vs GRID)",
  "auto_avsr": {
    "transcript": "hello world",
    "time_seconds": 1.43,
    "vocab": "Open Vocabulary (SentencePiece 5,000 units)"
  },
  "grid": {
    "transcript": "bin blue at f two now",
    "time_seconds": 0.51,
    "vocab": "51 Words (Fixed Grammar Formula)"
  }
}
```

---

## Academic Credits and Acknowledgements

* **Auto-AVSR:** Developed by Pingchuan Ma, Stavros Petridis, and Maja Pantic (University of Oxford and Imperial College London).
  * Paper: *"Visual Speech Recognition for Small Datasets"* / *"Auto-AVSR: Audio-Visual Speech Recognition System"*
  * Source: Pre-trained weights and SentencePiece unigram model based on the official Auto-AVSR LRS3 visual speech recognition release.
* **GRID Corpus:** University of Sheffield / Cooke et al., used as the baseline comparison dataset.
* **MediaPipe Face Mesh:** Google LLC, used for mouth region-of-interest extraction and stabilization.

---

## College Demo Notes
This project is configured strictly for academic evaluation and demonstration.
* Lazy loading ensures models are loaded into memory on first request.
* VRAM is preserved by running comparison models sequentially and calling `torch.cuda.empty_cache()` between runs.
* Automatic CPU fallback prevents server crashes if GPU memory is constrained.
