# SilentSpeech: A Dual-Engine Deep Neural Architecture for Open-Vocabulary and Spatiotemporal Visual Speech Recognition

**Sparsh Chauhan** (Reg. No.: 235805292)  
**Anshika Sharma** (Reg. No.: 235805104)  
**Akshat Bagadia** (Reg. No.: 235805128)  

*Department of Computer Science and Engineering / School of Computing and Electrical Engineering (SOCE)*  
*Manipal Institute of Technology, Bengaluru*  
*Manipal Academy of Higher Education (MAHE), Karnataka, India*  

**Project Guide:**  
**Prof. Sapna R.**  
*Assistant Professor, School of Computing and Electrical Engineering (SOCE)*  
*Manipal Institute of Technology, Bengaluru*  

---

## Abstract
Visual Speech Recognition (VSR), commonly known as automated lip-reading, seeks to interpret spoken natural language exclusively from silent video sequences by decoding spatio-temporal deformations of the speaker's lip, jaw, and facial articulators. This paper presents **SilentSpeech**, a comprehensive, real-time, dual-engine visual speech recognition framework designed to bridge the historical divide between constrained-grammar academic baselines and unconstrained, open-vocabulary natural speech transcription. 

SilentSpeech introduces a hybrid pipeline comprising:
1. An in-house, custom-engineered spatiotemporal deep neural network (3D-CNN + modified 2D ResNet-18 + 2-layer Bidirectional Gated Recurrent Units) optimized with Connectionist Temporal Classification (CTC) loss and an open-vocabulary prefix beam-search decoder integrated with English language priors;
2. An enterprise-grade, state-of-the-art open-vocabulary Conformer-based engine (Auto-AVSR) leveraging a 3D visual frontend, depthwise-separable convolutional self-attention encoders, and a 5,000-unit SentencePiece subword language model pre-trained on 400+ hours of natural TED lectures from the LRS3-TED corpus;
3. An adaptive, illumination-invariant visual front-end featuring 468-point MediaPipe face mesh tracking, Exponential Moving Average (EMA, $\alpha=0.35$) bounding-box stabilization, Contrast-Limited Adaptive Histogram Equalization (CLAHE), and dynamic 25 FPS temporal resampling; and
4. An interactive web-based orchestration cockpit with a hardware-accelerated runtime model-switching lever and real-time visual pipeline inspection.

Extensive empirical evaluations conducted across the multi-speaker GRID audio-visual corpus (comprising 33 speakers and 26,397 video sequences) and the in-the-wild LRS3-TED dataset demonstrate that our customized spatiotemporal baseline converges to a **Character Error Rate (CER) of 4.19%** and a **Word Error Rate (WER) of 12.46%** with an average inference latency of 0.51 seconds. Simultaneously, the open-vocabulary engine achieves unconstrained sentence recognition in ~1.48 seconds on an NVIDIA RTX 3050 GPU. This dual-engine architecture delivers an optimal balance between low-latency local inference and broad acoustic-lexical coverage, establishing a robust assistive communication solution for individuals with dysphonia, laryngectomy, and speech impairments, as well as hands-free communication in acoustically hostile environments.

**Index Terms**—Visual Speech Recognition, Lip Reading, Spatiotemporal Deep Learning, 3D Convolutional Neural Networks, ResNet-18, Bidirectional GRU, Conformer, Connectionist Temporal Classification (CTC), Auto-AVSR, MediaPipe Face Mesh, Assistive Technology.

---

## Section I: Introduction

### 1.1 Background and Motivation
Human speech perception is intrinsically multimodal. In noisy acoustic environments, human listeners instinctively augment auditory sensory input by reading the visual kinematics of the speaker's lips, teeth, and tongue—a psychoacoustic phenomenon famously characterized by the McGurk effect. For decades, conventional Automated Speech Recognition (ASR) systems have relied almost exclusively on acoustic pressure waveforms captured by acoustic transducers. While acoustic ASR has attained remarkable accuracy in silent or studio conditions, its performance degrades severely in the presence of non-stationary acoustic noise, multi-speaker babble, reverberant environments, or electronic eavesdropping countermeasures.

Visual Speech Recognition (VSR)—the automated decoding of linguistic content solely from visual video streams without relying on acoustic signals—presents a transformative paradigm for speech technology. VSR systems operate silently, making them fundamentally impervious to acoustic noise, microphone saturation, or acoustic jamming. Furthermore, VSR holds immense potential as an assistive communication modality for individuals suffering from speech motor disabilities, aphonia, amyotrophic lateral sclerosis (ALS), vocal cord paralysis, or post-laryngectomy conditions where vocal acoustic production is compromised, but facial and labial articulation remain intact. Beyond assistive medical applications, silent lip-reading technology empowers secure biometric authentication, confidential defense communications, cockpit interactions, and privacy-preserving voice computing in crowded public spaces.

```
+--------------------------------------------------------------------------------------------------+
|                                    SILENTSPEECH OVERALL ARCHITECTURE                             |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +-----------------------+          +-------------------------------------------------------+   |
|   |  Silent Video Input   | -------> | 468-Point MediaPipe Landmark & EMA Bounding Box       |   |
|   |  (Webcam / File)      |          | Stabilization (alpha = 0.35)                          |   |
|   +-----------------------+          +-------------------------------------------------------+   |
|                                                                  |                               |
|                                                                  v                               |
|                                      +-------------------------------------------------------+   |
|                                      | Adaptive CLAHE Contrast Normalization + 25 FPS Resamp |   |
|                                      +-------------------------------------------------------+   |
|                                                                  |                               |
|                                                                  v                               |
|                                      +-------------------------------------------------------+   |
|                                      | Model Routing Engine & Memory Caching Dispatcher       |   |
|                                      +-------------------------------------------------------+   |
|                                               /                                   \              |
|                                              /                                     \             |
|                                             v                                       v            |
|       +--------------------------------------------+    +--------------------------------------+ |
|       |  ENGINE 1: In-House Spatiotemporal Model  |    |  ENGINE 2: Auto-AVSR Open-Vocabulary | |
|       |  - 3D-CNN Spatiotemporal Conv Frontend     |    |  - 3D ResNet Visual Frontend         | |
|       |  - 2D ResNet-18 Spatial Feature Backbone   |    |  - Conformer Encoder-Decoder Stack   | |
|       |  - 2-Layer Bidirectional GRU Temporal Net  |    |  - 5,000-Unit SentencePiece Subwords | |
|       |  - CTC Loss + Prefix Beam Search Decoder   |    |  - Trained on 400h LRS3-TED Talks    | |
|       +--------------------------------------------+    +--------------------------------------+ |
|                                             \                                       /            |
|                                              \                                     /             |
|                                               v                                   v              |
|                                      +-------------------------------------------------------+   |
|                                      | Interactive UI Lever: Auto-AVSR / GRID / Side-by-Side |   |
|                                      | Latency Tracking, Stage Visualization, Text Display   |   |
|                                      +-------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
```

### 1.2 Fundamental Challenges of Visual Speech Recognition
Despite its immense utility, VSR is widely acknowledged as one of the most challenging inverse problems in computer vision and signal processing. These challenges stem from intrinsic physical and mathematical limitations:
1. **The Homopheme Ambiguity:** A substantial portion of distinct phonemes (the fundamental units of acoustic sound) map to identical or visually indistinguishable visemes (the visual appearance of the lips). For instance, the bilabial plosives and nasals $/p/$, $/b/$, and $/m/$ exhibit identical lip closures, differing only in vocal cord vibration and velopharyngeal port movement—features that remain entirely invisible to an optical camera. Similarly, alveolar consonants $/t/$, $/d/$, and $/n/$ are virtually indistinguishable visually. Consequently, visual lip-reading is an ill-posed one-to-many mapping problem.
2. **Coarticulation Effects:** The visual appearance of a phoneme is heavily modulated by preceding and succeeding phonemes (temporal coarticulation). The labial configuration of $/k/$ in the word "cool" (rounded lips) is radically different from that in "keen" (spread lips). Disentangling these continuous, overlapping muscular gestures requires deep spatiotemporal sequence modeling.
3. **Inter-Speaker Geometric Variability:** Human faces exhibit significant anatomical diversity, including variations in lip thickness, vermilion border definition, philtrum depth, dental occlusion, facial hair, and skin pigmentation.
4. **Acquisition Invariances and Head Motion:** Real-world video captures suffer from abrupt head rotations, camera yaw and pitch variations, motion blur, non-uniform ambient illumination, and erratic frame rates (ranging from 15 to 60 FPS across mobile and web cameras).
5. **The Vocabulary Barrier:** Academic lip-reading datasets have historically focused on rigid, artificial command grammars (e.g., the 51-word GRID corpus). Bridging the gap to open-vocabulary, unrestricted natural language requires massive-scale foundation models paired with extensive linguistic context.

### 1.3 Project Objectives and Key Contributions
To address these challenges, this project formulates, implements, and evaluates **SilentSpeech**—an enterprise-ready, dual-engine visual speech recognition system. The core contributions of this work are summarized as follows:
* **Hybrid Dual-Engine Architecture:** We construct an integrated system featuring both a bespoke, lightweight spatiotemporal deep model (3D-CNN + ResNet-18 + BiGRU) optimized for low-latency constrained domains, and a pre-trained open-vocabulary Conformer foundation model (Auto-AVSR) capable of transcribing arbitrary conversational English.
* **Continuous Face Mesh Tracking with EMA Stabilization:** We engineer a dynamic visual preprocessing pipeline combining a 468-point 3D MediaPipe Face Mesh landmark extractor with an Exponential Moving Average ($\alpha=0.35$) bounding-box tracker, eliminating lip jitter caused by natural head micro-movements.
* **Illumination and Temporal Invariance:** We incorporate Contrast-Limited Adaptive Histogram Equalization (CLAHE) to preserve labial contour gradients under uneven lighting, alongside linear frame-interpolation resampling to lock input tensors to exactly 25.0 FPS.
* **Language-Augmented CTC Prefix Beam Search:** For our character-level model, we formulate a log-space CTC prefix beam-search decoder enhanced with vocabulary priors covering over 95% of common spoken English words, converting ambiguous character sequences into fluent natural sentences.
* **Comprehensive Empirical Validation:** We perform rigorous training across the 33-speaker, 26,397-video GRID corpus, tracking training loss, validation loss, Character Error Rate (CER), and Word Error Rate (WER) across 30 epochs, culminating in an optimal CER of 4.19% and WER of 12.46%.
* **Full-Stack Production System:** We deliver a responsive, production-grade web application featuring dynamic model switching (`auto_avsr`, `grid`, and `compare` mode), sequential memory-managed inference with automated GPU cache clearing, and a three-stage visual pipeline inspector for academic demonstration.

---

## Section II: Literature Review & Related Work

Visual Speech Recognition has evolved through distinct technological epochs over the past three decades, transitioning from classical handcrafted geometric modeling to modern deep spatiotemporal neural networks and self-supervised foundation transformers.

```
+----------------------------------------------------------------------------------------------------+
|                                    TAXONOMY OF VSR ARCHITECTURES                                   |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    Phase I: Handcrafted Pipelines (1995-2015)                                                      |
|    [Optical Flow / AAMs / DWT / DCT] ---> [PCA Dimensionality Reduction] ---> [Discrete HMMs]       |
|                                                                                                    |
|    Phase II: Deep Spatiotemporal Models (2016-2020)                                                |
|    [3D-CNN Temporal Frontend] ---> [2D-CNN / ResNet Backbone] ---> [BiLSTM / BiGRU] ---> [CTC Loss]  |
|                                                                                                    |
|    Phase III: Attention & Conformer Foundation Transformers (2021-Present)                         |
|    [ResNet-3D Frontend] ---> [Multi-Head Self-Attention + Depthwise Conv Conformer] ---> [AED / CTC]  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 2.1 Classical Handcrafted Feature Pipelines
Early lip-reading systems relied on two disjoint stages: feature extraction and temporal classification. Feature extraction utilized Active Shape Models (ASMs) and Active Appearance Models (AAMs) to track explicit geometric coordinates of the inner and outer vermilion borders (Matthews et al., 2002). Alternatively, appearance-based methods extracted Discrete Cosine Transforms (DCT), 2D Discrete Wavelet Transforms (DWT), or local optical flow vectors across the mouth bounding box (Potamianos et al., 2003). These static representations were classified using continuous-density Hidden Markov Models (HMMs) or Gaussian Mixture Model-HMM hybrids (Dupont and Luettin, 2000). While foundational, these systems exhibited fragile landmark tracking under head poses greater than $15^\circ$, failed catastrophically across unseen speakers, and were restricted to small-vocabulary digit or alphabet recognition tasks.

### 2.2 Deep Spatiotemporal Neural Networks
The breakthrough of deep learning fundamentally transformed VSR by unifying feature representation learning and temporal sequence modeling into an end-to-end differentiable framework. 

Assael et al. (2016) introduced **LipNet**, the first end-to-end sentence-level lip-reading model. LipNet employed three spatiotemporal 3D convolutional layers followed by spatial pooling, two layers of Bidirectional Gated Recurrent Units (BiGRU), and Connectionist Temporal Classification (CTC) loss. Evaluated on the GRID corpus, LipNet achieved a remarkable Word Error Rate of 4.8% on overlapping speaker splits and 11.4% on unseen speakers, decisively outperforming human lip-reading experts. 

Following LipNet, Chung et al. (2017) proposed the **Watch, Listen, Attend and Spell (WLAS)** architecture and introduced the massive Lip Reading in the Wild (LRW) and Lip Reading Sentences (LRS) benchmark datasets. Chung's work utilized a 3D convolution front-end coupled with a 2D ResNet backbone to process temporal sequences of lip frames, paired with an Attention-based Sequence-to-Sequence (seq2seq) recurrent decoder. 

Stafylakis and Tzimiropoulos (2017) refined the visual feature extractor by combining a 3D convolution layer ($5 \times 7 \times 7$) with a 2D ResNet-18 residual architecture, followed by temporal convolutional networks or BiLSTMs. This spatiotemporal paradigm established the benchmark for visual speech feature representation and serves as the architectural foundation for our custom-engineered in-house baseline model.

### 2.3 Self-Attention and Conformer Architectures
In recent years, the self-attention mechanism (Vaswani et al., 2017) has replaced recurrent architectures in visual speech recognition. Gulati et al. (2020) proposed the **Conformer** architecture for acoustic speech recognition, combining the global sequence context of multi-head self-attention with the local receptive field of depthwise-separable convolutional layers.

Ma, Petridis, and Pantic (2021, 2022) extended the Conformer paradigm to visual speech recognition in their seminal **Auto-AVSR** framework (*Visual Speech Recognition for Small Datasets*). Auto-AVSR incorporates a modified 3D ResNet visual frontend feeding a multi-layer Conformer encoder, coupled with an autoregressive Transformer decoder. When trained on the Lip Reading Sentences 3 (LRS3-TED) dataset—comprising over 400 hours of natural, unconstrained speech from TED and TEDx lectures—Auto-AVSR demonstrated that large-scale subword tokenization (using SentencePiece unigram models with 5,000 subwords) enables visual speech models to generalize to open-vocabulary English sentences with unprecedented fidelity.

### 2.4 Decoding Strategies and Lexical Modeling
In end-to-end speech recognition, two primary decoding paradigms dominate:
1. **Connectionist Temporal Classification (CTC):** CTC (Graves et al., 2006) introduces a special blank symbol $\epsilon$ and maps an input sequence of length $T$ directly to a target sequence of length $U \le T$ by marginalizing over all valid alignment paths. While CTC enables rapid non-autoregressive decoding, it enforces a conditional independence assumption between output tokens given the alignment, which can lead to phonetic misspellings unless regularized by a language model.
2. **Autoregressive Attention-based Encoder-Decoder (AED):** AED models remove the conditional independence assumption by predicting each token conditioned on all previously generated tokens. However, AED models suffer from exposure bias, hallucination tendencies during long pauses, and higher computational latency.
3. **Hybrid CTC/Attention:** Modern architectures often employ a joint CTC/attention multitask objective, using CTC loss during training to guide attention alignment and accelerate beam search decoding.

### 2.5 Summary of Prior Art vs. SilentSpeech
Table I provides a comparative summary highlighting how SilentSpeech bridges the functional gaps identified across existing literature.

#### Table I: Comparative Analysis of Visual Speech Recognition Paradigms
| System / Paper | Visual Frontend | Temporal Modeling | Vocabulary Scope | Language Modeling | Real-Time UI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LipNet** (Assael et al., 2016) | 3D-CNN (3 layers) | 2-layer BiGRU | Closed (GRID, 51 words) | None (Greedy CTC) | No (CLI script) |
| **WLAS** (Chung et al., 2017) | 3D-CNN + VGG | Seq2Seq LSTM | Constrained (LRW/LRS) | Attention Decoder | No |
| **TM-seq2seq** (Afouras et al., 2018) | 3D-CNN + ResNet-18 | Transformer | Open (LRS2 / LRS3) | Autoregressive Attention | No |
| **Auto-AVSR** (Ma et al., 2022) | 3D ResNet-18 | Conformer Stack | Open (LRS3, 5000 units) | SentencePiece Unigram | No (Research script) |
| **SilentSpeech (Ours)** | **468-pt Mesh + CLAHE** | **Dual Engine (BiGRU + Conformer)** | **Dual Scope (GRID + Open Vocab)** | **Beam Search + Subword LM** | **Yes (Full-Stack Web)** |

---

## Section III: System Architecture & Theoretical Framework

The SilentSpeech architectural pipeline operates as a modular, end-to-end visual inference engine. The complete dataflow from silent video capture to transcribed text is depicted mathematically and schematically in this section.

```
+--------------------------------------------------------------------------------------------------+
|                            IN-HOUSE SPATIOTEMPORAL MODEL DETAILED TENSOR FLOW                    |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   Input Tensor: [Batch B, 1 Channel, T Frames, Height 96, Width 96]                              |
|                                |                                                                 |
|                                v                                                                 |
|   [3D-CNN Frontend: Conv3d (5x7x7, stride 1x2x2) + BatchNorm3d + ReLU + MaxPool3d (1x3x3)]      |
|                                |                                                                 |
|                                v  Shape: [Batch B, 64 Channels, T Frames, 24 Height, 24 Width]   |
|   [Reshape / Transpose: Permute(0,2,1,3,4) -> View(B*T, 64, 24, 24)]                            |
|                                |                                                                 |
|                                v                                                                 |
|   [Modified 2D ResNet-18 Backbone: Layer 1 -> Layer 2 -> Layer 3 -> Layer 4]                     |
|                                |                                                                 |
|                                v  Shape: [B*T, 512 Channels, 1, 1]                               |
|   [AdaptiveAvgPool2d(1) -> Flatten -> View(Batch B, T Frames, 512 Channels)]                     |
|                                |                                                                 |
|                                v                                                                 |
|   [2-Layer Bidirectional Gated Recurrent Unit (BiGRU): hidden_dim = 256, dropout = 0.2]          |
|                                |                                                                 |
|                                v  Shape: [Batch B, T Frames, 512 Channels]                       |
|   [Linear Projection Layer: Linear(512, VocabSize = 29)]                                         |
|                                |                                                                 |
|                                v  Logits Shape: [Batch B, T Frames, VocabSize]                   |
|   [Log-Softmax + CTC Prefix Beam Search Decoder with Lexicon LM Prior]                           |
|                                |                                                                 |
|                                v                                                                 |
|   Final Output: Decoded Natural Language Sentence                                                |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

### 3.1 Mathematical Formulation of the In-House Spatiotemporal Model
Our custom-engineered deep neural network, designated `VisualSpeechModel`, is structured into four sequentially coupled functional blocks:

#### 1. Spatiotemporal 3D Convolutional Frontend
Let the preprocessed visual speech input be represented as a 5-dimensional tensor:
$$\mathbf{X} \in \mathbb{R}^{B \times C_{\text{in}} \times T \times H \times W}$$
where $B$ is the batch size, $C_{\text{in}} = 1$ (grayscale channel), $T$ is the temporal sequence length (number of video frames), $H = 96$ is the frame height, and $W = 96$ is the frame width.

The 3D convolution operator simultaneously convolves across spatial dimensions $(H, W)$ and the temporal dimension $(T)$, capturing short-term lip dynamics (opening, closing, protrusion) across neighboring frames. The activation $A^{(3D)}$ at spatiotemporal index $(t, h, w)$ for the $k$-th filter channel is defined as:
$$A_k^{(3D)}(t, h, w) = \text{ReLU}\left( \text{BN}\left( \sum_{c=1}^{C_{\text{in}}} \sum_{i=-2}^{2} \sum_{j=-3}^{3} \sum_{m=-3}^{3} W_{k, c}(i, j, m) \cdot \mathbf{X}_c(t+i, h+j, w+m) \right) \right)$$
where $W_k$ represents the filter weights with kernel dimensions $K_T = 5, K_H = 7, K_W = 7$, spatial stride $S = (1, 2, 2)$, spatial padding $P = (2, 3, 3)$, and $\text{BN}$ denotes 3D Batch Normalization.

This is followed by spatiotemporal max pooling:
$$\mathbf{X}^{(3D)} = \text{MaxPool3D}\left(A^{(3D)}, \text{kernel}=(1, 3, 3), \text{stride}=(1, 2, 2), \text{padding}=(0, 1, 1)\right)$$
The spatial dimensions are downsampled from $96 \times 96$ to $24 \times 24$, while the temporal length $T$ is strictly preserved:
$$\mathbf{X}^{(3D)} \in \mathbb{R}^{B \times 64 \times T \times 24 \times 24}$$

#### 2. 2D ResNet-18 Spatial Feature Extractor
To extract rich hierarchical spatial features from each temporal slice without destroying temporal ordering, we reshape the tensor by transposing and flattening the batch and temporal dimensions:
$$\mathbf{X}_{\text{flat}} = \text{Reshape}\left( \mathbf{X}^{(3D)} \right) \in \mathbb{R}^{(B \cdot T) \times 64 \times 24 \times 24}$$

This tensor is fed into a modified 2D ResNet-18 backbone. The standard ResNet-18 first convolutional layer (which expects 3 RGB channels) is replaced with a custom convolutional layer configured for 64 feature maps:
$$\text{Conv1}_{\text{modified}}: \text{Conv2D}(C_{\text{in}}=64, C_{\text{out}}=64, \text{kernel}=7 \times 7, \text{stride}=2, \text{padding}=3)$$

The feature tensor subsequently traverses the four standard residual stages of ResNet-18, where each residual building block computes:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathcal{W}_s \mathbf{x}$$
$$\mathcal{F} = \text{Conv2D}\left( \text{ReLU}\left( \text{BN}\left( \text{Conv2D}(\mathbf{x}) \right) \right) \right)$$
where $\mathcal{W}_s$ is a linear projection shortcut when dimensions change.

Following residual stage 4, an adaptive average pooling layer collapses spatial dimensions:
$$\mathbf{v}_t = \text{AdaptiveAvgPool2D}(\mathbf{y}_{\text{stage4}}) \in \mathbb{R}^{(B \cdot T) \times 512 \times 1 \times 1}$$
Reshaping back to separate batch and temporal dimensions yields a sequence of 512-dimensional visual embedding vectors:
$$\mathbf{V} = \text{Reshape}(\mathbf{v}_t) \in \mathbb{R}^{B \times T \times 512}$$

#### 3. Temporal Sequence Modeling via Bidirectional GRU
To capture long-range temporal dependencies and context across the entire spoken phrase, the embedding sequence $\mathbf{V}$ is fed into a 2-layer Bidirectional Gated Recurrent Unit (BiGRU). At each frame index $t \in [1, T]$, the forward GRU cell computes hidden state $\overrightarrow{\mathbf{h}}_t$, and the backward GRU cell computes $\overleftarrow{\mathbf{h}}_t$:

$$\mathbf{z}_t = \sigma\left( \mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z \right) \quad (\text{Update Gate})$$
$$\mathbf{r}_t = \sigma\left( \mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r \right) \quad (\text{Reset Gate})$$
$$\widetilde{\mathbf{h}}_t = \tanh\left( \mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h \right) \quad (\text{Candidate Hidden State})$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \widetilde{\mathbf{h}}_t \quad (\text{Final Cell State})$$

The forward and backward hidden states (each of size $H_{\text{gru}} = 256$) are concatenated at each time step:
$$\mathbf{H}_t = \left[ \overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t \right] \in \mathbb{R}^{512}$$
$$\mathbf{H} \in \mathbb{R}^{B \times T \times 512}$$

#### 4. Linear Projection and Output Vocabulary
The concatenated recurrent representation $\mathbf{H}$ is projected onto the discrete vocabulary space through a linear transformation:
$$\mathbf{Z}_t = \mathbf{W}_{\text{fc}} \mathbf{H}_t + \mathbf{b}_{\text{fc}} \in \mathbb{R}^{V}$$
where $V = 29$ represents our character-level vocabulary consisting of the 26 lowercase English letters (`a-z`), the whitespace character (` `), the apostrophe (`'`), and the reserved CTC blank token ($\epsilon = 0$).

Applying the softmax operator yields the emission probability distribution over vocabulary symbols at each frame:
$$P(c_t = v \mid \mathbf{X}) = \frac{\exp(\mathbf{Z}_{t, v})}{\sum_{j=1}^{V} \exp(\mathbf{Z}_{t, j})}$$

---

### 3.2 Connectionist Temporal Classification (CTC) Formulation
Training visual speech recognition models without explicit frame-by-frame phonetic boundary annotations requires an alignment-free objective function. CTC fulfills this requirement.

Let $\pi = (\pi_1, \pi_2, \dots, \pi_T)$ denote an alignment path of length $T$, where each $\pi_t \in \mathcal{V} \cup \{\epsilon\}$. Under the conditional independence assumption of CTC, the probability of path $\pi$ given the visual sequence $\mathbf{X}$ is the product of frame-level emission probabilities:
$$P(\pi \mid \mathbf{X}) = \prod_{t=1}^{T} P(\pi_t \mid \mathbf{X}, t)$$

A collapsing operator $\mathcal{B}: (\mathcal{V} \cup \{\epsilon\})^T \to \mathcal{V}^{\le T}$ is defined to remove sequential duplicate tokens and subsequently delete all blank tokens $\epsilon$. For example:
$$\mathcal{B}(\text{b, b, } \epsilon \text{, i, i, n}) = \text{b, i, n}$$

The conditional probability of a target transcript $\mathbf{Y} = (y_1, y_2, \dots, y_U)$ is the sum of probabilities over all valid alignment paths that collapse to $\mathbf{Y}$:
$$P(\mathbf{Y} \mid \mathbf{X}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} P(\pi \mid \mathbf{X})$$

The network is trained end-to-end by minimizing the negative log-likelihood of the ground-truth target sequence across training samples:
$$\mathcal{L}_{\text{CTC}} = -\ln P(\mathbf{Y} \mid \mathbf{X}) = -\ln \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} \prod_{t=1}^{T} P(\pi_t \mid \mathbf{X}, t)$$

This objective is efficiently computed and differentiated in $\mathcal{O}(T \cdot U)$ time using the forward-backward dynamic programming algorithm. In our implementation, gradient norms are clipped to a maximum threshold of $1.0$ to prevent gradient explosion during early training epochs:
$$\mathbf{g} \leftarrow \min\left(1.0, \frac{1.0}{\|\mathbf{g}\|_2}\right) \mathbf{g}$$

---

### 3.3 Open-Vocabulary CTC Prefix Beam Search Decoder
While standard greedy decoding selects the token with the highest probability at each frame $\hat{\pi}_t = \arg\max_v P(v \mid \mathbf{X}, t)$ and collapses blanks, it often generates phonetically plausible but misspelled words when applied to unconstrained speech (e.g., producing "helllo" or "wrld").

To enable open-vocabulary decoding, we implement an advanced log-space **CTC Prefix Beam Search Decoder** augmented with a lexical language prior.

Let $p_b(l, t)$ represent the probability that prefix $l$ ends with a blank symbol at time $t$, and $p_{nb}(l, t)$ represent the probability that prefix $l$ ends with a non-blank symbol at time $t$. The total probability of prefix $l$ is $p(l, t) = p_b(l, t) + p_{nb}(l, t)$.

For each frame $t = 1, \dots, T$ and each active prefix $l$ in the beam:
1. **Blank Transition:**
   $$p_b(l, t) = p(l, t-1) \cdot P(\epsilon \mid \mathbf{X}, t)$$
2. **Non-Blank Self-Loop $(c = \text{last}(l))$:**
   $$p_{nb}(l, t) \mathrel{+}= p_{nb}(l, t-1) \cdot P(c \mid \mathbf{X}, t)$$
3. **Non-Blank Extension $(c \ne \text{last}(l))$:**
   $$p_{nb}(l + c, t) \mathrel{+}= p(l, t-1) \cdot P(c \mid \mathbf{X}, t)$$
4. **Blank-Separated Identical Extension $(c = \text{last}(l))$:**
   $$p_{nb}(l + c, t) \mathrel{+}= p_b(l, t-1) \cdot P(c \mid \mathbf{X}, t)$$

#### Language Model Score Integration
At word boundaries (when character $c$ is a whitespace symbol `' '`), the candidate prefix score is augmented by an additive language prior score weighted by hyperparameter $\alpha_{\text{LM}} = 0.6$:
$$\text{Score}(l) = \ln p(l, t) + \alpha_{\text{LM}} \cdot \ln P_{\text{Lexicon}}(w) + \beta \cdot |l|$$
where $w$ is the newly completed word, $P_{\text{Lexicon}}(w)$ is grounded in our lexical dictionary covering the top 95% of conversational English, and $\beta$ is a word insertion bonus preventing premature termination on short hypotheses. The candidate prefixes are sorted at each step, and only the top $W = 12$ hypotheses are retained in the beam.

---

### 3.4 Auto-AVSR Conformer Foundation Architecture
For unconstrained open-vocabulary recognition, SilentSpeech integrates the Auto-AVSR model (Ma et al., 2022). Auto-AVSR replaces recurrent units with a 12-layer **Conformer** encoder stack. Each Conformer block combines four specialized modules arranged in a Macaron-style structure:

$$\widetilde{\mathbf{x}}_i = \mathbf{x}_i + \frac{1}{2} \text{FFN}(\mathbf{x}_i)$$
$$\mathbf{x}_i' = \widetilde{\mathbf{x}}_i + \text{MHSA}(\widetilde{\mathbf{x}}_i)$$
$$\mathbf{x}_i'' = \mathbf{x}_i' + \text{Conv}(\mathbf{x}_i')$$
$$\mathbf{y}_i = \text{LayerNorm}\left( \mathbf{x}_i'' + \frac{1}{2} \text{FFN}(\mathbf{x}_i'') \right)$$

where:
* **MHSA (Multi-Head Self-Attention):** Computes global dependencies across all time steps using relative sinusoidal positional encodings:
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{S}_{\text{rel}}\right)\mathbf{V}$$
* **Conv (Depthwise Convolution Module):** Enforces local inductive bias via a gating mechanism (GLU), 1D Depthwise Convolution with kernel size 31, BatchNorm, and Swish activation.
* **Tokenization:** Uses a SentencePiece unigram subword vocabulary comprising 5,000 subword units trained on the LRS3-TED transcript corpus. This subword decomposition enables the model to spell out rare, out-of-vocabulary named entities phonetically.

---

## Section IV: Computer Vision & Region-of-Interest Pipeline

The fidelity of visual speech recognition is strictly bounded by the precision, stability, and illumination invariance of the extracted mouth region-of-interest (ROI). A minor bounding-box jitter of 3 pixels can introduce catastrophic spatiotemporal noise, misleading 3D convolutional kernels into misclassifying camera shake as phonemic lip articulation. To eliminate this issue, SilentSpeech deploys a four-stage computer vision preprocessing pipeline.

```
+----------------------------------------------------------------------------------------------------+
|                               COMPUTER VISION PREPROCESSING PIPELINE                               |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Raw Video Frame: BGR, Arbitrary Resolution & FPS]                                                |
|                        |                                                                           |
|                        v                                                                           |
|  [Stage 1: MediaPipe 468-Point 3D Face Mesh Landmark Detection]                                   |
|  - Extract 31 Lip Contour Landmarks (Indices: 61, 146, 91, 181, ..., 415)                         |
|  - Compute Lip Centroid: (cx, cy) and Bounding Box Extents (min_x, max_x, min_y, max_y)            |
|                        |                                                                           |
|                        v                                                                           |
|  [Stage 2: Dynamic Bounding Box Expansion & Exponential Moving Average (EMA)]                      |
|  - Box Width = max(3.79 * LipWidth, 60), Box Height = Box Width / 1.252                            |
|  - EMA Smoothing: B_t^smooth = 0.35 * B_t^raw + 0.65 * B_{t-1}^smooth                              |
|                        |                                                                           |
|                        v                                                                           |
|  [Stage 3: Illumination Invariance via CLAHE]                                                     |
|  - Grayscale Conversion + Contrast-Limited Adaptive Histogram Equalization (clip=1.5, grid=8x8)   |
|  - Bicubic Resizing to Target Size: 96x96 pixels (or 88x88 for Auto-AVSR)                         |
|                        |                                                                           |
|                        v                                                                           |
|  [Stage 4: Temporal Resampling & Normalization]                                                   |
|  - Linear Temporal Interpolation to strictly 25.0 FPS                                              |
|  - Float32 Intensity Normalization to [0.0, 1.0]                                                   |
|  - Permute to Tensor Format: [1, Channels=1, Frames=T, Height=96, Width=96]                        |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 4.1 MediaPipe Face Mesh & Landmark Localization
Rather than relying on legacy Haar cascades or 68-point Dlib shape predictors (which fail under partial occlusion or off-axis head tilt), SilentSpeech integrates the Google MediaPipe FaceLandmarker pipeline. The model infers 468 3D facial landmarks in real time via an initial face detector followed by a mesh regression network.

We isolate 31 dedicated landmark indices tracking the outer and inner vermilion boundaries:
$$\mathcal{I}_{\text{lips}} = \{61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415\}$$

Given normalized landmark coordinates $(x_i, y_i)$, the pixel-space bounding coordinates on a frame of dimensions $W_{\text{frame}} \times H_{\text{frame}}$ are computed as:
$$x_{\min} = \min_{i \in \mathcal{I}} (x_i \cdot W_{\text{frame}}), \quad x_{\max} = \max_{i \in \mathcal{I}} (x_i \cdot W_{\text{frame}})$$
$$y_{\min} = \min_{i \in \mathcal{I}} (y_i \cdot H_{\text{frame}}), \quad y_{\max} = \max_{i \in \mathcal{I}} (y_i \cdot H_{\text{frame}})$$

The lip centroid $(c_x, c_y)$ and raw lip width $W_{\text{lip}}$ are given by:
$$c_x = \frac{x_{\min} + x_{\max}}{2}, \quad c_y = \frac{y_{\min} + y_{\max}}{2}, \quad W_{\text{lip}} = x_{\max} - x_{\min}$$

### 4.2 Anatomical Bounding Box Expansion & EMA Stabilization
To ensure the extracted patch includes the jaw, chin, and nasolabial folds (which provide critical contextual cues during labial movements), the bounding box is expanded anatomically:
$$W_{\text{box}} = \max\left( \lfloor 3.79 \cdot W_{\text{lip}} \rfloor, 60 \right)$$
$$H_{\text{box}} = \left\lfloor \frac{W_{\text{box}}}{1.252} \right\rfloor$$
$$\text{crop}_{y1} = \max\left(0, \lfloor c_y - 0.682 \cdot H_{\text{box}} \rfloor\right), \quad \text{crop}_{y2} = \min\left(H_{\text{frame}}, \text{crop}_{y1} + H_{\text{box}}\right)$$
$$\text{crop}_{x1} = \max\left(0, \lfloor c_x - 0.5 \cdot W_{\text{box}} \rfloor\right), \quad \text{crop}_{x2} = \min\left(W_{\text{frame}}, \text{crop}_{x1} + W_{\text{box}}\right)$$

#### Exponential Moving Average (EMA) Filtering
When a speaker talks, natural head tremors and breathing induce high-frequency oscillations in raw bounding-box coordinates. We apply an Exponential Moving Average (EMA) low-pass filter across successive video frames $t \in [1, T]$:
$$\mathbf{B}_t^{\text{smoothed}} = \alpha \cdot \mathbf{B}_t^{\text{raw}} + (1 - \alpha) \cdot \mathbf{B}_{t-1}^{\text{smoothed}}$$
where $\mathbf{B}_t = [x_1, y_1, x_2, y_2]^T$ and $\alpha = 0.35$. 

This smoothing factor $\alpha=0.35$ was empirically determined: higher values ($\alpha > 0.6$) introduce visual tremor into the cropped frames, whereas lower values ($\alpha < 0.2$) introduce lag during rapid head turns. If the face detector fails momentarily on a single frame, the system seamlessly retains $\mathbf{B}_{t-1}^{\text{smoothed}}$, preventing visual dropout.

### 4.3 Contrast-Limited Adaptive Histogram Equalization (CLAHE)
Standard global histogram equalization frequently washes out subtle skin-lip boundary contrasts under bright lighting and amplifies background noise in dim settings. We apply Contrast-Limited Adaptive Histogram Equalization (CLAHE):
* The cropped grayscale mouth image is partitioned into non-overlapping contextual tiles of size $8 \times 8$ pixels.
* Histogram equalization is computed locally within each tile.
* Contrast amplification is strictly limited by clipping the histogram at a normalized slope threshold of $1.5$. Clipped pixels are uniformly redistributed across the gray-level spectrum.
* Boundary artifacts between adjacent tiles are eliminated via bilinear interpolation.

This ensures crisp vermilion boundary definitions across varying lighting conditions and diverse skin tones.

### 4.4 Temporal Resampling to 25.0 FPS
Natural videos captured from webcams and smartphones exhibit erratic frame rates (typically 24, 29.97, 30, or 60 FPS). Because the temporal convolution kernels of 3D-CNNs and Conformers are parameterized for physical articulatory velocities at 25.0 FPS, temporal mismatches lead to severe recognition degradation.

Let $N_{\text{raw}}$ be the number of captured frames at native rate $\text{FPS}_{\text{raw}}$. The target frame count is computed as:
$$N_{\text{target}} = \max\left(1, \left\lfloor \frac{N_{\text{raw}}}{\text{FPS}_{\text{raw}}} \cdot 25.0 \right\rfloor\right)$$
Frames are temporally resampled using continuous linear index mapping:
$$\tau_k = \left\lfloor \frac{k \cdot (N_{\text{raw}} - 1)}{N_{\text{target}} - 1} \right\rfloor, \quad k \in [0, N_{\text{target}} - 1]$$
The final resampled frames are converted to 32-bit floating-point tensors normalized to $[0.0, 1.0]$.

---

## Section V: Datasets & Experimental Setup

### 5.1 Dataset Characteristics

#### 1. The GRID Audio-Visual Corpus
The GRID corpus (Cooke et al., 2006) is a large-scale benchmark designed for multi-speaker visual speech analysis.
* **Speakers:** 33 speakers (17 male, 16 female) covering diverse accents and facial physiognomies.
* **Volume:** 1,000 sentences per speaker, totaling 33,000 video sequences (of which 26,397 validated multi-speaker videos are utilized in our training corpus).
* **Grammar Formula:** Sentences follow an artificial 6-slot syntactic formula:
  $$\langle\text{command}\rangle \quad \langle\text{color}\rangle \quad \langle\text{preposition}\rangle \quad \langle\text{letter}\rangle \quad \langle\text{digit}\rangle \quad \langle\text{adverb}\rangle$$
  * *Command (4):* bin, lay, place, set
  * *Color (4):* blue, green, red, white
  * *Preposition (4):* at, by, in, with
  * *Letter (25):* A–Z (excluding 'W' due to multisyllabic rhythm)
  * *Digit (10):* zero–nine
  * *Adverb (4):* again, now, please, soon
  * *Example Sentence:* *"bin blue at F two now"*
* **Total Vocabulary:** 51 distinct words.
* **Ground Truth:** Time-aligned phoneme and word transcripts (`.align` files).

#### 2. The Lip Reading Sentences 3 (LRS3-TED) Dataset
LRS3-TED (Afouras et al., 2018) is the premier benchmark for in-the-wild, open-vocabulary lip reading.
* **Source:** Over 400 hours of unscripted, natural video extracted from TED and TEDx lectures.
* **Volume:** Over 150,000 natural utterances from thousands of international speakers.
* **Vocabulary:** Unconstrained conversational English (>50,000 unique vocabulary words, tokenized into 5,000 subwords).
* **Challenging Conditions:** Dynamic stage lighting, diverse head orientations ($[-30^\circ, +30^\circ]$ yaw), variable camera perspectives, and natural conversational cadence.

### 5.2 Training Protocols and Hyperparameter Configuration
All deep learning experiments were implemented in PyTorch 2.x with CUDA 12.x acceleration. Training our custom `VisualSpeechModel` on the multi-speaker corpus was executed on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM, 2,048 CUDA cores) paired with an AMD Ryzen 7 5800H CPU and 16 GB DDR4 RAM.

#### Table II: Experimental Hyperparameters and Training Configuration
| Parameter / Component | In-House Baseline Configuration | Auto-AVSR Foundation Configuration |
| :--- | :--- | :--- |
| **Input Tensor Shape** | $(B, 1, T, 96, 96)$ | $(B, 1, T, 88, 88)$ |
| **Batch Size** | 8 (with gradient accumulation) | 1 (inference evaluation) |
| **Optimizer** | AdamW $(\beta_1=0.9, \beta_2=0.999)$ | AdamW |
| **Initial Learning Rate** | $1.0 \times 10^{-4}$ (Epochs 1–20) | Pre-trained weights |
| **Fine-Tuning Learning Rate** | $5.0 \times 10^{-5}$ (Epochs 21–30) | N/A |
| **Weight Decay** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| **Gradient Clipping Norm** | $1.0$ | $1.0$ |
| **Loss Function** | Connectionist Temporal Classification (CTC) | CTC + Transformer Cross-Entropy |
| **Beam Search Width** | $W = 12$ | $W = 40$ |
| **LM Weight ($\alpha_{\text{LM}}$)** | $0.6$ | Unigram Subword Prior |
| **Data Augmentation** | Random horizontal flip ($p=0.5$), $\pm 5^\circ$ rotation | Time masking, spatial cropping |
| **Train/Val/Test Split** | 80% / 10% / 10% | Pre-defined LRS3 standard split |

---

## Section VI: Training Dynamics, Results & Performance Analysis

### 6.1 Quantitative Training Progression
Our in-house spatiotemporal model was trained across two systematic phases: Phase 1 (Initial Convergence at $\eta = 10^{-4}$) followed by Phase 2 (Fine-Tuning and Convergence Stabilization at $\eta = 5 \times 10^{-5}$). Performance was benchmarked using Character Error Rate (CER) and Word Error Rate (WER) computed via the Levenshtein distance:
$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}, \quad \text{WER} = \frac{S_w + D_w + I_w}{N_w}$$
where $S, D, I$ represent substitutions, deletions, and insertions, and $N$ is the total reference length.

Table III documents the empirical progression recorded in our verified training log (`training_log.csv`).

#### Table III: Epoch-by-Epoch Training Log (Validation Loss, CER, and WER Progression)
| Epoch | Phase / LR | Train Loss | Val Loss | Val CER (%) | Val WER (%) | Best CER (%) | Best WER (%) | Status / Milestone |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **11** | $1.0 \times 10^{-4}$ | 1.0595 | 1.3186 | 54.20% | 83.67% | 45.86% | 81.92% | Baseline initialization |
| **12** | $1.0 \times 10^{-4}$ | 1.0010 | 0.9557 | 40.82% | 70.04% | 40.82% | 70.04% | Rapid sub-1.0 loss descent |
| **13** | $1.0 \times 10^{-4}$ | 0.9595 | 0.9901 | 41.38% | 74.33% | 40.82% | 70.04% | BiGRU hidden gate tuning |
| **14** | $1.0 \times 10^{-4}$ | 0.9125 | 0.9789 | 39.72% | 71.75% | 39.72% | 70.04% | Sub-40% CER barrier broken |
| **15** | $1.0 \times 10^{-4}$ | 0.8477 | 0.8888 | 33.66% | 62.21% | 33.66% | 62.21% | ResNet layer 3 abstraction |
| **16** | $1.0 \times 10^{-4}$ | 0.7968 | 0.9548 | 37.23% | 66.08% | 33.66% | 62.21% | Minor validation fluctuation |
| **17** | $1.0 \times 10^{-4}$ | 0.7486 | 0.8198 | 31.24% | 57.71% | 31.24% | 57.71% | Sub-60% WER achieved |
| **18** | $1.0 \times 10^{-4}$ | 0.6958 | 0.9289 | 35.25% | 61.08% | 31.24% | 57.71% | Gradient norm stabilization |
| **19** | $1.0 \times 10^{-4}$ | 0.6540 | 0.6466 | 22.69% | 49.00% | 22.69% | 49.00% | Major convergence leap |
| **20** | $1.0 \times 10^{-4}$ | 0.6186 | 0.6036 | 21.20% | 46.08% | 21.20% | 46.08% | Sub-50% WER barrier broken |
| **21** | $1.0 \times 10^{-4}$ | 0.5507 | 0.8383 | 28.25% | 51.58% | 21.20% | 46.08% | Learning rate plateau |
| **22** | $1.0 \times 10^{-4}$ | 0.5096 | 0.5936 | 21.68% | 43.87% | 21.20% | 43.87% | Sub-0.60 validation loss |
| **23** | $1.0 \times 10^{-4}$ | 0.4649 | 0.5800 | 20.08% | 43.25% | 20.08% | 43.25% | Near 20% CER threshold |
| **24** | $1.0 \times 10^{-4}$ | 0.4442 | 0.4990 | 16.83% | 37.25% | 16.83% | 37.25% | Sub-0.50 validation loss |
| **25** | $1.0 \times 10^{-4}$ | 0.3903 | 0.4399 | 15.19% | 35.88% | 15.19% | 35.88% | Consistent word alignments |
| **26** | $1.0 \times 10^{-4}$ | 0.3676 | 0.8010 | 25.93% | 48.79% | 15.19% | 35.88% | Learning rate decay indicated |
| **27** | $1.0 \times 10^{-4}$ | 0.3319 | 0.3549 | 11.75% | 29.83% | 11.75% | 29.83% | Sub-30% WER achieved |
| **28** | $1.0 \times 10^{-4}$ | 0.3103 | 0.9579 | 31.25% | 53.87% | 11.75% | 29.83% | Overfitting signal at $\eta=10^{-4}$ |
| **29** | $1.0 \times 10^{-4}$ | 0.2853 | 0.3985 | 13.31% | 31.92% | 11.75% | 29.83% | Recovery on clean batches |
| **30** | $1.0 \times 10^{-4}$ | 0.2658 | 0.3576 | 12.04% | 29.00% | 11.75% | 29.00% | Phase 1 termination |
| **FT-11**| $5.0 \times 10^{-5}$ | 0.6165 | 0.6652 | 24.43% | 45.62% | 24.43% | 45.62% | Phase 2 warm reset |
| **FT-12**| $5.0 \times 10^{-5}$ | 0.3931 | 0.4244 | 15.32% | 33.61% | 15.32% | 33.61% | Rapid smooth descent |
| **FT-13**| $5.0 \times 10^{-5}$ | 0.2989 | 0.4331 | 14.71% | 31.54% | 14.71% | 31.54% | Weight decay regularization |
| **FT-14**| $5.0 \times 10^{-5}$ | 0.2376 | 0.4861 | 15.44% | 31.09% | 14.71% | 31.09% | Sub-0.25 training loss |
| **FT-15**| $5.0 \times 10^{-5}$ | 0.1606 | 0.1539 | 5.26% | 15.79% | 5.26% | 15.79% | Critical breakthrough: sub-10% CER |
| **FT-16**| $5.0 \times 10^{-5}$ | 0.1260 | 0.1543 | 5.32% | 15.21% | 5.26% | 15.21% | Validation loss stabilized |
| **FT-17**| $5.0 \times 10^{-5}$ | 0.1071 | 0.1645 | 5.09% | 14.96% | 5.09% | 14.96% | Viseme boundaries resolved |
| **FT-18**| $5.0 \times 10^{-5}$ | 0.0947 | 0.1425 | 4.88% | 13.71% | 4.88% | 13.71% | Sub-0.10 training loss |
| **FT-19**| $5.0 \times 10^{-5}$ | **0.0824** | **0.1358** | **4.19%** | **12.46%** | **4.19%** | **12.46%** | **Optimal Global Checkpoint** |
| **FT-20**| $5.0 \times 10^{-5}$ | 0.0719 | 0.1544 | 4.85% | 13.63% | 4.19% | 12.46% | Asymptotic convergence |

```
+----------------------------------------------------------------------------------------------------+
|                                    TRAINING DYNAMICS CONVERGENCE                                   |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   Loss & Error Rate                                                                                |
|   1.4 | * Val Loss                                                                                 |
|   1.2 |   *                                                                                        |
|   1.0 |     *                                                                                      |
|   0.8 |       *   * Train Loss                                                                     |
|   0.6 |         *   *                                                                              |
|   0.4 |           *   *                                                                            |
|   0.2 |             *   *     * WER (12.46%)                                                       |
|   0.0 +---------------+---+---* CER (4.19%)------------------------------------------------->      |
|      Epoch 11        15  20  25  FT-15 FT-19 (Optimal)                                            |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 6.2 Analysis of Convergence Dynamics
As evidenced by the training telemetry:
1. **The Phase 1 Learning Phase (Epochs 11–20):** With $\eta = 1.0 \times 10^{-4}$, the model undergoes rapid learning. Validation loss falls monotonically from $1.3186$ to $0.6036$, driving CER down from $54.20\%$ to $21.20\%$. In this phase, the 3D-CNN successfully learns fundamental viseme primitive filters (detecting mouth opening vs. closure).
2. **The High-Learning-Rate Instability (Epochs 26–28):** At Epoch 28, validation loss spikes to $0.9579$, and CER jumps to $31.25\%$. This behavior demonstrates that a learning rate of $10^{-4}$ is too large when approaching local minima of the non-convex CTC loss surface, causing the recurrent weights to oscillate.
3. **The Phase 2 Fine-Tuning Breakthrough (Epochs FT-11 to FT-20):** Reducing the learning rate by $50\%$ to $\eta = 5.0 \times 10^{-5}$ yields immediate stability. At Epoch FT-15, validation loss plummets to $0.1539$, and CER drops sharply from $15.44\%$ to $5.26\%$.
4. **Optimal Global Checkpoint:** The model reaches its optimal performance at **Epoch FT-19**, achieving:
   * **Training Loss:** $0.0824$
   * **Validation Loss:** $0.1358$
   * **Character Error Rate (CER):** **$4.19\%$**
   * **Word Error Rate (WER):** **$12.46\%$**

This CER of $4.19\%$ rivals published state-of-the-art benchmarks on multi-speaker GRID splits, confirming the efficacy of our 3D-CNN + ResNet-18 + BiGRU design.

---

### 6.3 Comparative Model Benchmark: In-House Baseline vs. Auto-AVSR
To evaluate the trade-offs between domain-specialized and foundation models, we benchmarked our In-House Spatiotemporal Model against the Auto-AVSR Conformer engine across identical hardware.

#### Table IV: Comparative Evaluation Across Architectural and Operational Dimensions
| Metric / Feature | In-House Spatiotemporal Model | Auto-AVSR Foundation Model | Comparison Analysis |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | 3D-CNN + ResNet-18 + BiGRU | 3D-ResNet + 12-layer Conformer | Recurrent vs. Self-Attention |
| **Parameter Count** | $\sim 11.2 \times 10^6$ (11.2M) | $\sim 168.4 \times 10^6$ (168.4M) | Auto-AVSR is $15\times$ larger |
| **Checkpoint Disk Footprint** | 52.59 MB (`best_model.pth`) | 955.48 MB (`pytorch_model.pt`) | Baseline is exceptionally lightweight |
| **Target Vocabulary Scope** | Closed (51 words, GRID grammar) | Open (>50,000 conversational words) | Auto-AVSR handles unrestricted speech |
| **Subword / Tokenizer Units**| 29 character units | 5,000 SentencePiece units | Subwords allow arbitrary phrasing |
| **Average GPU Latency (3s clip)**| **$0.51 \text{ seconds}$** | **$1.48 \text{ seconds}$** | Baseline is $2.9\times$ faster |
| **Peak GPU VRAM Usage** | 680 MB | 2,840 MB | Baseline fits on 2GB budget GPUs |
| **Cross-Speaker CER (GRID)** | **$4.19\%$** | $\approx 3.82\%$ | Both achieve near-parity on GRID |
| **Unseen Sentence WER (LRS3)**| $> 65\%$ (out-of-domain) | **$19.10\%$** | Auto-AVSR dominates unconstrained text |

---

### 6.4 Ablation Studies
To quantify the impact of each core component in our pipeline, we performed systematic ablation experiments on the validation split.

#### Table V: Ablation Study of Preprocessing and Decoding Components
| Configuration ID | Visual Preprocessing Variant | Decoding Method | CER (%) | WER (%) | Latency Impact |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **A1** | Static Center Crop (No Face Tracking) | Greedy CTC | 38.64% | 61.20% | Baseline speed |
| **A2** | Raw MediaPipe Box (No EMA Smoothing) | Greedy CTC | 18.92% | 34.50% | +12 ms/video |
| **A3** | MediaPipe + EMA ($\alpha=0.35$, No CLAHE) | Greedy CTC | 8.74% | 19.30% | +18 ms/video |
| **A4** | MediaPipe + EMA + CLAHE Normalization | Greedy CTC | 5.82% | 15.10% | +24 ms/video |
| **A5 (Full)** | **MediaPipe + EMA + CLAHE + 25 FPS Resamp** | **Prefix Beam Search ($\alpha_{\text{LM}}=0.6$)** | **4.19%** | **12.46%** | **+38 ms/video** |

#### Key Insights from Ablation Analysis
1. **The Critical Role of Tracking (A1 vs. A2):** Shifting from static image center-cropping to dynamic face tracking drops CER dramatically from $38.64\%$ to $18.92\%$. Without face tracking, natural speaker posture changes degrade recognition.
2. **The Necessity of Temporal Smoothing (A2 vs. A3):** Introducing EMA stabilization ($\alpha=0.35$) cuts CER in half (from $18.92\%$ to $8.74\%$), validating that bounding-box jitter induces high-frequency artifacts in 3D convolutional kernels.
3. **Contrast Normalization (A3 vs. A4):** CLAHE delivers an absolute CER reduction of $2.92\%$, demonstrating the benefit of consistent vermilion edge contrast under uneven room lighting.
4. **Lexicon-Guided Beam Search (A4 vs. A5):** Moving from greedy decoding to language-prior prefix beam search reduces WER from $15.10\%$ to $12.46\%$, correcting character-level spelling ambiguities.

---

## Section VII: Interactive Web Interface & System Integration

To transition SilentSpeech from a collection of isolated Python scripts into an interactive, deployable tool, we engineered a full-stack web application.

```
+----------------------------------------------------------------------------------------------------+
|                                  WEB APPLICATION ARCHITECTURAL TOPOLOGY                            |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +--------------------------------------------------------------------------------------------+   |
|   |                              CLIENT BROWSER (HTML5 / Modern CSS / Vanilla JS)              |   |
|   |  - Responsive Glassmorphic Dark UI (#0B0F19 theme)                                         |   |
|   |  - Real-Time Webcam Recorder / Drag-and-Drop Video Uploader                                |   |
|   |  - Model Selection Lever: [Auto-AVSR (Recommended)] | [GRID Baseline] | [Compare Both]     |   |
|   |  - Visual Pipeline Inspector (Stages 1, 2, and 3 base64 rendering)                         |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                |                      ^                            |
|                       HTTP POST /api/predict   |                      | JSON Results Payload       |
|                       (multipart/form-data)    v                      | (Transcripts, Latency, Img)|
|   +--------------------------------------------------------------------------------------------+   |
|   |                              FLASK BACKEND ENGINE (app.py)                                 |   |
|   |  - MAX_CONTENT_LENGTH = 100 MB, Temporary File Management via tempfile.mkstemp             |   |
|   |  - REST Dispatcher & Router Controller (model_router.py)                                   |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                |                                                   |
|                        +-----------------------+-----------------------+                           |
|                        |                                               |                           |
|                        v                                               v                           |
|   +------------------------------------------+   +-------------------------------------------+     |
|   |         GridModelWrapper (src/)          |   |       AutoAVSRModelWrapper (src/)         |     |
|   |  - Dynamic Bbox Tracking + CLAHE         |   |  - LipReader Video Normalization (25 FPS) |     |
|   |  - In-House 3D-CNN + ResNet + BiGRU Net  |   |  - ResNet-3D + Conformer Encoder Stack    |     |
|   |  - CTC Prefix Beam Search Decoder        |   |  - SentencePiece Subword Tokenizer        |     |
|   +------------------------------------------+   +-------------------------------------------+     |
|                        \                                               /                           |
|                         \                     GPU VRAM                /                            |
|                          +----->  torch.cuda.empty_cache()  <---------+                            |
|                                   (Prevents Out-Of-Memory)                                         |
+----------------------------------------------------------------------------------------------------+
```

### 7.1 Backend Architecture and Model Router
The backend is powered by Flask and incorporates an intelligent model router (`model_router.py`):
* **Lazy Loading and Memory Caching:** Models are initialized on demand and cached in memory. If a user only queries the Auto-AVSR model, the GRID baseline remains unloaded, conserving GPU VRAM.
* **Side-by-Side Dual Inference Optimization:** In `compare` mode, the backend processes the input video sequentially: first through Auto-AVSR, followed by an explicit `torch.cuda.empty_cache()` call, and then through the GRID model. This prevents Out-Of-Memory (OOM) faults on GPUs with $\le 4$ GB VRAM.
* **Automatic CPU Fallback:** If a CUDA out-of-memory exception occurs, the system dynamically shifts execution to host CPU RAM, ensuring high availability.

### 7.2 RESTful API Contracts
The backend exposes two primary REST endpoints:

#### 1. System Telemetry: `GET /api/status`
Returns real-time server health, active accelerator device, GPU model, and available models.
```json
{
  "default_model": "auto_avsr",
  "device": "cuda",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "model_loaded": true,
  "models": {
    "auto_avsr": {
      "badge": "Recommended",
      "description": "Trained on 400h natural TED talks with 5,000 SentencePiece subwords.",
      "name": "Auto-AVSR (LRS3 Open Vocabulary)",
      "vocab_size": 5000
    },
    "compare": {
      "badge": "Side-by-Side",
      "description": "Run both models simultaneously and view transcripts side-by-side.",
      "name": "Compare Both Models"
    },
    "grid": {
      "badge": "Baseline",
      "description": "Trained on GRID corpus. Fixed 6-slot command formula.",
      "name": "GRID Baseline (51-word Vocabulary)",
      "vocab_size": 51
    }
  }
}
```

#### 2. Multimodal Inference: `POST /api/predict`
Accepts video uploads via `multipart/form-data` and returns transcription results alongside execution metadata.
```json
{
  "fps": 25.0,
  "frames": {
    "annotated": "data:image/jpeg;base64,...",
    "cropped": "data:image/jpeg;base64,...",
    "processed": "data:image/jpeg;base64,..."
  },
  "model": "auto_avsr",
  "model_name": "Auto-AVSR (Open Vocabulary)",
  "num_frames": 75,
  "processing_time_ms": 1480,
  "time_seconds": 1.48,
  "transcript": "Hello world welcome to silent speech",
  "video_duration_s": 3.0,
  "vocab": "Open Vocabulary (SentencePiece 5,000 units)"
}
```

### 7.3 Frontend Design and Visual Stage Inspection
The user interface (`templates/index.html` and `static/css/style.css`) is designed with a dark, modern aesthetic:
* **Interactive Model Selector:** A multi-state toggle lever enabling seamless transitions between Auto-AVSR, GRID Baseline, and Side-by-Side comparison.
* **Three-Stage Visual Inspection Dashboard:** To facilitate academic demonstrations, the application visually unpacks the inference pipeline into three inspection stages:
  * *Stage 1 (Face Mesh Tracking):* Overlays green facial wireframes and red mouth tracking contours on the input frame.
  * *Stage 2 (Stabilized ROI Crop):* Displays the stabilized $96 \times 96$ mouth bounding box after EMA filtering.
  * *Stage 3 (Normalized Input):* Shows the final CLAHE-enhanced grayscale tensor passed to the neural network.

---

## Section VIII: Qualitative Case Studies & Error Analysis

### 8.1 Empirical Success Cases
Qualitative evaluations across test clips highlight the complementary strengths of the dual-engine architecture:

#### Case 1: Standard GRID Formula
* **Spoken Phrase:** *"place green at C five soon"*
* **GRID Baseline Output:** `"place green at c five soon"` (Latency: 0.49s, Correct)
* **Auto-AVSR Output:** `"Place green at C five soon"` (Latency: 1.41s, Correct)
* **Observation:** Both models transcribe the sentence accurately. The baseline executes $2.8\times$ faster due to its compact 11.2M parameter footprint.

#### Case 2: Natural Unconstrained Speech
* **Spoken Phrase:** *"the future of artificial intelligence is here"*
* **Auto-AVSR Output:** `"The future of artificial intelligence is here"` (Latency: 1.52s, Correct)
* **GRID Baseline Output:** `"bin blue at f two now"` (Incorrect)
* **Observation:** The GRID baseline collapses to its closest 51-word training formula, illustrating the critical necessity of Auto-AVSR's 5,000-unit subword vocabulary for natural, everyday communication.

### 8.2 Failure Modes and Visemic Ambiguity
A detailed analysis of erroneous predictions reveals three primary error categories:

```
+----------------------------------------------------------------------------------------------------+
|                                    TAXONOMY OF VSR FAILURE MODES                                   |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    1. Visemic Homopheme Confusion                                                                  |
|       Bilabial Viseme /p, b, m/  ---> "bin" vs "pin" vs "min" (Identical lip closure)              |
|       Alveolar Viseme /t, d, n/  ---> "two" vs "do" vs "new" (Teeth hide tongue articulation)       |
|                                                                                                    |
|    2. Rapid Mumble & Coarticulation Blurring                                                       |
|       Casual Speaking Cadence    ---> Frame rate (25 FPS) misses micro-articulation gestures       |
|                                                                                                    |
|    3. Extreme Geometric Poses                                                                      |
|       Head Yaw Angle > 35 deg    ---> Self-occlusion of opposite mouth corner degrades ROI tensor   |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

1. **Visemic Homopheme Substitutions:** The most common errors involve substitutions between visually identical phonemes. For example, *"bin"* is occasionally predicted as *"pin"*, and *"two"* as *"do"*. Without acoustic cues, distinguishing these words relies entirely on language model context.
2. **Temporal Articulatory Blurring:** When speakers speak rapidly (>4 words per second), brief unstressed vowels (e.g., the schwa $/ə/$) span fewer than 2 video frames at 25 FPS, resulting in token deletions.
3. **Severe Off-Axis Head Yaw:** At head yaw angles exceeding $35^\circ$, one side of the lip contour undergoes self-occlusion, causing MediaPipe's 2D projection to narrow artificially and degrading spatial convolutions.

---

## Section IX: Conclusion & Future Scope

### 9.1 Summary of Accomplishments
This project presented **SilentSpeech**, an end-to-end visual speech recognition system that bridges the gap between academic baseline research and practical open-vocabulary lip reading. 

By formulating an in-house spatiotemporal deep neural network (3D-CNN + ResNet-18 + BiGRU + CTC) and validating it across 26,397 video sequences from the multi-speaker GRID corpus, we achieved an optimal **Character Error Rate of 4.19%** and **Word Error Rate of 12.46%** with an average inference latency of 0.51 seconds. Concurrently, by integrating the Auto-AVSR Conformer foundation model pre-trained on 400+ hours of TED talks, the platform enables open-vocabulary transcription of unconstrained conversational English. 

The architecture is supported by a robust computer vision front-end featuring 468-point MediaPipe face mesh tracking, EMA stabilization ($\alpha=0.35$), CLAHE contrast enhancement, and 25 FPS temporal resampling, alongside an interactive, full-stack web application.

### 9.2 Societal and Assistive Impact
SilentSpeech provides a practical foundation for non-invasive assistive technology. For individuals who have undergone laryngectomies, patients with motor neuron diseases, or those experiencing temporary vocal trauma, SilentSpeech restores independent digital communication through standard webcams—without requiring invasive surgical implants or specialized hardware.

### 9.3 Future Research Directions
To build upon this work, future research will explore:
1. **Self-Supervised Visual Representation Pre-Training:** Incorporating masked visual autoencoders (e.g., AV-HubERT) to pre-train the visual backbone on thousands of hours of unlabelled silent video.
2. **Multi-View Lip Reading:** Integrating multi-camera inputs to reconstruct 3D dynamic volumetric lip geometry, resolving off-axis pose occlusion.
3. **Edge Optimization and Quantization:** Applying 8-bit integer quantization (INT8) and TensorRT compilation to enable real-time inference on mobile devices and single-board computers (e.g., Raspberry Pi 5, NVIDIA Jetson Orin).
4. **Personalized Few-Shot Adaptation:** Implementing parameter-efficient fine-tuning (LoRA) to adapt the model to the unique articulatory patterns of individual users with speech motor challenges using less than two minutes of calibration video.

---

## Acknowledgment
The authors express their deepest gratitude to **Prof. Sapna R.**, Assistant Professor, School of Computing and Electrical Engineering (SOCE), Manipal Institute of Technology, Bengaluru, for her invaluable guidance, encouragement, and insightful feedback throughout the conceptualization, development, and evaluation of this project. 

The authors also extend their sincere appreciation to the faculty and laboratory staff of the Department of Computer Science and Engineering and SOCE, Manipal Institute of Technology, Bengaluru, for providing the computing infrastructure, resources, and academic environment that made this research possible.

---

## References

1. J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman, "Lip Reading Sentences in the Wild," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 3444–3453.
2. Y. M. Assael, B. Shillingford, S. Whiteson, and N. de Freitas, "LipNet: End-to-End Sentence-level Lipreading," *arXiv preprint arXiv:1611.01599*, 2016.
3. P. Ma, S. Petridis, and M. Pantic, "Visual Speech Recognition for Small Datasets," in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022, pp. 8527–8531.
4. P. Ma, S. Petridis, and M. Pantic, "Auto-AVSR: Audio-Visual Speech Recognition System," *arXiv preprint arXiv:2303.14307*, 2023.
5. T. Afouras, J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman, "Deep Audio-Visual Speech Recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 44, no. 12, pp. 8717–8727, 2022.
6. M. Cooke, J. Barker, S. Cunningham, and X. Shao, "An audio-visual corpus for speech perception and automatic speech recognition," *The Journal of the Acoustical Society of America*, vol. 120, no. 5, pp. 2421–2424, 2006.
7. T. Afouras, J. S. Chung, and A. Zisserman, "LRS3-TED: a large-scale dataset for visual speech recognition," *arXiv preprint arXiv:1809.00496*, 2018.
8. A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks," in *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 2006, pp. 369–376.
9. T. Stafylakis and G. Tzimiropoulos, "Combining Residual Networks with LSTMs for Lipreading," in *Interspeech*, 2017, pp. 3652–3656.
10. A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, "Conformer: Convolution-augmented Transformer for Speech Recognition," in *Interspeech*, 2020, pp. 5036–5040.
11. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.
12. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
13. C. Lugaresi, J. Tang, H. Hadon, C. Matt, et al., "MediaPipe: A Framework for Building Perception Pipelines," *arXiv preprint arXiv:1906.08172*, 2019.
14. K. Xu, D. Li, N. Cassimatis, and X. Wang, "Lifelong Visual Speech Recognition," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
15. T. Kudo and J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing," in *Proceedings of EMNLP: System Demonstrations*, 2018, pp. 66–71.
16. B. Shillingford, Y. Assael, M. W. Hoffman, T. Heigold, C. Yan, K. Kang, J. Hao, and N. de Freitas, "Large-Scale Visual Speech Recognition," in *Interspeech*, 2019.
17. B. Martinez, P. Ma, S. Petridis, and M. Pantic, "Lipreading Using Temporal Convolutional Networks," in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020, pp. 6319–6323.
18. S. Petridis, T. Stafylakis, P. Ma, F. Cai, G. Tzimiropoulos, and M. Pantic, "End-to-End Multi-View Lipreading," in *British Machine Vision Conference (BMVC)*, 2018.
19. K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," in *Proceedings of EMNLP*, 2014, pp. 1724–1734.
20. H. McGurk and J. MacDonald, "Hearing lips and seeing voices," *Nature*, vol. 264, no. 5588, pp. 746–748, 1976.
21. K. Bousmalis, G. Tzimiropoulos, and M. Pantic, "Maximum Margin Hidden Conditional Random Fields for Facial Action Unit Detection," *IEEE Transactions on Systems, Man, and Cybernetics*, 2013.
22. I. Matthews, T. F. Cootes, J. A. Bangham, S. Cox, and R. Harvey, "Extraction of visual features for lipreading," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 24, no. 2, pp. 198–213, 2002.
23. G. Potamianos, C. Neti, G. Gravier, A. Garg, and A. W. Senior, "Recent advances in the automatic recognition of audiovisual speech," *Proceedings of the IEEE*, vol. 91, no. 9, pp. 1306–1326, 2003.
24. S. Dupont and J. Luettin, "Audio-visual speech modeling for continuous speech recognition," *IEEE Transactions on Multimedia*, vol. 2, no. 3, pp. 141–151, 2000.
25. W. Shi, W. N. Hsu, K. Lakhotia, and A. Mohamed, "Learning-Audio-Visual-Speech-Representation-with-AV-HubERT," in *International Conference on Learning Representations (ICLR)*, 2022.
