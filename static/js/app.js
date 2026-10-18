/**
 * SilentSpeech: Neural Lip Reading Studio Client
 * Zero-scroll front-frame architecture with modal popups,
 * live webcam recorder, instant ground truth diffing,
 * and multi-model dispatch.
 */

(function () {
    'use strict';

    // ---------------------------------------------------------------
    // 1. DOM References
    // ---------------------------------------------------------------
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // Navigation & Modals
    const btnOpenVisualModal      = $('#btn-open-visual-modal');
    const btnOpenGroundTruthModal = $('#btn-open-ground-truth-modal');
    const btnOpenInfoModal        = $('#btn-open-info-modal');
    const btnInspectRoiLink       = $('#btn-inspect-roi-link');
    const gtActiveIndicator       = $('#gt-active-indicator');
    const btnClearGt              = $('#btn-clear-gt');
    const themeToggleBtn          = $('#theme-toggle');

    // Video Input
    const uploadZone     = $('#upload-zone');
    const fileInput      = $('#file-input');
    const videoPreview   = $('#video-preview');
    const videoPlayer    = $('#video-player');
    const videoFilename  = $('#video-filename');
    const videoDuration  = $('#video-duration');
    const btnChangeVideo = $('#btn-change-video');
    const groundTruthInput = $('#ground-truth-input');

    // Actions & States
    const btnAnalyze           = $('#btn-analyze');
    const actionNote           = $('#action-note');
    const predictionEmpty      = $('#prediction-empty');
    const predictionProcessing = $('#prediction-processing');
    const predictionResult     = $('#prediction-result');

    // Result Containers
    const singleResultView    = $('#single-result-view');
    const compareResultView   = $('#compare-result-view');
    const singleModelTitle    = $('#single-model-title');
    const singleModelTime     = $('#single-model-time');
    const resultTranscript    = $('#result-transcript');
    const singleAccuracyBox   = $('#single-accuracy-box');

    const compareAvsrText     = $('#compare-avsr-text');
    const compareAvsrTime     = $('#compare-avsr-time');
    const compareAvsrAccuracy = $('#compare-avsr-accuracy');
    const compareGridText     = $('#compare-grid-text');
    const compareGridTime     = $('#compare-grid-time');
    const compareGridAccuracy = $('#compare-grid-accuracy');

    const btnCopy             = $('#btn-copy');
    const btnRunAgain         = $('#btn-run-again');

    // Error & Status
    const errorBanner      = $('#error-banner');
    const errorText        = $('#error-text');
    const errorBannerClose = $('#error-banner-close');
    const modelStatus      = $('#model-status');
    const modelStatusText  = $('#model-status-text');
    const gpuBadge         = $('#gpu-badge');
    const gpuName          = $('#gpu-name');
    const resultsSubhead   = $('#results-subhead');
    const toast            = $('#toast');

    // Metrics
    const metricDuration = $('#metric-duration');
    const metricFrames   = $('#metric-frames');
    const metricTime     = $('#metric-time');

    // ---------------------------------------------------------------
    // 2. Application State
    // ---------------------------------------------------------------
    let selectedFile = null;
    let isProcessing = false;
    let currentModel = 'auto_avsr';
    let lastResultData = null;

    // Preset Sample Ground Truth
    const SAMPLE_GROUND_TRUTH = {
        'sample.mp4': "The first lesson is about humility",
        'sample_ted_2.mp4': "I would love to hear from you",
        'sample_grid.mp4': "bin blue at F two now",
        'sample_grid_2.mp4': "bin blue at L six now"
    };

    // ---------------------------------------------------------------
    // 3. Initialization
    // ---------------------------------------------------------------
    function init() {
        bindThemeToggle();
        bindModelPills();
        bindUploadEvents();
        bindSampleChips();
        bindPredictionEvents();
        bindGroundTruthInput();
        bindModals();
        bindKeyboardShortcuts();
        fetchModelStatus();
    }

    // ---------------------------------------------------------------
    // 4. Theme Switcher
    // ---------------------------------------------------------------
    function bindThemeToggle() {
        if (!themeToggleBtn) return;
        themeToggleBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            try {
                localStorage.setItem('silentspeech-theme', next);
            } catch (e) {}
        });
    }

    // ---------------------------------------------------------------
    // 5. Model Selection Segmented Control
    // ---------------------------------------------------------------
    function bindModelPills() {
        const pills = $$('.model-pill');
        const radios = $$('input[name="model-choice"]');

        radios.forEach((radio) => {
            radio.addEventListener('change', (e) => {
                currentModel = e.target.value;
                pills.forEach((p) => {
                    const r = p.querySelector('input[type="radio"]');
                    p.classList.toggle('model-option--active', r && r.value === currentModel);
                });

                showToast(`Switched engine: ${formatModelName(currentModel)}`);
            });
        });

        pills.forEach((pill) => {
            pill.addEventListener('click', () => {
                const radio = pill.querySelector('input[type="radio"]');
                if (radio && !radio.checked) {
                    radio.checked = true;
                    radio.dispatchEvent(new Event('change'));
                }
            });
        });
    }

    function formatModelName(model) {
        if (model === 'auto_avsr') return 'Auto-AVSR (Open Vocab)';
        if (model === 'grid') return 'GRID Baseline (51 Words)';
        if (model === 'compare') return 'Side-by-Side Comparison';
        return model;
    }

    // ---------------------------------------------------------------
    // 6. Modals & Popups Manager
    // ---------------------------------------------------------------
    function openModal(id) {
        const modal = document.getElementById(id);
        if (modal) {
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    }

    function closeModal(id) {
        const modal = document.getElementById(id);
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }
    }

    function bindModals() {
        // Modal triggers
        if (btnOpenVisualModal) {
            btnOpenVisualModal.addEventListener('click', () => openModal('modal-visual-roi'));
        }
        if (btnInspectRoiLink) {
            btnInspectRoiLink.addEventListener('click', () => openModal('modal-visual-roi'));
        }
        if (btnOpenGroundTruthModal) {
            btnOpenGroundTruthModal.addEventListener('click', () => openModal('modal-ground-truth'));
        }
        if (btnOpenInfoModal) {
            btnOpenInfoModal.addEventListener('click', () => openModal('modal-info'));
        }

        // Close triggers
        $$('[data-close-modal]').forEach((btn) => {
            btn.addEventListener('click', () => {
                const modalId = btn.getAttribute('data-close-modal');
                closeModal(modalId);
            });
        });

        // Click outside dialog to close
        $$('.app-modal-overlay').forEach((overlay) => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    closeModal(overlay.id);
                }
            });
        });

        // ESC key to close any active modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                $$('.app-modal-overlay').forEach((overlay) => {
                    if (overlay.style.display !== 'none') {
                        closeModal(overlay.id);
                    }
                });
            }
        });

        // Error banner close
        if (errorBannerClose) {
            errorBannerClose.addEventListener('click', hideError);
        }
    }



    // ---------------------------------------------------------------
    // 8. Video Upload & Preview Events
    // ---------------------------------------------------------------
    function bindUploadEvents() {
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('upload-canvas--drag-over');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('upload-canvas--drag-over');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('upload-canvas--drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        btnChangeVideo.addEventListener('click', () => {
            resetUpload();
            fileInput.click();
        });
    }

    // ---------------------------------------------------------------
    // 9. Sample Clips Quick Preset
    // ---------------------------------------------------------------
    function bindSampleChips() {
        const chips = $$('.sample-chip');
        chips.forEach((chip) => {
            chip.addEventListener('click', async () => {
                const sampleName = chip.getAttribute('data-sample');
                if (!sampleName) return;

                showToast(`Loading sample: ${sampleName}...`);
                try {
                    const response = await fetch(`/static/samples/${sampleName}`);
                    if (!response.ok) throw new Error('Could not load sample video.');
                    const blob = await response.blob();
                    const file = new File([blob], sampleName, { type: 'video/mp4' });

                    // Auto-fill ground truth if known
                    if (SAMPLE_GROUND_TRUTH[sampleName]) {
                        groundTruthInput.value = SAMPLE_GROUND_TRUTH[sampleName];
                        updateGroundTruthIndicator();
                    }

                    handleFile(file);
                } catch (err) {
                    showError(`Failed to load sample clip: ${err.message}`);
                }
            });
        });
    }

    function handleFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        const allowed = ['mp4', 'avi', 'mov', 'mkv', 'mpg', 'mpeg', 'webm', 'wmv'];
        if (!allowed.includes(ext)) {
            showError(`Unsupported video format ".${ext}". Allowed: ${allowed.join(', ')}`);
            return;
        }

        hideError();
        selectedFile = file;

        const url = URL.createObjectURL(file);
        videoPlayer.src = url;
        videoPlayer.onloadedmetadata = () => {
            videoDuration.textContent = formatDuration(videoPlayer.duration);
        };

        videoFilename.textContent = truncateFilename(file.name, 28);
        uploadZone.style.display = 'none';
        videoPreview.style.display = 'flex';
        btnChangeVideo.style.display = 'inline-flex';

        btnAnalyze.disabled = false;
        if (actionNote) actionNote.textContent = 'Ready to transcribe • Click below or press Ctrl+Enter';

        showState('empty');
        resetAnalysisFrames();
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';

        videoPlayer.src = '';
        videoPlayer.load();
        videoFilename.textContent = '';
        videoDuration.textContent = '';

        uploadZone.style.display = 'flex';
        videoPreview.style.display = 'none';
        btnChangeVideo.style.display = 'none';

        btnAnalyze.disabled = true;
        if (actionNote) actionNote.textContent = 'Select or drop a video to begin';

        showState('empty');
        resetAnalysisFrames();
        resetSpecs();
    }

    // ---------------------------------------------------------------
    // 10. Ground Truth Dynamic Word Diffing
    // ---------------------------------------------------------------
    function bindGroundTruthInput() {
        if (!groundTruthInput) return;

        groundTruthInput.addEventListener('input', () => {
            updateGroundTruthIndicator();
            if (lastResultData) {
                renderResultData(lastResultData);
            }
        });

        if (btnClearGt) {
            btnClearGt.addEventListener('click', () => {
                groundTruthInput.value = '';
                updateGroundTruthIndicator();
                if (lastResultData) {
                    renderResultData(lastResultData);
                }
                showToast('Ground truth cleared');
            });
        }
    }

    function updateGroundTruthIndicator() {
        const hasText = (groundTruthInput?.value || '').trim().length > 0;
        if (gtActiveIndicator) {
            gtActiveIndicator.style.display = hasText ? 'block' : 'none';
        }
    }

    // ---------------------------------------------------------------
    // 11. Prediction Workflow
    // ---------------------------------------------------------------
    function bindPredictionEvents() {
        btnAnalyze.addEventListener('click', runPrediction);
        btnRunAgain.addEventListener('click', runPrediction);

        btnCopy.addEventListener('click', () => {
            if (!lastResultData) return;

            let textToCopy = '';
            if (lastResultData.compare) {
                const avsr = lastResultData.auto_avsr?.transcript || '';
                const grid = lastResultData.grid?.transcript || '';
                textToCopy = `Auto-AVSR (Open Vocabulary): ${avsr}\nGRID Baseline: ${grid}`;
            } else {
                textToCopy = lastResultData.transcript || '';
            }

            if (!textToCopy) return;

            navigator.clipboard.writeText(textToCopy).then(() => {
                showToast('Transcript copied to clipboard');
            }).catch(() => {
                showToast('Failed to copy');
            });
        });
    }

    function bindKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                if (!btnAnalyze.disabled && !isProcessing) {
                    e.preventDefault();
                    runPrediction();
                }
            }
        });
    }

    async function runPrediction() {
        if (!selectedFile || isProcessing) return;
        isProcessing = true;
        hideError();

        btnAnalyze.disabled = true;
        btnAnalyze.querySelector('.btn-spinner').style.display = 'inline-block';
        btnAnalyze.querySelector('.btn-transcribe-icon').style.display = 'none';
        if (actionNote) actionNote.textContent = 'Neural engine transcribing lips...';

        showState('processing');
        startProcessingTerminal();

        const formData = new FormData();
        formData.append('video', selectedFile);
        formData.append('model', currentModel);

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            stopProcessingTerminal();
            lastResultData = data;

            renderResultData(data);
            showState('result');

            if (data.frames) {
                updateAnalysisFrames(data.frames);
            }

            updateSpecs(data);
            if (btnCopy) btnCopy.style.display = 'inline-flex';
            if (btnRunAgain) btnRunAgain.style.display = 'inline-flex';
            if (actionNote) actionNote.textContent = 'Transcription complete';

        } catch (err) {
            stopProcessingTerminal();
            showState('empty');
            showError(err.message || 'An error occurred during prediction.');
            if (actionNote) actionNote.textContent = 'Error occurred during processing';
        } finally {
            isProcessing = false;
            btnAnalyze.disabled = false;
            btnAnalyze.querySelector('.btn-spinner').style.display = 'none';
            btnAnalyze.querySelector('.btn-transcribe-icon').style.display = 'inline-block';
        }
    }

    // ---------------------------------------------------------------
    // 12. Render Results & Compute Accuracy
    // ---------------------------------------------------------------
    function renderResultData(data) {
        const groundTruth = (groundTruthInput?.value || '').trim();

        if (data.compare) {
            singleResultView.style.display = 'none';
            compareResultView.style.display = 'grid';
            if (resultsSubhead) resultsSubhead.textContent = `Comparison [${data.time_seconds || 0}s]`;

            // Auto-AVSR
            const avsrRaw = data.auto_avsr?.transcript || '(No speech detected)';
            const avsrSec = data.auto_avsr?.time_seconds != null ? `[${data.auto_avsr.time_seconds}s]` : '';
            compareAvsrTime.textContent = avsrSec;

            if (groundTruth) {
                const diff = computeWordDiff(avsrRaw, groundTruth);
                compareAvsrText.innerHTML = diff.html;
                compareAvsrAccuracy.style.display = 'flex';
                compareAvsrAccuracy.innerHTML = `<span class="accuracy-rate">${diff.accuracy}% match</span> (${diff.matched}/${diff.totalGround} target words)`;
            } else {
                compareAvsrText.textContent = avsrRaw;
                compareAvsrAccuracy.style.display = 'none';
            }

            // GRID Baseline
            const gridRaw = data.grid?.transcript || '(No speech detected)';
            const gridSec = data.grid?.time_seconds != null ? `[${data.grid.time_seconds}s]` : '';
            compareGridTime.textContent = gridSec;

            if (groundTruth) {
                const diff = computeWordDiff(gridRaw, groundTruth);
                compareGridText.innerHTML = diff.html;
                compareGridAccuracy.style.display = 'flex';
                compareGridAccuracy.innerHTML = `<span class="accuracy-rate">${diff.accuracy}% match</span> (${diff.matched}/${diff.totalGround} target words)`;
            } else {
                compareGridText.textContent = gridRaw;
                compareGridAccuracy.style.display = 'none';
            }

        } else {
            singleResultView.style.display = 'flex';
            compareResultView.style.display = 'none';

            const rawText = data.transcript || '(No speech detected)';
            const modelName = data.model_name || (data.model === 'grid' ? 'GRID Baseline' : 'Auto-AVSR');
            const timeStr = data.time_seconds != null ? `[${data.time_seconds}s]` : '';

            if (resultsSubhead) resultsSubhead.textContent = `${modelName} ${timeStr}`;
            if (singleModelTitle) singleModelTitle.textContent = modelName;
            if (singleModelTime) singleModelTime.textContent = timeStr;

            if (groundTruth) {
                const diff = computeWordDiff(rawText, groundTruth);
                resultTranscript.innerHTML = diff.html;
                singleAccuracyBox.style.display = 'flex';
                singleAccuracyBox.innerHTML = `<span class="accuracy-rate">${diff.accuracy}% match</span> (${diff.matched}/${diff.totalGround} target words)`;
            } else {
                resultTranscript.textContent = rawText;
                singleAccuracyBox.style.display = 'none';
            }
        }
    }

    function computeWordDiff(predictedStr, groundTruthStr) {
        if (!predictedStr || predictedStr.startsWith('(')) {
            return { html: escapeHtml(predictedStr), accuracy: 0, matched: 0, totalGround: 1 };
        }

        const normalizeWord = (w) => w.toLowerCase().replace(/[^a-z0-9]/gi, '');
        const groundWords = groundTruthStr.trim().split(/\s+/).filter(Boolean);
        const groundNorm = groundWords.map(normalizeWord);
        const predWords = predictedStr.trim().split(/\s+/).filter(Boolean);

        let matchedCount = 0;
        const groundMatched = new Array(groundNorm.length).fill(false);

        const renderedTokens = predWords.map((word) => {
            const norm = normalizeWord(word);
            const idx = groundNorm.findIndex((gw, i) => !groundMatched[i] && gw === norm);
            if (idx !== -1) {
                groundMatched[idx] = true;
                matchedCount++;
                return `<span class="word-tag word-tag--correct">${escapeHtml(word)}</span>`;
            } else {
                return `<span class="word-tag word-tag--wrong">${escapeHtml(word)}</span>`;
            }
        });

        const totalGround = Math.max(groundNorm.length, 1);
        const accuracy = Math.round((matchedCount / totalGround) * 100);

        return {
            html: renderedTokens.join(' '),
            accuracy,
            matched: matchedCount,
            totalGround: groundNorm.length
        };
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ---------------------------------------------------------------
    // 13. UI Canvas State Management
    // ---------------------------------------------------------------
    function showState(state) {
        predictionEmpty.style.display      = state === 'empty'      ? 'flex' : 'none';
        predictionProcessing.style.display = state === 'processing' ? 'flex' : 'none';
        predictionResult.style.display     = state === 'result'     ? 'flex' : 'none';
    }

    // ---------------------------------------------------------------
    // 14. Terminal Log Scanning Progress
    // ---------------------------------------------------------------
    let stageInterval = null;

    function startProcessingTerminal() {
        const stages = [
            $('#proc-stage-0'),
            $('#proc-stage-1'),
            $('#proc-stage-2')
        ];

        stages.forEach((s) => {
            if (s) s.className = 'infer-step';
        });

        if (stages[0]) stages[0].className = 'infer-step infer-step--active';

        let current = 0;
        stageInterval = setInterval(() => {
            if (current < stages.length && stages[current]) {
                stages[current].className = 'infer-step infer-step--done';
            }
            current++;
            if (current < stages.length && stages[current]) {
                stages[current].className = 'infer-step infer-step--active';
            } else {
                clearInterval(stageInterval);
            }
        }, 750);
    }

    function stopProcessingTerminal() {
        if (stageInterval) clearInterval(stageInterval);
        const stages = [
            $('#proc-stage-0'),
            $('#proc-stage-1'),
            $('#proc-stage-2')
        ];
        stages.forEach((s) => {
            if (s) s.className = 'infer-step infer-step--done';
        });
    }

    // ---------------------------------------------------------------
    // 15. Visual Analysis (Mouth Crop Frames)
    // ---------------------------------------------------------------
    function updateAnalysisFrames(frames) {
        if (frames.cropped) {
            setFrameImage('analysis-cropped', frames.cropped);
        }
        if (frames.processed) {
            setFrameImage('analysis-processed', frames.processed);
        }
        if (frames.annotated) {
            setFrameImage('analysis-original', frames.annotated);
        } else if (frames.original) {
            setFrameImage('analysis-original', frames.original);
        }
    }

    function setFrameImage(containerId, base64) {
        const el = document.getElementById(containerId);
        if (!el || !base64) return;

        el.innerHTML = '';
        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + base64;
        img.alt = containerId;
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.25s ease-in-out';
        img.onload = () => { img.style.opacity = '1'; };
        el.appendChild(img);
    }


    function resetAnalysisFrames() {
        ['analysis-cropped', 'analysis-processed', 'analysis-original'].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = '<span class="roi-empty-tag">Awaiting video run</span>';
        });
    }

    // ---------------------------------------------------------------
    // 16. Specifications & Metrics
    // ---------------------------------------------------------------
    function updateSpecs(data) {
        if (data.video_duration_s != null && metricDuration) {
            metricDuration.textContent = formatDuration(data.video_duration_s);
        }
        if (data.num_frames != null && metricFrames) {
            metricFrames.textContent = data.num_frames + 'f';
        }
        if (data.time_seconds != null && metricTime) {
            metricTime.textContent = data.time_seconds + 's';
        }
    }

    function resetSpecs() {
        if (metricDuration) metricDuration.textContent = '-';
        if (metricFrames) metricFrames.textContent = '-';
        if (metricTime) metricTime.textContent = '-';
    }

    // ---------------------------------------------------------------
    // 17. Utilities & Status Polling
    // ---------------------------------------------------------------
    async function fetchModelStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();

            if (data.model_loaded) {
                if (modelStatus) modelStatus.style.opacity = '1';
                if (modelStatusText) modelStatusText.textContent = 'Engine Ready';
            } else {
                if (modelStatusText) modelStatusText.textContent = 'Loading weights...';
            }

            if (data.gpu_available && data.gpu_name) {
                if (gpuName) gpuName.textContent = data.gpu_name.replace('Laptop GPU', '').trim();
                if (gpuBadge) gpuBadge.style.display = 'inline-flex';
            }
        } catch (err) {
            if (modelStatusText) modelStatusText.textContent = 'Offline';
        }
    }

    function showError(msg) {
        if (errorText) errorText.textContent = msg;
        if (errorBanner) errorBanner.style.display = 'flex';
    }

    function hideError() {
        if (errorBanner) errorBanner.style.display = 'none';
    }

    let toastTimer = null;
    function showToast(msg) {
        if (!toast) return;
        toast.textContent = msg;
        toast.classList.add('toast-box--visible');
        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            toast.classList.remove('toast-box--visible');
        }, 2200);
    }

    function formatDuration(sec) {
        if (!sec || isNaN(sec)) return '0:00';
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
    }

    function truncateFilename(name, maxLen) {
        if (!name || name.length <= maxLen) return name;
        const parts = name.split('.');
        const ext = parts.length > 1 ? '.' + parts.pop() : '';
        const base = parts.join('.');
        return base.slice(0, maxLen - ext.length - 3) + '...' + ext;
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
