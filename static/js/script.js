/* ===================================
   Plant Disease Detection System
   Interactive JavaScript - UPDATED
   =================================== */

// Global variables
let uploadedFile = null;

// DOM Elements - use functions to get them safely after DOM loads
let uploadArea, fileInput, imagePreview, previewImage,
    removeImageBtn, predictBtn, loading, resultsCard,
    newPredictionBtn, downloadReportBtn, clearHistoryBtn, historyGrid;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Assign DOM elements here so they are always available
    uploadArea       = document.getElementById('uploadArea');
    fileInput        = document.getElementById('fileInput');
    imagePreview     = document.getElementById('imagePreview');
    previewImage     = document.getElementById('previewImage');
    removeImageBtn   = document.getElementById('removeImage');
    predictBtn       = document.getElementById('predictBtn');
    loading          = document.getElementById('loading');
    resultsCard      = document.getElementById('resultsCard');
    newPredictionBtn = document.getElementById('newPrediction');
    downloadReportBtn= document.getElementById('downloadReport');
    clearHistoryBtn  = document.getElementById('clearHistory');
    historyGrid      = document.getElementById('historyGrid');

    initializeEventListeners();
    loadHistory();
    setupSmoothScroll();
});

// ===================================
// Event Listeners Setup
// ===================================
function initializeEventListeners() {
    if (!fileInput) return;

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Remove image button
    if (removeImageBtn) removeImageBtn.addEventListener('click', removeImage);

    // Predict button
    if (predictBtn) predictBtn.addEventListener('click', predictDisease);

    // New prediction button
    if (newPredictionBtn) newPredictionBtn.addEventListener('click', resetUpload);

    // Download report button
    if (downloadReportBtn) downloadReportBtn.addEventListener('click', downloadReport);

    // Clear history button
    if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', clearHistory);

    // Prevent default drag behavior on document
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());

    // Allow clicking anywhere on upload area (not just the label)
    // But skip if click is on the label or fileInput itself to avoid double dialog
    uploadArea.addEventListener('click', function(e) {
        if (uploadedFile) return; // already has file
        if (e.target.closest('.btn-remove')) return;
        if (e.target.closest('label')) return; // label already triggers fileInput
        if (e.target === fileInput) return;
        fileInput.click();
    });
}

// ===================================
// File Upload Handling
// ===================================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) processFile(file);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
}

// ===================================
// MobileNet model instance (loaded once)
// ===================================
let mobileNetModel = null;
let modelLoading = false;

async function loadMobileNet() {
    if (mobileNetModel) return mobileNetModel;
    if (modelLoading) {
        // Wait for existing load
        while (modelLoading) await new Promise(r => setTimeout(r, 100));
        return mobileNetModel;
    }
    modelLoading = true;
    try {
        mobileNetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
        console.log('✅ MobileNet loaded!');
    } catch(e) {
        console.warn('MobileNet load failed:', e);
        mobileNetModel = null;
    }
    modelLoading = false;
    return mobileNetModel;
}

// Plant-related keywords MobileNet recognizes
const PLANT_KEYWORDS = [
    'leaf', 'plant', 'flower', 'tree', 'vegetable', 'herb', 'shrub',
    'moss', 'fern', 'grass', 'weed', 'vine', 'bush', 'blossom',
    'petal', 'stem', 'branch', 'twig', 'foliage', 'crop', 'garden',
    'nature', 'botanical', 'tomato', 'potato', 'apple', 'grape',
    'corn', 'maize', 'pepper', 'strawberry', 'peach', 'cherry',
    'orange', 'raspberry', 'soybean', 'squash', 'blueberry',
    'cabbage', 'spinach', 'lettuce', 'broccoli', 'cucumber',
    'zucchini', 'artichoke', 'head cabbage', 'daisy', 'sunflower',
    'acorn', 'mushroom', 'fungus', 'algae', 'seaweed', 'cactus',
    'aloe', 'bamboo', 'banana', 'fig', 'mango', 'coconut',
    'conifer', 'pine', 'oak', 'maple', 'eucalyptus'
];

// Non-plant keywords to always reject
const REJECT_KEYWORDS = [
    'person', 'man', 'woman', 'girl', 'boy', 'human', 'face', 'people',
    'laptop', 'computer', 'phone', 'mobile', 'screen', 'monitor', 'keyboard',
    'car', 'vehicle', 'bus', 'truck', 'bicycle', 'motorcycle',
    'building', 'house', 'room', 'office', 'street', 'road',
    'dog', 'cat', 'animal', 'bird', 'fish',
    'food', 'pizza', 'burger', 'sandwich', 'cake', 'bread',
    'book', 'paper', 'pen', 'table', 'chair', 'furniture',
    'sky', 'cloud', 'mountain', 'ocean', 'beach', 'sand',
    'cartoon', 'drawing', 'painting', 'art'
];

async function processFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a valid image file (JPG, PNG, JPEG)', 'error');
        return;
    }
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File size must be less than 16MB', 'error');
        return;
    }

    // Show validating overlay
    showValidationOverlay('🔍 Checking your image...');

    const imgEl = new Image();
    const url = URL.createObjectURL(file);

    imgEl.onload = async function() {
        try {
            // Step 1 — Blur check (fast, no model needed)
            const blurScore = getBlurScore(imgEl);
            if (blurScore < 25) {
                hideValidationOverlay();
                URL.revokeObjectURL(url);
                showSmartWarning(
                    '📷 Image Too Blurry!',
                    'Your image is too blurry for accurate disease detection. Please upload a clear, well-lit photo of the plant leaf.',
                    'warning', null
                );
                return;
            }

            // Step 2 — MobileNet AI check
            showValidationOverlay('🤖 AI is checking if this is a plant...');
            const model = await loadMobileNet();

            if (model) {
                const predictions = await model.classify(imgEl, 5);
                console.log('MobileNet predictions:', predictions);

                // Check top predictions
                const topLabels = predictions.map(p => p.className.toLowerCase());
                const topProbs  = predictions.map(p => p.probability);

                // Check for hard reject (non-plant detected with high confidence)
                let rejectReason = null;
                for (let i = 0; i < topLabels.length; i++) {
                    const label = topLabels[i];
                    const prob  = topProbs[i];
                    for (const kw of REJECT_KEYWORDS) {
                        if (label.includes(kw) && prob > 0.15) {
                            rejectReason = label;
                            break;
                        }
                    }
                    if (rejectReason) break;
                }

                if (rejectReason) {
                    hideValidationOverlay();
                    URL.revokeObjectURL(url);
                    showSmartWarning(
                        '🚫 Not a Plant Leaf!',
                        `Our AI detected this looks like a "${rejectReason}" image, not a plant leaf. Please upload a clear photo of a plant leaf only.`,
                        'error', '/contact'
                    );
                    return;
                }

                // Check if plant keywords found
                let isPlant = false;
                for (const label of topLabels) {
                    for (const kw of PLANT_KEYWORDS) {
                        if (label.includes(kw)) { isPlant = true; break; }
                    }
                    if (isPlant) break;
                }

                // Also check top prediction probability
                // If top prediction is plant with >20% confidence, allow
                if (!isPlant && topProbs[0] > 0.5) {
                    // High confidence non-plant
                    hideValidationOverlay();
                    URL.revokeObjectURL(url);
                    showSmartWarning(
                        '🌿 Not Recognized as a Plant!',
                        `This image doesn't appear to be a plant leaf. Our system only analyzes plant leaf images. If this IS a plant not in our dataset, you can request to add it!`,
                        'error', '/contact'
                    );
                    return;
                }

            } else {
                // MobileNet failed to load — fall back to color check
                console.warn('MobileNet unavailable, using color check fallback');
                const greenScore = getGreenScore(imgEl);
                if (greenScore < 8) {
                    hideValidationOverlay();
                    URL.revokeObjectURL(url);
                    showSmartWarning(
                        '🌿 Not a Plant Leaf!',
                        'This does not look like a plant leaf image. Please upload a clear photo of a plant leaf.',
                        'error', '/contact'
                    );
                    return;
                }
            }

            // ✅ All checks passed!
            hideValidationOverlay();
            URL.revokeObjectURL(url);
            uploadedFile = file;
            displayImagePreview(file);
            showNotification('✅ Plant leaf detected! Ready to analyze.', 'success');

        } catch (err) {
            console.error('Validation error:', err);
            hideValidationOverlay();
            URL.revokeObjectURL(url);
            // On any error, allow the image (don't block user)
            uploadedFile = file;
            displayImagePreview(file);
            showNotification('✅ Image loaded! Ready to analyze.', 'success');
        }
    };

    imgEl.onerror = function() {
        hideValidationOverlay();
        showNotification('Could not read image. Please try another file.', 'error');
    };

    imgEl.src = url;
}

// ===================================
// Image Validation Helpers
// ===================================

function getBlurScore(img) {
    const canvas = document.createElement('canvas');
    const size = 100;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;

    // Laplacian variance — higher = sharper
    let sum = 0, sumSq = 0, count = 0;
    for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
        sum += gray;
        sumSq += gray * gray;
        count++;
    }
    const mean = sum / count;
    const variance = (sumSq / count) - (mean * mean);
    return Math.sqrt(variance); // higher = sharper
}

function getGreenScore(img) {
    const canvas = document.createElement('canvas');
    const size = 100;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;

    let greenPixels = 0;
    const total = data.length / 4;

    for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i+1], b = data[i+2];
        // Check for greenish or brownish/yellowish (plant colors)
        const isGreen  = g > r * 0.9 && g > b * 0.9 && g > 40;
        const isBrown  = r > 80 && g > 50 && b < 80 && r > g;
        const isYellow = r > 150 && g > 150 && b < 100;
        if (isGreen || isBrown || isYellow) greenPixels++;
    }

    return (greenPixels / total) * 100;
}

function showValidationOverlay(message) {
    let overlay = document.getElementById('validationOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'validationOverlay';
        overlay.style.cssText = `
            position:fixed;top:0;left:0;width:100%;height:100%;
            background:rgba(0,0,0,0.5);z-index:9999;
            display:flex;align-items:center;justify-content:center;
            backdrop-filter:blur(4px);
        `;
        overlay.innerHTML = `
            <div style="background:white;padding:2rem 3rem;border-radius:1rem;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
                <div style="font-size:2.5rem;margin-bottom:1rem;">🔍</div>
                <div id="validationMsg" style="font-family:Poppins,sans-serif;font-size:1.1rem;color:#1f2937;font-weight:600;">${message}</div>
                <div style="margin-top:0.75rem;color:#6b7280;font-size:0.85rem;">Please wait...</div>
            </div>
        `;
        document.body.appendChild(overlay);
    }
}

function hideValidationOverlay() {
    const overlay = document.getElementById('validationOverlay');
    if (overlay) overlay.remove();
}

function showSmartWarning(title, message, type, contactLink) {
    // Remove existing
    const existing = document.getElementById('smartWarning');
    if (existing) existing.remove();

    const colors = {
        error:   { bg: '#fef2f2', border: '#ef4444', icon: '❌', btn: '#ef4444' },
        warning: { bg: '#fffbeb', border: '#f59e0b', icon: '⚠️', btn: '#f59e0b' },
    };
    const c = colors[type] || colors.error;

    const div = document.createElement('div');
    div.id = 'smartWarning';
    div.style.cssText = `
        position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
        background:white;border-radius:1.25rem;padding:2rem;
        box-shadow:0 25px 60px rgba(0,0,0,0.25);z-index:9999;
        max-width:420px;width:90%;text-align:center;
        border-top:5px solid ${c.border};
        animation:popIn 0.3s ease;
        font-family:Poppins,sans-serif;
    `;

    div.innerHTML = `
        <div style="font-size:3rem;margin-bottom:0.5rem;">${c.icon}</div>
        <h3 style="color:#1f2937;margin:0 0 0.75rem;font-size:1.2rem;">${title}</h3>
        <p style="color:#6b7280;font-size:0.9rem;line-height:1.6;margin:0 0 1.25rem;">${message}</p>
        ${contactLink ? `<a href="${contactLink}" style="display:inline-block;background:#10b981;color:white;padding:0.5rem 1.25rem;border-radius:9999px;text-decoration:none;font-size:0.85rem;font-weight:600;margin-bottom:0.75rem;">📩 Request to Add Plant</a><br>` : ''}
        <button onclick="document.getElementById('smartWarning').remove();document.getElementById('fileInput').value='';"
            style="background:${c.btn};color:white;border:none;padding:0.6rem 1.5rem;border-radius:9999px;cursor:pointer;font-family:Poppins,sans-serif;font-weight:600;font-size:0.9rem;margin-top:0.5rem;">
            Try Another Image
        </button>
    `;

    // Backdrop
    const backdrop = document.createElement('div');
    backdrop.id = 'smartWarningBackdrop';
    backdrop.style.cssText = `position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.4);z-index:9998;backdrop-filter:blur(3px);`;
    backdrop.onclick = () => { div.remove(); backdrop.remove(); document.getElementById('fileInput').value = ''; };

    document.body.appendChild(backdrop);
    document.body.appendChild(div);
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;

        const uploadContent = uploadArea.querySelector('.upload-content');
        if (uploadContent) uploadContent.style.display = 'none';

        imagePreview.style.display = 'block';
        predictBtn.style.display = 'block';

        imagePreview.style.animation = 'fadeIn 0.5s ease';
        predictBtn.style.animation  = 'fadeInUp 0.5s ease';
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    uploadedFile = null;
    fileInput.value = '';

    const uploadContent = uploadArea.querySelector('.upload-content');
    if (uploadContent) uploadContent.style.display = 'block';

    imagePreview.style.display = 'none';
    predictBtn.style.display   = 'none';
    resultsCard.style.display  = 'none';
}

function resetUpload() {
    removeImage();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ===================================
// Disease Prediction
// ===================================
async function predictDisease() {
    if (!uploadedFile) {
        showNotification('Please select an image first', 'error');
        return;
    }

    loading.style.display     = 'flex';
    predictBtn.style.display  = 'none';
    resultsCard.style.display = 'none';

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            addToHistory(data);
            // Low confidence = unknown/unsupported plant
            if (data.confidence < 50) {
                setTimeout(() => {
                    showSmartWarning(
                        '🌱 Unknown Plant Detected!',
                        `Our model isn't confident about this plant (${Math.round(data.confidence)}% confidence). This plant may not be in our training dataset yet. You can request to add it!`,
                        'warning',
                        '/contact'
                    );
                }, 1000);
            }
        } else {
            showNotification(data.error || 'Prediction failed', 'error');
            predictBtn.style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification(error.message || 'Connection error. Please try again.', 'error');
        predictBtn.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

// ===================================
// Results Display
// ===================================
function displayResults(data) {
    document.getElementById('diseaseName').textContent = formatDiseaseName(data.disease);

    const confidence = Math.round(data.confidence);
    document.getElementById('confidenceBadge').textContent = `${confidence}%`;

    const confidenceFill = document.getElementById('confidenceFill');
    setTimeout(() => { confidenceFill.style.width = `${confidence}%`; }, 100);

    document.getElementById('diseaseDescription').textContent = data.info?.description || '-';
    document.getElementById('diseaseTreatment').textContent   = data.info?.treatment   || '-';
    document.getElementById('diseasePrevention').textContent  = data.info?.prevention  || '-';

    displayTopPredictions(data.top_predictions || []);

    resultsCard.style.display   = 'block';
    resultsCard.style.animation = 'slideInRight 0.5s ease';

    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

function displayTopPredictions(predictions) {
    const container = document.getElementById('topPredictions');
    container.innerHTML = '';

    if (!predictions || predictions.length === 0) {
        container.innerHTML = '<p style="color:#6b7280;font-size:0.85rem;padding:0.5rem 0">Top predictions not available for this result.</p>';
        return;
    }

    predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.animation = `fadeInUp 0.5s ease ${index * 0.1}s both`;
        item.innerHTML = `
            <span class="prediction-name">${index + 1}. ${formatDiseaseName(pred.disease)}</span>
            <span class="prediction-confidence">${Math.round(pred.confidence)}%</span>
        `;
        container.appendChild(item);
    });
}

function formatDiseaseName(name) {
    if (!name) return 'Unknown';
    return name.replace(/___/g, ' - ').replace(/_/g, ' ');
}

// ===================================
// History Management (UPDATED)
// ===================================
async function loadHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        if (data.success) {
            displayHistory(data.history || []);
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function displayHistory(history) {
    if (!historyGrid) return;
    historyGrid.innerHTML = '';

    if (!history || history.length === 0) {
        historyGrid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-history"></i>
                <p>No predictions yet. Upload an image to get started!</p>
            </div>`;
        return;
    }

    [...history].reverse().forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'history-card';
        card.style.animation = `fadeInUp 0.5s ease ${index * 0.1}s both`;
        card.innerHTML = `
            <img src="${item.image_url || ''}" alt="Plant leaf" class="history-image" onerror="this.src=''">
            <div class="history-disease">${formatDiseaseName(item.disease)}</div>
            <div class="history-confidence">Confidence: ${Math.round(item.confidence || 0)}%</div>
            <div class="history-timestamp">${item.timestamp || ''}</div>
        `;
        historyGrid.appendChild(card);
    });
}

function addToHistory(data) {
    loadHistory();
}

// ✅ UPDATED CLEAR HISTORY (REAL-TIME FIX)
async function clearHistory() {
    if (!confirm('Are you sure you want to clear all history?')) return;

    try {
        const response = await fetch('/clear-history', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification('History cleared successfully', 'success');
            historyGrid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-history"></i>
                    <p>No predictions yet. Upload an image to get started!</p>
                </div>`;
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        showNotification('Failed to clear history', 'error');
    }
}

// ✅ FILTER BY DATE
async function filterByDate() {
    const selectedDate = document.getElementById('historyDate')?.value;
    if (!selectedDate) {
        showNotification('Please select a date', 'error');
        return;
    }

    try {
        const response = await fetch('/history-by-date', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ date: selectedDate })
        });
        const data = await response.json();

        if (data.success) {
            displayHistory(data.history);
            if (data.history.length === 0) {
                showNotification('No predictions found for this date', 'info');
            }
        } else {
            showNotification('No data found', 'error');
        }
    } catch (error) {
        console.error(error);
        showNotification('Error fetching data', 'error');
    }
}

// ===================================
// Download Report
// ===================================
function downloadReport() {
    const diseaseName  = document.getElementById('diseaseName')?.textContent  || '-';
    const confidence   = document.getElementById('confidenceBadge')?.textContent || '-';
    const description  = document.getElementById('diseaseDescription')?.textContent || '-';
    const treatment    = document.getElementById('diseaseTreatment')?.textContent   || '-';
    const prevention   = document.getElementById('diseasePrevention')?.textContent  || '-';
    const timestamp    = new Date().toLocaleString();

    const reportContent = `
PLANT DISEASE DETECTION REPORT
===============================

Date & Time: ${timestamp}

DIAGNOSIS
---------
Disease: ${diseaseName}
Confidence: ${confidence}

DESCRIPTION
-----------
${description}

TREATMENT RECOMMENDATIONS
------------------------
${treatment}

PREVENTION MEASURES
------------------
${prevention}

===============================
Generated by Plant Disease Detection System
Powered by Deep Learning & TensorFlow

DISCLAIMER: Information is for educational purposes only.
Consult a certified agricultural expert for professional advice.
    `.trim();

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url  = window.URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `plant-disease-report-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    showNotification('Report downloaded successfully', 'success');
}

// ===================================
// Notifications
// ===================================
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.15);
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        max-width: 380px;
        font-family: Poppins, sans-serif;
        font-size: 0.9rem;
    `;
    notification.innerHTML = `
        <div style="display:flex;align-items:center;gap:0.75rem">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => { if (notification.parentNode) document.body.removeChild(notification); }, 300);
    }, 3500);
}

// ===================================
// Smooth Scroll
// ===================================
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (!href || href === '#') return;
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });
}

// ===================================
// Dynamic CSS for fadeOut
// ===================================
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; transform: translateX(0); }
        to   { opacity: 0; transform: translateX(20px); }
    }
    @keyframes popIn {
        from { opacity:0; transform:translate(-50%,-50%) scale(0.8); }
        to   { opacity:1; transform:translate(-50%,-50%) scale(1); }
    }
    #loading {
        display: none;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
    }
`;
document.head.appendChild(style);

// ===================================
// Console Welcome
// ===================================
console.log('%c🌿 Plant Disease Detection System', 'color:#10b981;font-size:20px;font-weight:bold;');
console.log('%cPowered by Deep Learning & TensorFlow', 'color:#6b7280;font-size:14px;');
console.log('%cBE Computer Engineering Project', 'color:#6b7280;font-size:12px;');
