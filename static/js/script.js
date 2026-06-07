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

function processFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a valid image file (JPG, PNG, JPEG)', 'error');
        return;
    }
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File size must be less than 16MB', 'error');
        return;
    }

    // Show validating message
    showValidationOverlay('🔍 Validating your image...');

    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = function() {
        // Step 1: Blur check
        const blurScore = getBlurScore(img);
        if (blurScore < 30) {
            hideValidationOverlay();
            URL.revokeObjectURL(url);
            showSmartWarning(
                '📷 Blurry Image Detected!',
                'Your image appears to be too blurry for accurate analysis. Please upload a clear, well-lit photo of the leaf.',
                'warning',
                null
            );
            return;
        }

        // Step 2: Leaf/green color check
        const greenScore = getGreenScore(img);
        if (greenScore < 8) {
            hideValidationOverlay();
            URL.revokeObjectURL(url);
            showSmartWarning(
                '🌿 Not a Plant Leaf!',
                'Oops! This doesn\'t look like a plant leaf image. Our system is trained exclusively on plant leaf photos. Please upload a clear image of a plant leaf.',
                'error',
                '/contact'
            );
            return;
        }

        hideValidationOverlay();
        URL.revokeObjectURL(url);
        uploadedFile = file;
        displayImagePreview(file);
        showNotification('✅ Image validated! Ready to analyze.', 'success');
    };
    img.onerror = function() {
        hideValidationOverlay();
        showNotification('Could not read image. Please try another file.', 'error');
    };
    img.src = url;
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
