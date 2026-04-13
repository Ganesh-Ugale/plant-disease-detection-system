/* ===================================
   Plant Disease Detection System
   Interactive JavaScript
   =================================== */

// Global variables
let uploadedFile = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImage');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const resultsCard = document.getElementById('resultsCard');
const newPredictionBtn = document.getElementById('newPrediction');
const downloadReportBtn = document.getElementById('downloadReport');
const clearHistoryBtn = document.getElementById('clearHistory');
const historyGrid = document.getElementById('historyGrid');

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadHistory();
    setupSmoothScroll();
});

// ===================================
// Event Listeners Setup
// ===================================
function initializeEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Remove image button
    removeImageBtn.addEventListener('click', removeImage);
    
    // Predict button
    predictBtn.addEventListener('click', predictDisease);
    
    // New prediction button
    newPredictionBtn.addEventListener('click', resetUpload);
    
    // Download report button
    downloadReportBtn.addEventListener('click', downloadReport);
    
    // Clear history button
    clearHistoryBtn.addEventListener('click', clearHistory);
    
    // Prevent default drag behavior on document
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

// ===================================
// File Upload Handling
// ===================================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
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
    if (file) {
        processFile(file);
    }
}

function processFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a valid image file (JPG, PNG)', 'error');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File size must be less than 16MB', 'error');
        return;
    }
    
    uploadedFile = file;
    displayImagePreview(file);
}

function displayImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        
        // Hide upload content and show preview
        uploadArea.querySelector('.upload-content').style.display = 'none';
        imagePreview.style.display = 'block';
        predictBtn.style.display = 'block';
        
        // Add animation
        imagePreview.style.animation = 'fadeIn 0.5s ease';
        predictBtn.style.animation = 'fadeInUp 0.5s ease';
    };
    
    reader.readAsDataURL(file);
}

function removeImage() {
    uploadedFile = null;
    fileInput.value = '';
    
    // Reset upload area
    uploadArea.querySelector('.upload-content').style.display = 'block';
    imagePreview.style.display = 'none';
    predictBtn.style.display = 'none';
    
    // Hide results if shown
    resultsCard.style.display = 'none';
}

function resetUpload() {
    removeImage();
    resultsCard.style.display = 'none';
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
    
    // Show loading
    loading.style.display = 'block';
    predictBtn.style.display = 'none';
    resultsCard.style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    try {
        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
            addToHistory(data);
        } else {
            showNotification(data.error || 'Prediction failed', 'error');
            predictBtn.style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Connection error. Please try again.', 'error');
        predictBtn.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

// ===================================
// Results Display
// ===================================
function displayResults(data) {
    // Update disease name
    document.getElementById('diseaseName').textContent = formatDiseaseName(data.disease);
    
    // Update confidence
    const confidence = Math.round(data.confidence);
    document.getElementById('confidenceBadge').textContent = `${confidence}%`;
    
    // Animate confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);
    
    // Update disease information
    document.getElementById('diseaseDescription').textContent = data.info.description;
    document.getElementById('diseaseTreatment').textContent = data.info.treatment;
    document.getElementById('diseasePrevention').textContent = data.info.prevention;
    
    // Display top 3 predictions
    displayTopPredictions(data.top_predictions);
    
    // Show results card with animation
    resultsCard.style.display = 'block';
    resultsCard.style.animation = 'slideInRight 0.5s ease';
    
    // Scroll to results
    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

function displayTopPredictions(predictions) {
    const container = document.getElementById('topPredictions');
    container.innerHTML = '';
    
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
    // Replace underscores with spaces and format
    return name.replace(/_/g, ' ').replace(/___/g, ' - ');
}

// ===================================
// History Management
// ===================================
async function loadHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            displayHistory(data.history);
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function displayHistory(history) {
    historyGrid.innerHTML = '';
    
    if (history.length === 0) {
        historyGrid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-history"></i>
                <p>No predictions yet. Upload an image to get started!</p>
            </div>
        `;
        return;
    }
    
    history.reverse().forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'history-card';
        card.style.animation = `fadeInUp 0.5s ease ${index * 0.1}s both`;
        
        card.innerHTML = `
            <img src="${item.image_url}" alt="Plant leaf" class="history-image">
            <div class="history-disease">${formatDiseaseName(item.disease)}</div>
            <div class="history-confidence">Confidence: ${Math.round(item.confidence)}%</div>
            <div class="history-timestamp">${item.timestamp}</div>
        `;
        
        historyGrid.appendChild(card);
    });
}

function addToHistory(data) {
    loadHistory(); // Reload history to show new prediction
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all history?')) {
        return;
    }
    
    try {
        const response = await fetch('/clear-history', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('History cleared successfully', 'success');
            loadHistory();
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        showNotification('Failed to clear history', 'error');
    }
}

// ===================================
// Download Report
// ===================================
function downloadReport() {
    const diseaseName = document.getElementById('diseaseName').textContent;
    const confidence = document.getElementById('confidenceBadge').textContent;
    const description = document.getElementById('diseaseDescription').textContent;
    const treatment = document.getElementById('diseaseTreatment').textContent;
    const prevention = document.getElementById('diseasePrevention').textContent;
    const timestamp = new Date().toLocaleString();
    
    // Create report content
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
    `.trim();
    
    // Create and download file
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
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
    // Create notification element
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
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        max-width: 400px;
    `;
    
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// ===================================
// Smooth Scroll
// ===================================
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
                
                // Update active nav link
                document.querySelectorAll('.nav-link').forEach(link => {
                    link.classList.remove('active');
                });
                this.classList.add('active');
            }
        });
    });
}

// ===================================
// Additional Animations
// ===================================

// Add fadeOut animation to CSS dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(20px);
        }
    }
`;
document.head.appendChild(style);

// ===================================
// Console Welcome Message
// ===================================
console.log('%c🌿 Plant Disease Detection System', 'color: #10b981; font-size: 20px; font-weight: bold;');
console.log('%cPowered by Deep Learning & TensorFlow', 'color: #6b7280; font-size: 14px;');
console.log('%cBE Computer Engineering Project', 'color: #6b7280; font-size: 12px;');
