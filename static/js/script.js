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
    historyGrid.innerHTML = '';
    
    if (!history || history.length === 0) {
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
    loadHistory(); // keep same logic
}


// ✅ UPDATED CLEAR HISTORY (REAL-TIME FIX)
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

            // 🔥 REAL-TIME UI CLEAR (NO REFRESH NEEDED)
            historyGrid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-history"></i>
                    <p>No predictions yet. Upload an image to get started!</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        showNotification('Failed to clear history', 'error');
    }
}


// ✅ NEW FEATURE: FILTER BY DATE
async function filterByDate() {
    const selectedDate = document.getElementById('historyDate').value;

    if (!selectedDate) {
        showNotification('Please select a date', 'error');
        return;
    }

    try {
        const response = await fetch('/history-by-date', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ date: selectedDate })
        });

        const data = await response.json();

        if (data.success) {
            displayHistory(data.history);
        } else {
            showNotification('No data found', 'error');
        }

    } catch (error) {
        console.error(error);
        showNotification('Error fetching data', 'error');
    }
}
