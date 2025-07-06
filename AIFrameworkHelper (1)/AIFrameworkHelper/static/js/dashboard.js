// Dashboard JavaScript functionality

// Load provider status on page load
document.addEventListener('DOMContentLoaded', function() {
    loadProviderStatus();
    loadRecentMetrics();
});

// Load provider status
async function loadProviderStatus() {
    try {
        const response = await fetch('/api/providers');
        const result = await response.json();
        
        if (result.success) {
            displayProviderStatus(result.providers);
        } else {
            displayProviderError('Failed to load provider status');
        }
    } catch (error) {
        displayProviderError('Error loading provider status');
    }
}

// Display provider status
function displayProviderStatus(providers) {
    if (!providers || providers.length === 0) {
        document.getElementById('provider-status').innerHTML = `
            <div class="text-warning text-center">
                <i data-feather="alert-triangle" class="me-2"></i>
                No providers configured
            </div>
        `;
        feather.replace();
        return;
    }
    
    const statusHtml = providers.map(provider => `
        <div class="d-flex align-items-center justify-content-between mb-2">
            <div class="d-flex align-items-center">
                <div class="badge bg-success me-2"></div>
                <span class="text-capitalize">${provider}</span>
            </div>
            <span class="badge bg-outline-success">Active</span>
        </div>
    `).join('');
    
    document.getElementById('provider-status').innerHTML = statusHtml;
    feather.replace();
}

// Display provider error
function displayProviderError(message) {
    document.getElementById('provider-status').innerHTML = `
        <div class="text-danger text-center">
            <i data-feather="x-circle" class="me-2"></i>
            ${message}
        </div>
    `;
    feather.replace();
}

// Load recent metrics
async function loadRecentMetrics() {
    try {
        const response = await fetch('/api/performance?hours=1');
        const result = await response.json();
        
        if (result.success) {
            displayRecentMetrics(result.metrics);
        } else {
            displayMetricsError('Failed to load recent metrics');
        }
    } catch (error) {
        displayMetricsError('Error loading recent metrics');
    }
}

// Display recent metrics
function displayRecentMetrics(metrics) {
    const metricsHtml = `
        <div class="row text-center">
            <div class="col-6">
                <div class="border-end">
                    <h5 class="text-primary">${metrics.total_requests}</h5>
                    <small class="text-muted">Requests</small>
                </div>
            </div>
            <div class="col-6">
                <h5 class="text-success">${(metrics.success_rate * 100).toFixed(1)}%</h5>
                <small class="text-muted">Success Rate</small>
            </div>
        </div>
        <hr>
        <div class="row text-center">
            <div class="col-6">
                <div class="border-end">
                    <h6 class="text-info">${Math.round(metrics.avg_latency_ms)}ms</h6>
                    <small class="text-muted">Avg Latency</small>
                </div>
            </div>
            <div class="col-6">
                <h6 class="text-warning">${metrics.total_tokens.toLocaleString()}</h6>
                <small class="text-muted">Tokens</small>
            </div>
        </div>
    `;
    
    document.getElementById('recent-metrics').innerHTML = metricsHtml;
}

// Display metrics error
function displayMetricsError(message) {
    document.getElementById('recent-metrics').innerHTML = `
        <div class="text-danger text-center">
            <i data-feather="x-circle" class="me-2"></i>
            ${message}
        </div>
    `;
    feather.replace();
}

// Refresh dashboard data
function refreshDashboard() {
    loadProviderStatus();
    loadRecentMetrics();
}

// Auto-refresh every 30 seconds
setInterval(refreshDashboard, 30000);
