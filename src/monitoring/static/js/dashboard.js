class TradingDashboard {
    constructor() {
        this.socket = io();
        this.config = {};
        this.updateInterval = 5000; // 5 seconds instead of 1 hour
        this.chart = null;
        this.lastMetrics = {};
        
        this.init();
    }

    init() {
        this.setupSocketHandlers();
        this.setupEventListeners();
        this.loadConfiguration();
        this.initializeChart();
        this.hideLoading();
        
        // Load initial data immediately
        this.loadInitialData();
        
        // Set up periodic updates as fallback
        setInterval(() => {
            this.loadInitialData();
        }, this.updateInterval);
    }
    
    async loadInitialData() {
        try {
            // Load metrics
            const metricsResponse = await fetch('/api/metrics');
            const metrics = await metricsResponse.json();
            this.updateMetrics(metrics);
            
            // Load positions
            const positionsResponse = await fetch('/api/positions');
            const positions = await positionsResponse.json();
            this.updatePositions(positions);
            
            // Load trades
            const tradesResponse = await fetch('/api/trades?limit=10');
            const trades = await tradesResponse.json();
            this.updateTrades(trades);
            
            // Load performance data
            const performanceResponse = await fetch('/api/performance?days=7');
            const performance = await performanceResponse.json();
            this.updatePerformanceChart(performance);
            
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    setupSocketHandlers() {
        this.socket.on('connect', () => {
            console.log('Connected to monitoring dashboard');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from monitoring dashboard');
            this.showConnectionError();
        });

        this.socket.on('metrics_update', (data) => {
            this.updateMetrics(data);
        });

        this.socket.on('positions_update', (data) => {
            this.updatePositions(data);
        });

        this.socket.on('trades_update', (data) => {
            this.updateTrades(data);
        });

        this.socket.on('performance_update', (data) => {
            this.updatePerformanceChart(data);
        });
    }

    setupEventListeners() {
        // Config panel toggle
        document.getElementById('configToggle').addEventListener('click', () => {
            this.toggleConfigPanel();
        });

        // Close config panel
        document.getElementById('closeConfig').addEventListener('click', () => {
            this.toggleConfigPanel();
        });

        // Overlay click to close
        document.getElementById('overlay').addEventListener('click', () => {
            this.toggleConfigPanel();
        });

        // Config form submission
        document.getElementById('configForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveConfiguration();
        });

        // Escape key to close config
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.toggleConfigPanel();
            }
        });
    }

    async loadConfiguration() {
        try {
            const response = await fetch('/api/config');
            this.config = await response.json();
            this.populateConfigForm();
        } catch (error) {
            console.error('Error loading configuration:', error);
        }
    }

    populateConfigForm() {
        const form = document.getElementById('configForm');
        const updateInterval = form.querySelector('#updateInterval');
        const metricsContainer = form.querySelector('#metricsConfig');

        // Set update interval
        updateInterval.value = this.config.update_interval || 5;

        // Clear existing metrics checkboxes
        metricsContainer.innerHTML = '';

        // Create checkboxes for available metrics
        Object.entries(this.config.metrics || {}).forEach(([key, metric]) => {
            const div = document.createElement('div');
            div.className = 'form-check';
            div.innerHTML = `
                <input class="form-check-input" type="checkbox" id="metric_${key}" 
                       value="${key}" ${metric.enabled ? 'checked' : ''}>
                <label class="form-check-label" for="metric_${key}">
                    ${metric.display_name}
                </label>
            `;
            metricsContainer.appendChild(div);
        });
    }

    async saveConfiguration() {
        const form = document.getElementById('configForm');
        const formData = new FormData(form);
        
        const config = {
            update_interval: parseInt(formData.get('updateInterval')),
            metrics: {}
        };

        // Get selected metrics
        const checkboxes = form.querySelectorAll('.form-check-input');
        checkboxes.forEach(checkbox => {
            const metricKey = checkbox.value;
            if (this.config.metrics[metricKey]) {
                config.metrics[metricKey] = {
                    ...this.config.metrics[metricKey],
                    enabled: checkbox.checked
                };
            }
        });

        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                this.config = config;
                this.updateInterval = config.update_interval * 1000;
                this.toggleConfigPanel();
                this.showSuccessMessage('Configuration saved successfully');
            } else {
                throw new Error('Failed to save configuration');
            }
        } catch (error) {
            console.error('Error saving configuration:', error);
            this.showErrorMessage('Failed to save configuration');
        }
    }

    updateMetrics(data) {
        if (!this.config.metrics) return;

        Object.entries(data).forEach(([key, value]) => {
            const metric = this.config.metrics[key];
            if (metric && metric.enabled) {
                this.updateMetricCard(key, value, metric);
            }
        });
    }

    updateMetricCard(key, value, metric) {
        const card = document.getElementById(`metric_${key}`);
        if (!card) return;

        const valueElement = card.querySelector('.metric-value');
        const changeElement = card.querySelector('.metric-change');

        if (valueElement) {
            // Format value based on metric type
            const formattedValue = this.formatMetricValue(value, metric.format);
            valueElement.textContent = formattedValue;
        }

        if (changeElement && this.lastMetrics[key] !== undefined) {
            const change = this.calculateChange(key, value);
            if (change !== null) {
                changeElement.textContent = change;
                changeElement.className = `metric-change ${change.startsWith('+') ? 'positive' : 'negative'}`;
            }
        }

        this.lastMetrics[key] = value;
    }

    formatMetricValue(value, format) {
        if (typeof value !== 'number') return value;

        switch (format) {
            case 'currency':
                return new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD'
                }).format(value);
            case 'percentage':
                return `${(value * 100).toFixed(2)}%`;
            case 'integer':
                return Math.round(value).toLocaleString();
            case 'decimal':
                return value.toFixed(2);
            default:
                return value.toString();
        }
    }

    calculateChange(key, value) {
        // This would compare with previous values to show trends
        // For now, return null (no change indicator)
        return null;
    }

    updatePositions(positions) {
        const tbody = document.querySelector('#positionsTable tbody');
        if (!tbody) return;

        if (!positions || positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">No active positions</td></tr>';
            return;
        }

        tbody.innerHTML = positions.map(position => {
            const unrealizedPnl = typeof position.unrealized_pnl === 'number' ? position.unrealized_pnl : 0.0;
            return `
            <tr>
                <td>${position.symbol}</td>
                <td><span class="badge ${position.side === 'long' ? 'bg-success' : 'bg-danger'}">${position.side}</span></td>
                <td>${position.size}</td>
                <td>$${position.entry_price}</td>
                <td>$${position.current_price}</td>
                <td class="${unrealizedPnl >= 0 ? 'text-success' : 'text-danger'}">
                    $${unrealizedPnl.toFixed(2)}
                </td>
            </tr>
            `;
        }).join('');
    }

    updateTrades(trades) {
        const tbody = document.querySelector('#tradesTable tbody');
        if (!tbody) return;

        if (!trades || trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">No recent trades</td></tr>';
            return;
        }

        tbody.innerHTML = trades.map(trade => {
            const pnl = typeof trade.pnl === 'number' ? trade.pnl : 0.0;
            return `
            <tr>
                <td>${trade.symbol}</td>
                <td><span class="badge ${trade.side === 'buy' ? 'bg-success' : 'bg-danger'}">${trade.side}</span></td>
                <td>${trade.quantity}</td>
                <td>$${trade.entry_price}</td>
                <td>$${trade.exit_price}</td>
                <td class="${pnl >= 0 ? 'text-success' : 'text-danger'}">
                    $${pnl.toFixed(2)}
                </td>
            </tr>
            `;
        }).join('');
    }

    initializeChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: 'rgb(37, 99, 235)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    updatePerformanceChart(data) {
        if (!this.chart || !data || !data.length) return;

        const labels = data.map(point => new Date(point.timestamp).toLocaleDateString());
        const values = data.map(point => point.value);

        this.chart.data.labels = labels;
        this.chart.data.datasets[0].data = values;
        this.chart.update();
    }

    toggleConfigPanel() {
        const panel = document.getElementById('configPanel');
        const overlay = document.getElementById('overlay');
        
        panel.classList.toggle('show');
        overlay.classList.toggle('show');
    }

    showConnectionError() {
        this.showErrorMessage('Connection to server lost. Attempting to reconnect...');
    }

    showSuccessMessage(message) {
        this.showMessage(message, 'success');
    }

    showErrorMessage(message) {
        this.showMessage(message, 'error');
    }

    showMessage(message, type) {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }

    hideLoading() {
        const loading = document.querySelector('.loading-spinner');
        if (loading) {
            loading.style.display = 'none';
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard();
});

// Add toast styles dynamically
const toastStyles = `
    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        color: white;
        font-weight: 500;
        z-index: 1100;
        animation: slideIn 0.3s ease;
    }
    
    .toast-success {
        background: var(--success-color);
    }
    
    .toast-error {
        background: var(--danger-color);
    }
    
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;

// Add styles to head
const styleSheet = document.createElement('style');
styleSheet.textContent = toastStyles;
document.head.appendChild(styleSheet); 