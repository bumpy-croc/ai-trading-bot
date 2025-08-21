class TradingDashboard {
    constructor() {
        this.socket = io();
        this.config = {};
        this.updateInterval = 5000; // 5 seconds instead of 1 hour
        this.chart = null;
        this.lastMetrics = {};
        this.currencyFormatter = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        });
        this.integerFormatter = new Intl.NumberFormat('en-US', {
            maximumFractionDigits: 0,
        });
        this.percentFormatter = new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 1,
            maximumFractionDigits: 1,
        });

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
            // Resolve per-symbol prices to avoid BTC defaulting
            await this.hydratePositionsWithPrices(positions);
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
            // Hydrate with latest prices before rendering
            this.hydratePositionsWithPrices(data).then(() => this.updatePositions(data));
        });

        this.socket.on('trades_update', (data) => {
            this.updateTrades(data);
        });

        this.socket.on('performance_update', (data) => {
            this.updatePerformanceChart(data);
        });
    }

    async hydratePositionsWithPrices(positions) {
        if (!Array.isArray(positions) || positions.length === 0) return;
        // Collect unique symbols
        const symbols = Array.from(new Set(positions.map(p => String(p.symbol || '').toUpperCase()).filter(Boolean)));
        if (symbols.length === 0) return;
        try {
            const url = `/api/prices?symbols=${encodeURIComponent(symbols.join(','))}`;
            const res = await fetch(url);
            if (!res.ok) return;
            const priceMap = await res.json();
            positions.forEach(p => {
                const s = String(p.symbol || '').toUpperCase();
                const price = priceMap[s];
                if (typeof price === 'number' && price > 0) {
                    p.current_price = price;
                    // Recompute Unrealized PnL client-side for visibility if backend had BTC price or 0
                    const entry = typeof p.entry_price === 'number' ? p.entry_price : 0;
                    const qty = typeof p.quantity === 'number' ? p.quantity : 0;
                    const side = String(p.side || '').toLowerCase();
                    if (entry > 0 && qty > 0) {
                        p.unrealized_pnl = side === 'long' ? (price - entry) * qty : (entry - price) * qty;
                    }
                }
            });
        } catch (e) {
            // Ignore fetch errors; UI will show existing values
        }
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
        Object.entries(this.config || {}).forEach(([key, metric]) => {
            if (metric && typeof metric === 'object' && 'enabled' in metric) {  // Ensure it's a metric object
                const div = document.createElement('div');
                div.className = 'form-check';
                div.innerHTML = `
                    <input class="form-check-input" type="checkbox" id="metric_${key}"
                           value="${key}" ${metric.enabled ? 'checked' : ''}>
                    <label class="form-check-label" for="metric_${key}">
                        ${key.replace(/_/g, ' ').toUpperCase()}
                    </label>
                `;
                metricsContainer.appendChild(div);
            }
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
        Object.keys(this.config).forEach(key => {
            if (key !== 'update_interval') {  // Skip non-metric keys
                const checkbox = document.getElementById(`metric_${key}`);
                const enabled = checkbox ? checkbox.checked : this.config[key].enabled;
                config.metrics[key] = {
                    ...this.config[key],
                    enabled: enabled
                };
            }
        });

        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config.metrics)  // Send only metrics
            });

            if (response.ok) {
                this.config = { ...config, ...config.metrics };  // Update local config
                this.updateInterval = config.update_interval * 1000;
                this.toggleConfigPanel();
                this.showSuccessMessage('Configuration saved successfully');
                // Reload metrics to reflect changes
                const metricsResponse = await fetch('/api/metrics');
                const metrics = await metricsResponse.json();
                this.updateMetrics(metrics);
            } else {
                throw new Error('Failed to save configuration');
            }
        } catch (error) {
            console.error('Error saving configuration:', error);
            this.showErrorMessage('Failed to save configuration');
        }
    }

    updateMetrics(data) {
        const container = document.getElementById('keyMetrics');
        if (!container) return;
        container.innerHTML = '';

        Object.entries(data).forEach(([key, value]) => {
            const metric = this.config[key];
            if (metric && metric.enabled) {
                const col = document.createElement('div');
                col.className = 'col-md-3 mb-3';
                col.innerHTML = `
                    <div class="metric-card" id="metric_${key}">
                        <div class="metric-title">${key.replace(/_/g, ' ').toUpperCase()}</div>
                        <div class="metric-value">${this.formatMetricValueOverride(key, value, metric.format)}</div>
                        <div class="metric-change"></div>
                    </div>
                `;
                container.appendChild(col);
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
            const formattedValue = this.formatMetricValueOverride(key, value, metric.format);
            valueElement.textContent = formattedValue;
        }

        // Special handling for dynamic risk metrics
        if (key === 'dynamic_risk_factor') {
            this.updateDynamicRiskDisplay(value);
        } else if (changeElement && this.lastMetrics[key] !== undefined) {
            const change = this.calculateChange(key, value);
            if (change !== null) {
                changeElement.textContent = change;
                changeElement.className = `metric-change ${change.startsWith('+') ? 'positive' : 'negative'}`;
            }
        }

        this.lastMetrics[key] = value;
    }

    updateDynamicRiskDisplay(factor) {
        // Update the main metric display
        const card = document.getElementById('metric_dynamic_risk_factor');
        if (!card) return;

        const valueElement = card.querySelector('.metric-value');
        const changeElement = card.querySelector('.metric-change');
        
        if (valueElement) {
            valueElement.textContent = `${Number(factor).toFixed(2)}x`;
        }

        // Get the reason from the other metric
        const reason = this.lastMetrics['dynamic_risk_reason'] || 'normal';
        if (changeElement) {
            changeElement.textContent = reason;
        }

        // Update the risk status indicator
        const indicator = document.getElementById('riskStatusIndicator');
        if (indicator) {
            // Remove existing status classes
            indicator.classList.remove('normal', 'active', 'critical');
            
            // Determine status based on factor
            if (factor === 1.0) {
                indicator.classList.add('normal');
                indicator.title = 'Risk management: Normal';
            } else if (factor >= 0.5) {
                indicator.classList.add('active');
                indicator.title = `Risk management: Active (${reason})`;
            } else {
                indicator.classList.add('critical');
                indicator.title = `Risk management: Critical reduction (${reason})`;
            }
        }

        // Check for significant changes and show alert
        const previousFactor = this.lastMetrics['dynamic_risk_factor'];
        if (previousFactor !== undefined && Math.abs(factor - previousFactor) > 0.1) {
            this.showDynamicRiskAlert(factor, reason, previousFactor);
        }
    }

    showDynamicRiskAlert(newFactor, reason, oldFactor) {
        // Create and show a temporary alert for dynamic risk changes
        const alertContainer = document.createElement('div');
        alertContainer.className = 'alert alert-warning alert-dismissible fade show dynamic-risk-alert';
        alertContainer.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
        
        const changeDirection = newFactor > oldFactor ? 'increased' : 'decreased';
        const changePercent = Math.abs((newFactor - oldFactor) / oldFactor * 100).toFixed(0);
        
        alertContainer.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <div>
                    <strong>Dynamic Risk Adjustment</strong><br>
                    Risk factor ${changeDirection} by ${changePercent}% to ${newFactor.toFixed(2)}x<br>
                    <small>Reason: ${reason}</small>
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        document.body.appendChild(alertContainer);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (alertContainer.parentNode) {
                alertContainer.remove();
            }
        }, 10000);
    }

    formatMetricValueOverride(key, value, format) {
        // Force specific keys to expected formats regardless of config
        const integerKeys = new Set(['active_positions_count', 'total_trades', 'failed_orders']);
        const currencyKeys = new Set(['current_balance', 'daily_pnl', 'weekly_pnl', 'total_pnl', 'position_sizes', 'total_position_value', 'available_margin', 'unrealized_pnl']);
        
        // Special handling for dynamic risk metrics
        if (key === 'dynamic_risk_factor') {
            return `${Number(value).toFixed(2)}x`;
        }
        if (key === 'dynamic_risk_reason') {
            return String(value);
        }
        if (key === 'dynamic_risk_active') {
            return value ? 'Active' : 'Normal';
        }
        
        if (integerKeys.has(key)) {
            return this.integerFormatter.format(Math.round(Number(value) || 0));
        }
        if (currencyKeys.has(key)) {
            return this.formatCurrency(Number(value) || 0);
        }
        return this.formatMetricValue(value, format);
    }

    formatMetricValue(value, format) {
        if (typeof value !== 'number') {
            if (format === 'datetime') {
                const date = new Date(value);
                if (isNaN(date.getTime())) {
                    return "Invalid Date";
                }
                const weekday = date.toLocaleDateString('en-US', { weekday: 'short' });
                const day = date.toLocaleDateString('en-US', { day: '2-digit' });
                const month = date.toLocaleDateString('en-US', { month: 'short' });
                const time = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
                return `${time} on ${weekday}, ${day} ${month}`;
            }
            return value;
        }

        switch (format) {
            case 'currency':
                return this.formatCurrency(value);
            case 'percentage':
                // Backend returns percentages in [0, 100]
                return `${this.percentFormatter.format(value)}%`;
            case 'integer':
                return this.integerFormatter.format(Math.round(value));
            case 'number':
                // falls through
            case 'decimal':
                return typeof value === 'number' ? value.toFixed(2) : '0.00';
            default:
                return value.toString();
        }
    }

    formatCurrency(value) {
        if (typeof value !== 'number') return '$0.00';
        return this.currencyFormatter.format(value);
    }

    formatQuantity(symbol, quantity) {
        const q = typeof quantity === 'number' ? quantity : 0;
        if (q === 0) return '0';
        const absQ = Math.abs(q);
        let digits = 2;
        if (absQ >= 1000) digits = 0;
        else if (absQ >= 1) digits = 2;
        else if (absQ >= 0.01) digits = 4;
        else if (absQ >= 0.0001) digits = 6;
        else digits = 8;
        return q.toFixed(digits).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
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
            tbody.innerHTML = '<tr><td colspan="8" class="text-center">No active positions</td></tr>';
            return;
        }

        tbody.innerHTML = positions.map(position => {
            const unrealizedPnl = typeof position.unrealized_pnl === 'number' ? position.unrealized_pnl : 0.0;
            const quantity = typeof position.quantity === 'number' ? position.quantity : 0;
            const entryPrice = typeof position.entry_price === 'number' ? position.entry_price : 0.0;
            const currentPrice = typeof position.current_price === 'number' ? position.current_price : 0.0;
            const mfe = typeof position.mfe === 'number' ? position.mfe : 0.0;
            const mae = typeof position.mae === 'number' ? position.mae : 0.0;
            return `
            <tr>
                <td>${position.symbol}</td>
                <td><span class="badge ${position.side === 'long' ? 'bg-success' : 'bg-danger'}">${position.side}</span></td>
                <td>${this.formatQuantity(position.symbol, quantity)}</td>
                <td>${this.formatCurrency(entryPrice)}</td>
                <td>${this.formatCurrency(currentPrice)}</td>
                <td class="${unrealizedPnl >= 0 ? 'text-success' : 'text-danger'}">
                    ${this.formatCurrency(unrealizedPnl)}
                </td>
                <td class="${mfe >= 0 ? 'text-success' : 'text-muted'}">${(mfe * 100).toFixed(2)}%</td>
                <td class="${mae <= 0 ? 'text-danger' : 'text-muted'}">${(mae * 100).toFixed(2)}%</td>
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
            const quantity = typeof trade.quantity === 'number' ? trade.quantity : 0;
            const entryPrice = typeof trade.entry_price === 'number' ? trade.entry_price : 0.0;
            const exitPrice = typeof trade.exit_price === 'number' ? trade.exit_price : 0.0;
            return `
            <tr>
                <td>${trade.symbol}</td>
                <td><span class="badge ${trade.side && trade.side.toLowerCase() === 'buy' ? 'bg-success' : 'bg-danger'}">${trade.side}</span></td>
                <td>${this.formatQuantity(trade.symbol, quantity)}</td>
                <td>${this.formatCurrency(entryPrice)}</td>
                <td>${this.formatCurrency(exitPrice)}</td>
                <td class="${pnl >= 0 ? 'text-success' : 'text-danger'}">
                    ${this.formatCurrency(pnl)}
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
                            callback: (value) => {
                                const num = typeof value === 'number' ? value : Number(value);
                                return this.formatCurrency(isNaN(num) ? 0 : num);
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
        if (!this.chart) return;
        if (!data || !data.timestamps || !data.balances || data.timestamps.length === 0) {
            this.chart.data.labels = [];
            this.chart.data.datasets[0].data = [];
            this.chart.update();
            return;
        }

        this.chart.data.labels = data.timestamps.map(ts => {
            const d = new Date(ts);
            return isNaN(d.getTime()) ? '' : d.toLocaleDateString('en-US', { month: 'short', day: '2-digit' });
        });
        this.chart.data.datasets[0].data = data.balances.map(v => (typeof v === 'number' ? v : Number(v)) || 0);
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
