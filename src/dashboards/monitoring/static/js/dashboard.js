class TradingDashboard {
    constructor() {
        this.socket = io();
        this.config = {};
        this.updateInterval = 5000; // 5 seconds instead of 1 hour
        this.chart = null;
        this.lastMetrics = {};
        this.positionsTableColumnCount = 11; // Number of columns in positions table
        this.tradesTableColumnCount = 6; // Number of columns in trades table
        this.showCloseTargetPercentages = false; // Toggle for percentage display in close targets
        
        // * Configuration options for Close Target display
        this.closeTargetConfig = {
            showTakeProfit: true,
            showStopLoss: true,
            showTrailingStop: true,
            showPartialExits: true,
            showRiskReward: true,
            showTimeInPosition: true,
            showMultipleTargets: true,
            maxTooltipLength: 200,
            refreshInterval: 5000
        };
        
        // * Performance optimization caches
        this.closeTargetCache = new Map(); // Cache for close target calculations
        this.priceCache = new Map(); // Cache for real-time prices
        this.cacheExpiry = 5000; // 5 seconds cache expiry
        this.debounceTimeout = null; // Debounce timer for rapid updates
        this.lastUpdateTime = 0; // Track last update to prevent excessive calculations
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
        this.loadCloseTargetConfig(); // Load close target configuration
        this.initializeChart();
        this.hideLoading();
        this.updateToggleButton(); // Initialize toggle button state

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
            // * Enhanced position update with live trading integration
            this._handlePositionUpdate(data);
        });

        this.socket.on('order_update', (data) => {
            // * Handle order updates that might affect close targets
            this._handleOrderUpdate(data);
        });

        this.socket.on('position_status_change', (data) => {
            // * Handle position status changes (e.g., partial exits, trailing stops)
            this._handlePositionStatusChange(data);
        });

        this.socket.on('trades_update', (data) => {
            this.updateTrades(data);
        });

        this.socket.on('performance_update', (data) => {
            this.updatePerformanceChart(data);
        });
    }

    async _handlePositionUpdate(data) {
        // * Enhanced position update with live trading integration
        try {
            // * Hydrate with latest prices before rendering
            await this.hydratePositionsWithPrices(data);
            
            // * Check for position changes that affect close targets
            this._detectPositionChanges(data);
            
            // * Update positions display
            this.updatePositions(data);
            
            // * Update any pending order indicators
            this._updatePendingOrderIndicators(data);
            
        } catch (error) {
            console.error('Error handling position update:', error);
            // * Fallback to basic update
            this.updatePositions(data);
        }
    }

    _handleOrderUpdate(data) {
        // * Handle order updates that might affect close targets
        if (data && data.position_id) {
            // * Clear cache for affected position
            const position = this._findPositionById(data.position_id);
            if (position) {
                this._clearPositionCache(position);
            }
            
            // * Refresh positions to show updated close targets
            this.loadInitialData();
        }
    }

    _handlePositionStatusChange(data) {
        // * Handle position status changes (e.g., partial exits, trailing stops)
        if (data && data.position_id) {
            // * Clear cache for affected position
            const position = this._findPositionById(data.position_id);
            if (position) {
                this._clearPositionCache(position);
                
                // * Show notification for significant changes
                this._showPositionChangeNotification(data);
            }
            
            // * Refresh positions to show updated close targets
            this.loadInitialData();
        }
    }

    _detectPositionChanges(newPositions) {
        // * Detect significant position changes that affect close targets
        if (!this.lastPositions) {
            this.lastPositions = new Map();
            return;
        }
        
        newPositions.forEach(position => {
            const key = `${position.symbol}_${position.side}_${position.entry_price}`;
            const lastPosition = this.lastPositions.get(key);
            
            if (lastPosition) {
                // * Check for changes that affect close targets
                const changes = [];
                
                if (lastPosition.stop_loss !== position.stop_loss) {
                    changes.push('Stop loss updated');
                }
                if (lastPosition.take_profit !== position.take_profit) {
                    changes.push('Take profit updated');
                }
                if (lastPosition.trailing_stop_activated !== position.trailing_stop_activated) {
                    changes.push('Trailing stop ' + (position.trailing_stop_activated ? 'activated' : 'deactivated'));
                }
                if (lastPosition.breakeven_triggered !== position.breakeven_triggered) {
                    changes.push('Breakeven ' + (position.breakeven_triggered ? 'triggered' : 'reset'));
                }
                if (lastPosition.partial_exits_taken !== position.partial_exits_taken) {
                    changes.push('Partial exit taken');
                }
                
                if (changes.length > 0) {
                    console.log(`Position ${position.symbol} changes:`, changes);
                    // * Clear cache for this position
                    this._clearPositionCache(position);
                }
            }
            
            // * Update last positions
            this.lastPositions.set(key, { ...position });
        });
    }

    _findPositionById(positionId) {
        // * Find position by ID in current positions
        const tbody = document.querySelector('#positionsTable tbody');
        if (!tbody) return null;
        
        // * This is a simplified implementation - in a real scenario,
        // * you'd maintain a positions map or query the backend
        return null;
    }

    _updatePendingOrderIndicators(positions) {
        // * Update visual indicators for pending orders
        positions.forEach(position => {
            if (position.pending_orders && position.pending_orders.length > 0) {
                // * Add visual indicator for positions with pending orders
                this._addPendingOrderIndicator(position);
            }
        });
    }

    _addPendingOrderIndicator(position) {
        // * Add visual indicator for positions with pending orders
        const indicator = document.querySelector(`[data-position-id="${position.id}"] .pending-orders-indicator`);
        if (!indicator) {
            // * Add indicator if not present
            const cell = document.querySelector(`[data-position-id="${position.id}"] .close-target-cell`);
            if (cell) {
                const pendingBadge = document.createElement('span');
                pendingBadge.className = 'badge bg-warning ms-1';
                pendingBadge.textContent = 'P';
                pendingBadge.title = 'Pending orders';
                cell.appendChild(pendingBadge);
            }
        }
    }

    _showPositionChangeNotification(data) {
        // * Show notification for significant position changes
        if (data.change_type === 'trailing_stop_activated') {
            this._showToast('Trailing stop activated', 'info');
        } else if (data.change_type === 'breakeven_triggered') {
            this._showToast('Breakeven triggered', 'success');
        } else if (data.change_type === 'partial_exit') {
            this._showToast('Partial exit executed', 'warning');
        }
    }

    _showToast(message, type = 'info') {
        // * Show toast notification
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        // * Add to toast container or create one
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            document.body.appendChild(container);
        }
        
        container.appendChild(toast);
        
        // * Show toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // * Remove after hiding
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    async hydratePositionsWithPrices(positions) {
        if (!Array.isArray(positions) || positions.length === 0) return;
        
        // * Performance optimization: Check cache first
        const symbols = Array.from(new Set(positions.map(p => String(p.symbol || '').toUpperCase()).filter(Boolean)));
        if (symbols.length === 0) return;
        
        const now = Date.now();
        const cachedPrices = new Map();
        const symbolsToFetch = [];
        
        // * Check which symbols need fresh prices
        for (const symbol of symbols) {
            const cached = this.priceCache.get(symbol);
            if (cached && (now - cached.timestamp) < this.cacheExpiry) {
                cachedPrices.set(symbol, cached.price);
            } else {
                symbolsToFetch.push(symbol);
            }
        }
        
        // * Fetch only symbols that need updates
        if (symbolsToFetch.length > 0) {
            try {
                const url = `/api/prices?symbols=${encodeURIComponent(symbolsToFetch.join(','))}`;
                const res = await fetch(url);
                if (res.ok) {
                    const priceMap = await res.json();
                    
                    // * Cache the prices and update cachedPrices map
                    for (const [symbol, price] of Object.entries(priceMap)) {
                        if (typeof price === 'number' && price > 0) {
                            this.priceCache.set(symbol, {
                                price: price,
                                timestamp: now
                            });
                            cachedPrices.set(symbol, price);
                        }
                    }
                }
            } catch (e) {
                console.warn('Failed to fetch real-time prices:', e);
                // * Use cached prices as fallback
                for (const symbol of symbolsToFetch) {
                    const fallback = this.priceCache.get(symbol);
                    if (fallback) {
                        cachedPrices.set(symbol, fallback.price);
                    }
                }
            }
        }
        
        // * Update positions with current prices and clear close target cache for updated positions
        positions.forEach(p => {
            const s = String(p.symbol || '').toUpperCase();
            const price = cachedPrices.get(s);
            if (typeof price === 'number' && price > 0) {
                const oldPrice = p.current_price;
                p.current_price = price;
                
                // * Clear cache for this position if price changed significantly
                if (oldPrice && Math.abs(price - oldPrice) / oldPrice > 0.001) { // 0.1% change
                    this._clearPositionCache(p);
                }
                
                // * Recompute Unrealized PnL client-side for visibility
                const entry = typeof p.entry_price === 'number' ? p.entry_price : 0;
                const qty = typeof p.quantity === 'number' ? p.quantity : 0;
                const side = String(p.side || '').toLowerCase();
                if (entry > 0 && qty > 0) {
                    p.unrealized_pnl = side === 'long' ? (price - entry) * qty : (entry - price) * qty;
                }
            }
        });
    }

    _clearPositionCache(position) {
        // * Clear cache entries for a specific position
        const positionKey = `${position.symbol}_${position.side}_${position.entry_price}`;
        for (const [key, value] of this.closeTargetCache.entries()) {
            if (key.startsWith(positionKey)) {
                this.closeTargetCache.delete(key);
            }
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

        // * Toggle close target display format
        const toggleButton = document.getElementById('toggleCloseTargetDisplay');
        if (toggleButton) {
            toggleButton.addEventListener('click', () => {
                this.showCloseTargetPercentages = !this.showCloseTargetPercentages;
                this.updateToggleButton();
                // Refresh positions to update display format
                this.loadInitialData();
            });
        }

        // * Close target configuration modal
        const configButton = document.getElementById('closeTargetConfigBtn');
        if (configButton) {
            configButton.addEventListener('click', () => {
                this.showCloseTargetConfigModal();
            });
        }

        const saveConfigButton = document.getElementById('saveCloseTargetConfig');
        if (saveConfigButton) {
            saveConfigButton.addEventListener('click', () => {
                this.saveCloseTargetConfig();
            });
        }
    }

    updateToggleButton() {
        // * Update toggle button appearance and tooltip
        const toggleButton = document.getElementById('toggleCloseTargetDisplay');
        if (toggleButton) {
            const icon = toggleButton.querySelector('i');
            if (this.showCloseTargetPercentages) {
                icon.className = 'fas fa-dollar-sign';
                toggleButton.title = 'Switch to currency display';
            } else {
                icon.className = 'fas fa-percentage';
                toggleButton.title = 'Switch to percentage display';
            }
        }
    }

    showCloseTargetConfigModal() {
        // * Populate modal with current configuration
        document.getElementById('showTakeProfit').checked = this.closeTargetConfig.showTakeProfit;
        document.getElementById('showStopLoss').checked = this.closeTargetConfig.showStopLoss;
        document.getElementById('showTrailingStop').checked = this.closeTargetConfig.showTrailingStop;
        document.getElementById('showPartialExits').checked = this.closeTargetConfig.showPartialExits;
        document.getElementById('showRiskReward').checked = this.closeTargetConfig.showRiskReward;
        document.getElementById('showTimeInPosition').checked = this.closeTargetConfig.showTimeInPosition;
        document.getElementById('showMultipleTargets').checked = this.closeTargetConfig.showMultipleTargets;
        document.getElementById('maxTooltipLength').value = this.closeTargetConfig.maxTooltipLength;
        document.getElementById('refreshInterval').value = this.closeTargetConfig.refreshInterval;
        
        // * Show modal
        const modal = new bootstrap.Modal(document.getElementById('closeTargetConfigModal'));
        modal.show();
    }

    saveCloseTargetConfig() {
        // * Save configuration from modal
        this.closeTargetConfig.showTakeProfit = document.getElementById('showTakeProfit').checked;
        this.closeTargetConfig.showStopLoss = document.getElementById('showStopLoss').checked;
        this.closeTargetConfig.showTrailingStop = document.getElementById('showTrailingStop').checked;
        this.closeTargetConfig.showPartialExits = document.getElementById('showPartialExits').checked;
        this.closeTargetConfig.showRiskReward = document.getElementById('showRiskReward').checked;
        this.closeTargetConfig.showTimeInPosition = document.getElementById('showTimeInPosition').checked;
        this.closeTargetConfig.showMultipleTargets = document.getElementById('showMultipleTargets').checked;
        this.closeTargetConfig.maxTooltipLength = parseInt(document.getElementById('maxTooltipLength').value);
        this.closeTargetConfig.refreshInterval = parseInt(document.getElementById('refreshInterval').value);
        
        // * Update refresh interval
        this.updateInterval = this.closeTargetConfig.refreshInterval;
        
        // * Clear cache to force recalculation with new settings
        this.closeTargetCache.clear();
        
        // * Hide modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('closeTargetConfigModal'));
        modal.hide();
        
        // * Refresh positions to apply new configuration
        this.loadInitialData();
        
        // * Save to localStorage
        localStorage.setItem('closeTargetConfig', JSON.stringify(this.closeTargetConfig));
    }

    loadCloseTargetConfig() {
        // * Load configuration from localStorage
        const saved = localStorage.getItem('closeTargetConfig');
        if (saved) {
            try {
                const config = JSON.parse(saved);
                this.closeTargetConfig = { ...this.closeTargetConfig, ...config };
                this.updateInterval = this.closeTargetConfig.refreshInterval;
            } catch (e) {
                console.warn('Failed to load close target config:', e);
            }
        }
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

    calculateCloseTarget(position) {
        // * Performance optimization: Check cache first
        const cacheKey = this._generateCacheKey(position);
        const cached = this._getCachedCloseTarget(cacheKey);
        if (cached) {
            return cached;
        }

        // * Calculate how much more profit/loss is needed to hit take profit or stop loss
        const currentPrice = this._safeNumber(position.current_price);
        const entryPrice = this._safeNumber(position.entry_price);
        const quantity = this._safeNumber(position.quantity);
        const stopLoss = this._safeNumber(position.stop_loss);
        const takeProfit = this._safeNumber(position.take_profit);
        const trailingStopPrice = this._safeNumber(position.trailing_stop_price);
        const trailingStopActivated = position.trailing_stop_activated;
        const breakevenTriggered = position.breakeven_triggered;
        const side = position.side?.toLowerCase();

        // * Validate required data with enhanced error handling
        const validationResult = this._validatePositionData(currentPrice, entryPrice, quantity, side, position);
        if (!validationResult.isValid) {
            return { 
                text: '-', 
                type: 'neutral', 
                tooltip: validationResult.error,
                error: true
            };
        }

        // * Calculate P&L for a given price
        const calculatePnL = (price) => {
            return side === 'long' 
                ? (price - entryPrice) * quantity
                : (entryPrice - price) * quantity;
        };

        const currentPnL = calculatePnL(currentPrice);

        // * Find valid targets with enhanced logic
        const targets = [];
        
        // * Take Profit target (if enabled)
        if (this.closeTargetConfig.showTakeProfit && takeProfit && this._isValidTarget(side, takeProfit, entryPrice, 'TP')) {
            targets.push({
                price: takeProfit,
                type: 'TP',
                pnl: calculatePnL(takeProfit),
                distance: Math.abs(currentPrice - takeProfit),
                isActive: true
            });
        }
        
        // * Stop Loss target (consider trailing stop, if enabled)
        if (this.closeTargetConfig.showStopLoss || this.closeTargetConfig.showTrailingStop) {
            const effectiveStopLoss = trailingStopActivated && trailingStopPrice ? trailingStopPrice : stopLoss;
            const shouldShowStopLoss = !trailingStopActivated && this.closeTargetConfig.showStopLoss;
            const shouldShowTrailingStop = trailingStopActivated && this.closeTargetConfig.showTrailingStop;
            
            if (effectiveStopLoss && (shouldShowStopLoss || shouldShowTrailingStop) && this._isValidTarget(side, effectiveStopLoss, entryPrice, 'SL')) {
                targets.push({
                    price: effectiveStopLoss,
                    type: trailingStopActivated ? 'TS' : 'SL', // TS = Trailing Stop
                    pnl: calculatePnL(effectiveStopLoss),
                    distance: Math.abs(currentPrice - effectiveStopLoss),
                    isActive: true,
                    isTrailing: trailingStopActivated,
                    isBreakeven: breakevenTriggered
                });
            }
        }

        // * Partial exit targets (if available and enabled)
        if (this.closeTargetConfig.showPartialExits) {
            const partialTargets = this._calculatePartialExitTargets(position, currentPrice, calculatePnL);
            targets.push(...partialTargets);
        }

        if (targets.length === 0) {
            return { text: '-', type: 'neutral', tooltip: 'No valid targets set' };
        }

        // * Choose closest target
        const closestTarget = targets.reduce((closest, target) => 
            target.distance < closest.distance ? target : closest
        );

        const remainingPnL = closestTarget.pnl - currentPnL;
        const remainingPercent = this._calculatePercentage(remainingPnL, currentPnL);

        // * Format display with enhanced information
        const isPositive = remainingPnL > 0;
        const sign = isPositive ? '+' : '';
        
        let text;
        if (this.showCloseTargetPercentages && Math.abs(remainingPercent) < 1000) {
            // Show percentage when it's reasonable (< 1000%)
            text = `${sign}${remainingPercent.toFixed(1)}% (${closestTarget.type})`;
        } else {
            // Default to currency display
            text = `${sign}${this.formatCurrency(remainingPnL)} (${closestTarget.type})`;
        }
        
        // * Enhanced type determination
        let type = 'neutral';
        if (closestTarget.type === 'TP' || closestTarget.type === 'PE') {
            type = 'success';
        } else if (closestTarget.type === 'SL' || closestTarget.type === 'TS') {
            type = 'danger';
        }
        
        // * Enhanced tooltip with additional context
        const tooltip = this._buildCloseTargetTooltip({
            target: closestTarget,
            remainingPnL,
            remainingPercent,
            currentPnL,
            currentPrice,
            entryPrice,
            position
        });

        const result = { text, type, tooltip, remainingPercent, targetCount: targets.length };
        
        // * Cache the result for performance
        this._cacheCloseTarget(cacheKey, result);
        
        return result;
    }

    _safeNumber(value) {
        // * Safely convert value to number, handling null/undefined
        if (value === null || value === undefined) return null;
        const num = typeof value === 'number' ? value : parseFloat(value);
        return isNaN(num) ? null : num;
    }

    _validatePositionData(currentPrice, entryPrice, quantity, side, position) {
        // * Enhanced validation with detailed error messages
        const errors = [];
        
        if (!currentPrice || currentPrice <= 0) {
            errors.push('Invalid current price');
        }
        
        if (!entryPrice || entryPrice <= 0) {
            errors.push('Invalid entry price');
        }
        
        if (!quantity || quantity <= 0) {
            errors.push('Invalid quantity');
        }
        
        if (side !== 'long' && side !== 'short') {
            errors.push('Invalid position side');
        }
        
        // * Additional validation for partial exit data
        if (position.partial_exits_taken > 0) {
            if (!position.original_size || !position.current_size) {
                errors.push('Incomplete partial exit data');
            }
        }
        
        // * Check for stale data
        if (position.entry_time) {
            const entryTime = new Date(position.entry_time);
            const ageHours = (Date.now() - entryTime.getTime()) / (1000 * 60 * 60);
            if (ageHours > 24 * 7) { // 7 days
                errors.push('Position data may be stale');
            }
        }
        
        return {
            isValid: errors.length === 0,
            error: errors.length > 0 ? errors.join(', ') : null
        };
    }

    _isValidPositionData(currentPrice, entryPrice, quantity, side) {
        // * Legacy method for backward compatibility
        return this._validatePositionData(currentPrice, entryPrice, quantity, side, {}).isValid;
    }

    _isValidTarget(side, targetPrice, entryPrice, targetType) {
        // * Validate that target price makes sense for the position side
        if (targetType === 'TP') {
            return (side === 'long' && targetPrice > entryPrice) || 
                   (side === 'short' && targetPrice < entryPrice);
        } else { // SL
            return (side === 'long' && targetPrice < entryPrice) || 
                   (side === 'short' && targetPrice > entryPrice);
        }
    }

    _calculatePercentage(remainingPnL, currentPnL) {
        // * Calculate percentage change, handling edge cases
        if (currentPnL === 0) return remainingPnL > 0 ? 100 : -100;
        return (remainingPnL / Math.abs(currentPnL)) * 100;
    }

    _calculatePartialExitTargets(position, currentPrice, calculatePnL) {
        // * Calculate partial exit targets if position has partial exit configuration
        const targets = [];
        
        // * Check if position has partial exit data
        const partialExitsTaken = position.partial_exits_taken || 0;
        const originalSize = position.original_size;
        const currentSize = position.current_size;
        
        // * If we have partial exit information, show next potential target
        if (partialExitsTaken > 0 && originalSize && currentSize) {
            const sizeReduction = (originalSize - currentSize) / originalSize;
            const nextTargetPercent = sizeReduction + 0.25; // Assume 25% increments
            
            if (nextTargetPercent < 1.0) {
                // * Estimate next partial exit target (this would need actual strategy data)
                const estimatedTargetPrice = this._estimatePartialExitPrice(position, nextTargetPercent);
                if (estimatedTargetPrice) {
                    targets.push({
                        price: estimatedTargetPrice,
                        type: 'PE', // Partial Exit
                        pnl: calculatePnL(estimatedTargetPrice),
                        distance: Math.abs(currentPrice - estimatedTargetPrice),
                        isActive: false, // Not yet active
                        isPartial: true,
                        targetLevel: partialExitsTaken + 1
                    });
                }
            }
        }
        
        return targets;
    }

    _estimatePartialExitPrice(position, targetPercent) {
        // * Estimate partial exit price based on position data
        // * This is a simplified estimation - in reality, this would come from strategy configuration
        const entryPrice = this._safeNumber(position.entry_price);
        const side = position.side?.toLowerCase();
        
        if (!entryPrice || !side) return null;
        
        // * Simple estimation: assume 2% profit per 25% exit
        const profitPerExit = 0.02; // 2%
        const targetProfit = targetPercent * profitPerExit;
        
        if (side === 'long') {
            return entryPrice * (1 + targetProfit);
        } else {
            return entryPrice * (1 - targetProfit);
        }
    }

    _buildCloseTargetTooltip(data) {
        // * Build informative tooltip with enhanced context
        const { target, remainingPnL, remainingPercent, currentPnL, currentPrice, entryPrice, position } = data;
        
        let targetLabel;
        switch (target.type) {
            case 'TP': targetLabel = 'Take Profit'; break;
            case 'SL': targetLabel = 'Stop Loss'; break;
            case 'TS': targetLabel = 'Trailing Stop'; break;
            case 'PE': targetLabel = 'Partial Exit'; break;
            default: targetLabel = 'Target';
        }
        
        const direction = remainingPnL > 0 ? 'to reach' : 'before hitting';
        let tooltip = `${targetLabel} at ${this.formatCurrency(target.price)} ` +
                     `(${direction} target: ${this.formatCurrency(remainingPnL)}, ` +
                     `${remainingPercent.toFixed(1)}% from current P&L)`;
        
        // * Add additional context
        if (target.isTrailing) {
            tooltip += `\n🔄 Trailing stop is active`;
        }
        
        if (target.isBreakeven) {
            tooltip += `\n✅ Breakeven triggered`;
        }
        
        if (target.isPartial) {
            tooltip += `\n📊 Partial exit target (${target.targetLevel} of series)`;
        }
        
        // * Add time in position (if enabled)
        if (this.closeTargetConfig.showTimeInPosition && position.entry_time) {
            const entryTime = new Date(position.entry_time);
            const timeInPosition = this._formatTimeInPosition(entryTime);
            tooltip += `\n⏰ Time in position: ${timeInPosition}`;
        }
        
        // * Add risk-reward ratio if we have both TP and SL (if enabled)
        if (this.closeTargetConfig.showRiskReward) {
            const riskReward = this._calculateRiskRewardRatio(position);
            if (riskReward) {
                tooltip += `\n⚖️ Risk:Reward = 1:${riskReward.toFixed(1)}`;
            }
        }
        
        // * Truncate tooltip if too long
        if (tooltip.length > this.closeTargetConfig.maxTooltipLength) {
            tooltip = tooltip.substring(0, this.closeTargetConfig.maxTooltipLength - 3) + '...';
        }
        
        return tooltip;
    }

    _formatTimeInPosition(entryTime) {
        // * Format time in position in a human-readable way
        const now = new Date();
        const diffMs = now - entryTime;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffHours / 24);
        
        if (diffDays > 0) {
            return `${diffDays}d ${diffHours % 24}h`;
        } else if (diffHours > 0) {
            return `${diffHours}h`;
        } else {
            const diffMinutes = Math.floor(diffMs / (1000 * 60));
            return `${diffMinutes}m`;
        }
    }

    _calculateRiskRewardRatio(position) {
        // * Calculate risk-reward ratio if both TP and SL are set
        const takeProfit = this._safeNumber(position.take_profit);
        const stopLoss = this._safeNumber(position.stop_loss);
        const entryPrice = this._safeNumber(position.entry_price);
        const side = position.side?.toLowerCase();
        
        if (!takeProfit || !stopLoss || !entryPrice || !side) return null;
        
        let risk, reward;
        if (side === 'long') {
            risk = entryPrice - stopLoss;
            reward = takeProfit - entryPrice;
        } else {
            risk = stopLoss - entryPrice;
            reward = entryPrice - takeProfit;
        }
        
        if (risk <= 0 || reward <= 0) return null;
        
        return reward / risk;
    }

    _generateCacheKey(position) {
        // * Generate a cache key based on position data that affects close target calculation
        const key = `${position.symbol}_${position.side}_${position.entry_price}_${position.current_price}_${position.quantity}_${position.stop_loss}_${position.take_profit}_${position.trailing_stop_price}_${position.trailing_stop_activated}_${position.breakeven_triggered}_${this.showCloseTargetPercentages}`;
        return key;
    }

    _getCachedCloseTarget(cacheKey) {
        // * Get cached close target result if valid
        const cached = this.closeTargetCache.get(cacheKey);
        if (cached && (Date.now() - cached.timestamp) < this.cacheExpiry) {
            return cached.result;
        }
        return null;
    }

    _cacheCloseTarget(cacheKey, result) {
        // * Cache close target result with timestamp
        this.closeTargetCache.set(cacheKey, {
            result: result,
            timestamp: Date.now()
        });
        
        // * Clean up old cache entries periodically
        if (this.closeTargetCache.size > 100) {
            this._cleanupCache();
        }
    }

    _cleanupCache() {
        // * Remove expired cache entries
        const now = Date.now();
        for (const [key, value] of this.closeTargetCache.entries()) {
            if ((now - value.timestamp) > this.cacheExpiry) {
                this.closeTargetCache.delete(key);
            }
        }
    }

    updatePositions(positions) {
        // * Performance optimization: Debounce rapid updates
        if (this.debounceTimeout) {
            clearTimeout(this.debounceTimeout);
        }
        
        this.debounceTimeout = setTimeout(() => {
            this._updatePositionsInternal(positions);
        }, 100); // 100ms debounce
    }

    _updatePositionsInternal(positions) {
        const tbody = document.querySelector('#positionsTable tbody');
        if (!tbody) return;

        // * Show loading state while processing
        this._showPositionsLoading();

        // * Use requestAnimationFrame for smooth updates
        requestAnimationFrame(() => {
            this._renderPositions(positions);
        });
    }

    _showPositionsLoading() {
        const tbody = document.querySelector('#positionsTable tbody');
        if (!tbody) return;
        
        tbody.innerHTML = `
            <tr>
                <td colspan="${this.positionsTableColumnCount}" class="text-center py-3">
                    <div class="d-flex align-items-center justify-content-center">
                        <div class="spinner-border spinner-border-sm me-2" role="status" aria-label="Loading positions">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <span>Updating positions...</span>
                    </div>
                </td>
            </tr>
        `;
    }

    _renderPositions(positions) {
        const tbody = document.querySelector('#positionsTable tbody');
        if (!tbody) return;

        if (!positions || positions.length === 0) {
            // * Enhanced no positions message with debugging help
            tbody.innerHTML = `
                <tr>
                    <td colspan="${this.positionsTableColumnCount}" class="text-center">
                        <div class="no-positions-message">
                            <i class="fas fa-info-circle text-muted me-2"></i>
                            No active positions found
                            <div class="mt-2">
                                <small class="text-muted">
                                    Active positions have "OPEN" status. 
                                    <a href="/api/debug/positions" target="_blank" class="text-primary">
                                        Debug positions
                                    </a>
                                </small>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = positions.map(position => {
            const unrealizedPnl = typeof position.unrealized_pnl === 'number' ? position.unrealized_pnl : 0.0;
            const quantity = typeof position.quantity === 'number' ? position.quantity : 0;
            const entryPrice = typeof position.entry_price === 'number' ? position.entry_price : 0.0;
            const currentPrice = typeof position.current_price === 'number' ? position.current_price : 0.0;
            const trailSL = position.trailing_stop_price ? this.formatCurrency(position.trailing_stop_price) : '-';
            const beBadge = position.breakeven_triggered ? '<span class="badge bg-info">BE</span>' : '';
            const mfe = typeof position.mfe === 'number' ? position.mfe : 0.0;
            const mae = typeof position.mae === 'number' ? position.mae : 0.0;
            const closeTarget = this.calculateCloseTarget(position);
            return `
            <tr role="row" aria-label="Position ${position.symbol} ${position.side}">
                <td role="cell" aria-label="Symbol">${position.symbol}</td>
                <td role="cell" aria-label="Position side">
                    <span class="badge ${position.side === 'long' ? 'bg-success' : 'bg-danger'}" 
                          aria-label="${position.side === 'long' ? 'Long position' : 'Short position'}">
                        ${position.side}
                    </span>
                </td>
                <td role="cell" aria-label="Position size">${this.formatQuantity(position.symbol, quantity)}</td>
                <td role="cell" aria-label="Entry price">${this.formatCurrency(entryPrice)}</td>
                <td role="cell" aria-label="Current price">${this.formatCurrency(currentPrice)}</td>
                <td role="cell" class="${unrealizedPnl >= 0 ? 'text-success' : 'text-danger'}" 
                    aria-label="Unrealized P&L: ${this.formatCurrency(unrealizedPnl)}">
                    ${this.formatCurrency(unrealizedPnl)}
                </td>
                <td role="cell" aria-label="Trailing stop loss">${trailSL}</td>
                <td role="cell" aria-label="Breakeven status">${beBadge}</td>
                <td role="cell" class="${mfe >= 0 ? 'text-success' : 'text-muted'}" 
                    aria-label="Maximum favorable excursion: ${(mfe * 100).toFixed(2)}%">
                    ${(mfe * 100).toFixed(2)}%
                </td>
                <td role="cell" class="${mae <= 0 ? 'text-danger' : 'text-muted'}" 
                    aria-label="Maximum adverse excursion: ${(mae * 100).toFixed(2)}%">
                    ${(mae * 100).toFixed(2)}%
                </td>
                <td role="cell" class="${closeTarget.type === 'success' ? 'text-success' : closeTarget.type === 'danger' ? 'text-danger' : closeTarget.error ? 'text-warning' : 'text-muted'}" 
                    title="${closeTarget.tooltip || ''}"
                    aria-label="Close target: ${closeTarget.text}">
                    ${closeTarget.error ? '<i class="fas fa-exclamation-triangle me-1" aria-hidden="true"></i>' : ''}
                    ${closeTarget.text}
                    ${this.closeTargetConfig.showMultipleTargets && closeTarget.targetCount > 1 ? '<span class="badge bg-info ms-1" title="Multiple targets available" aria-label="' + closeTarget.targetCount + ' targets available">' + closeTarget.targetCount + '</span>' : ''}
                </td>
            </tr>
            `;
        }).join('');
    }

    updateTrades(trades) {
        const tbody = document.querySelector('#tradesTable tbody');
        if (!tbody) return;

        if (!trades || trades.length === 0) {
            tbody.innerHTML = `<tr><td colspan="${this.tradesTableColumnCount}" class="text-center">No recent trades</td></tr>`;
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
