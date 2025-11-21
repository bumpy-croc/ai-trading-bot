/**
 * Enhanced Trading Dashboard - Advanced Analytics and Visualizations
 * Extends the base TradingDashboard with advanced features
 */

class EnhancedDashboard {
    constructor() {
        this.charts = {};
        this.currentTheme = localStorage.getItem('theme') || 'dark';
        this.init();
    }

    init() {
        this.applyTheme(this.currentTheme);
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        this.setupTabListeners();
        this.loadInitialData();
    }

    setupEventListeners() {
        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // Export buttons
        document.getElementById('exportPositionsBtn')?.addEventListener('click', () => this.exportData('positions'));
        document.getElementById('exportTradesBtn')?.addEventListener('click', () => this.exportData('trades'));
        document.getElementById('exportPerformanceBtn')?.addEventListener('click', () => this.exportData('performance'));

        // Time range selectors
        document.getElementById('perfTimeRange')?.addEventListener('change', (e) => this.loadPerformanceData(e.target.value));
        document.getElementById('tradeTimeRange')?.addEventListener('change', (e) => this.loadTradeAnalysisData(e.target.value));
        document.getElementById('modelTimeRange')?.addEventListener('change', (e) => this.loadModelData(e.target.value));
        document.getElementById('modelSelect')?.addEventListener('change', (e) => this.loadModelData(document.getElementById('modelTimeRange').value, e.target.value));
    }

    setupTabListeners() {
        // Load data when tab becomes active
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const target = e.target.getAttribute('data-bs-target');
                this.onTabActivated(target.replace('#', ''));
            });
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ignore if user is typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            switch(e.key.toLowerCase()) {
                case '1':
                    this.switchTab('overview-tab');
                    break;
                case '2':
                    this.switchTab('performance-tab');
                    break;
                case '3':
                    this.switchTab('trades-tab');
                    break;
                case '4':
                    this.switchTab('models-tab');
                    break;
                case '5':
                    this.switchTab('risk-tab');
                    break;
                case '6':
                    this.switchTab('system-tab');
                    break;
                case 'r':
                    this.refreshCurrentTab();
                    break;
                case 'e':
                    this.exportCurrentTab();
                    break;
                case '?':
                    this.showKeyboardShortcuts();
                    break;
            }
        });
    }

    switchTab(tabId) {
        const tab = document.getElementById(tabId);
        if (tab) {
            new bootstrap.Tab(tab).show();
        }
    }

    refreshCurrentTab() {
        const activeTab = document.querySelector('.tab-pane.active');
        if (activeTab) {
            this.onTabActivated(activeTab.id);
        }
    }

    exportCurrentTab() {
        const activeTab = document.querySelector('.tab-pane.active');
        const tabId = activeTab?.id;

        switch(tabId) {
            case 'overview':
                this.exportData('trades');
                break;
            case 'performance':
                this.exportData('performance');
                break;
            case 'trades':
                this.exportData('trades');
                break;
            default:
                console.log('Export not available for this tab');
        }
    }

    showKeyboardShortcuts() {
        alert(`Keyboard Shortcuts:

1-6: Switch between tabs
R: Refresh current tab
E: Export current tab data
?: Show this help

Navigation:
1 - Overview
2 - Performance
3 - Trade Analysis
4 - ML Models
5 - Risk
6 - System`);
    }

    onTabActivated(tabId) {
        console.log('Tab activated:', tabId);

        switch(tabId) {
            case 'performance':
                const perfDays = document.getElementById('perfTimeRange')?.value || 30;
                this.loadPerformanceData(perfDays);
                break;
            case 'trades':
                const tradeDays = document.getElementById('tradeTimeRange')?.value || 30;
                this.loadTradeAnalysisData(tradeDays);
                break;
            case 'models':
                const modelDays = document.getElementById('modelTimeRange')?.value || 30;
                const modelName = document.getElementById('modelSelect')?.value || '';
                this.loadModelData(modelDays, modelName);
                break;
            case 'risk':
                this.loadRiskData();
                break;
            case 'system':
                this.loadSystemHealth();
                break;
        }
    }

    async loadInitialData() {
        // Load model list for dropdown
        try {
            const response = await fetch('/api/models/list');
            const data = await response.json();
            this.populateModelSelect(data.models || []);
        } catch (error) {
            console.error('Error loading model list:', error);
        }
    }

    populateModelSelect(models) {
        const select = document.getElementById('modelSelect');
        if (!select) return;

        // Clear existing options except "All Models"
        while (select.options.length > 1) {
            select.remove(1);
        }

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_name;
            option.textContent = `${model.model_name} (${model.data_points} points)`;
            select.appendChild(option);
        });
    }

    async loadPerformanceData(days) {
        try {
            const response = await fetch(`/api/performance/advanced?days=${days}&window=7`);
            const data = await response.json();

            if (data.error) {
                console.error('Performance data error:', data.error);
                return;
            }

            this.updateSharpeChart(data.rolling_sharpe || []);
            this.updateDrawdownChart(data.drawdown_series || []);
            this.updateWinRateChart(data.win_rate_series || []);
            this.updateEquityDrawdownChart(days);
        } catch (error) {
            console.error('Error loading performance data:', error);
        }
    }

    async loadTradeAnalysisData(days) {
        try {
            // Load trade analysis
            const analysisResponse = await fetch(`/api/trades/analysis?days=${days}`);
            const analysis = await analysisResponse.json();

            if (!analysis.error) {
                this.updateTradeStats(analysis);
                this.updateProfitByHourChart(analysis.profit_by_hour || {});
                this.updateProfitByDowChart(analysis.profit_by_day_of_week || {});
                this.updateBestWorstTrades(analysis.best_trades || [], analysis.worst_trades || []);
            }

            // Load trade distribution
            const distResponse = await fetch(`/api/trades/distribution?days=${days}&bins=20`);
            const dist = await distResponse.json();

            if (!dist.error) {
                this.updatePnLDistributionChart(dist);
            }
        } catch (error) {
            console.error('Error loading trade analysis:', error);
        }
    }

    async loadModelData(days, modelName = '') {
        try {
            const url = `/api/models/performance?days=${days}${modelName ? `&model=${modelName}` : ''}`;
            const response = await fetch(url);
            const data = await response.json();

            if (data.error) {
                console.error('Model data error:', data.error);
                return;
            }

            if (data.series && data.series.length > 0) {
                document.getElementById('noModelData').style.display = 'none';
                this.updateModelSummary(data.summary);
                this.updateMAEChart(data.series);
                this.updateRMSEChart(data.series);
            } else {
                document.getElementById('noModelData').style.display = 'block';
            }
        } catch (error) {
            console.error('Error loading model data:', error);
            document.getElementById('noModelData').style.display = 'block';
        }
    }

    async loadRiskData() {
        try {
            const response = await fetch('/api/risk/detailed');
            const data = await response.json();

            if (!data.error) {
                this.updateRiskMetrics(data);
                this.updateConcentrationChart(data.position_concentration || {});
                this.updateRiskAdjustmentsTable(data.recent_risk_adjustments || []);
            }

            // Load correlation matrix
            const corrResponse = await fetch('/api/correlation/matrix-formatted');
            const corrData = await corrResponse.json();
            if (!corrData.error && corrData.symbols && corrData.symbols.length > 0) {
                this.updateCorrelationHeatmap(corrData);
            }
        } catch (error) {
            console.error('Error loading risk data:', error);
        }
    }

    async loadSystemHealth() {
        try {
            const response = await fetch('/api/system/health-detailed');
            const data = await response.json();

            if (!data.error) {
                this.updateSystemMetrics(data);
                this.updateRecentErrorsTable(data.recent_errors || []);
            }
        } catch (error) {
            console.error('Error loading system health:', error);
        }
    }

    // ===== CHART UPDATE METHODS =====

    updateSharpeChart(data) {
        const canvas = document.getElementById('sharpeChart');
        if (!canvas) return;

        if (this.charts.sharpe) {
            this.charts.sharpe.destroy();
        }

        this.charts.sharpe = new Chart(canvas, {
            type: 'line',
            data: {
                labels: data.map(d => new Date(d.timestamp).toLocaleDateString()),
                datasets: [{
                    label: 'Sharpe Ratio',
                    data: data.map(d => d.sharpe),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    updateDrawdownChart(data) {
        const canvas = document.getElementById('drawdownChart');
        if (!canvas) return;

        if (this.charts.drawdown) {
            this.charts.drawdown.destroy();
        }

        this.charts.drawdown = new Chart(canvas, {
            type: 'line',
            data: {
                labels: data.map(d => new Date(d.timestamp).toLocaleDateString()),
                datasets: [{
                    label: 'Drawdown %',
                    data: data.map(d => d.drawdown),
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        reverse: false,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    updateWinRateChart(data) {
        const canvas = document.getElementById('winRateChart');
        if (!canvas) return;

        if (this.charts.winRate) {
            this.charts.winRate.destroy();
        }

        this.charts.winRate = new Chart(canvas, {
            type: 'line',
            data: {
                labels: data.map(d => new Date(d.date).toLocaleDateString()),
                datasets: [{
                    label: 'Win Rate %',
                    data: data.map(d => d.win_rate),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    async updateEquityDrawdownChart(days) {
        try {
            const response = await fetch(`/api/performance?days=${days}`);
            const data = await response.json();

            const canvas = document.getElementById('equityDrawdownChart');
            if (!canvas || !data.balances) return;

            if (this.charts.equityDrawdown) {
                this.charts.equityDrawdown.destroy();
            }

            this.charts.equityDrawdown = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: data.timestamps.map(ts => new Date(ts).toLocaleDateString()),
                    datasets: [{
                        label: 'Equity',
                        data: data.balances,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        yAxisID: 'y',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error updating equity/drawdown chart:', error);
        }
    }

    updatePnLDistributionChart(data) {
        const canvas = document.getElementById('pnlDistributionChart');
        if (!canvas || !data.bins || data.bins.length < 2) return;

        if (this.charts.pnlDist) {
            this.charts.pnlDist.destroy();
        }

        // Create labels from bin edges
        const labels = [];
        for (let i = 0; i < data.bins.length - 1; i++) {
            const start = data.bins[i].toFixed(2);
            const end = data.bins[i + 1].toFixed(2);
            labels.push(`${start} to ${end}`);
        }

        this.charts.pnlDist = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of Trades',
                    data: data.counts,
                    backgroundColor: 'rgba(59, 130, 246, 0.5)',
                    borderColor: '#3b82f6',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                return `Mean: $${data.mean.toFixed(2)}\nMedian: $${data.median.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Count' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        title: { display: true, text: 'P&L Range ($)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    updateProfitByHourChart(data) {
        const canvas = document.getElementById('profitByHourChart');
        if (!canvas) return;

        if (this.charts.profitHour) {
            this.charts.profitHour.destroy();
        }

        const hours = Array.from({length: 24}, (_, i) => i);
        const values = hours.map(h => data[h] || 0);
        const colors = values.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)');

        this.charts.profitHour = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: hours.map(h => `${h}:00`),
                datasets: [{
                    label: 'Profit',
                    data: values,
                    backgroundColor: colors,
                    borderColor: values.map(v => v >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    updateProfitByDowChart(data) {
        const canvas = document.getElementById('profitByDowChart');
        if (!canvas) return;

        if (this.charts.profitDow) {
            this.charts.profitDow.destroy();
        }

        const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
        const values = [0, 1, 2, 3, 4, 5, 6].map(d => data[d] || 0);
        const colors = values.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)');

        this.charts.profitDow = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: dayNames,
                datasets: [{
                    label: 'Profit',
                    data: values,
                    backgroundColor: colors,
                    borderColor: values.map(v => v >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    updateMAEChart(data) {
        const canvas = document.getElementById('maeChart');
        if (!canvas) return;

        if (this.charts.mae) {
            this.charts.mae.destroy();
        }

        const validData = data.filter(d => d.mae !== null);

        this.charts.mae = new Chart(canvas, {
            type: 'line',
            data: {
                labels: validData.map(d => new Date(d.timestamp).toLocaleDateString()),
                datasets: [{
                    label: 'MAE',
                    data: validData.map(d => d.mae),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    updateRMSEChart(data) {
        const canvas = document.getElementById('rmseChart');
        if (!canvas) return;

        if (this.charts.rmse) {
            this.charts.rmse.destroy();
        }

        const validData = data.filter(d => d.rmse !== null);

        this.charts.rmse = new Chart(canvas, {
            type: 'line',
            data: {
                labels: validData.map(d => new Date(d.timestamp).toLocaleDateString()),
                datasets: [{
                    label: 'RMSE',
                    data: validData.map(d => d.rmse),
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    updateConcentrationChart(data) {
        const canvas = document.getElementById('concentrationChart');
        if (!canvas || Object.keys(data).length === 0) return;

        if (this.charts.concentration) {
            this.charts.concentration.destroy();
        }

        const labels = Object.keys(data);
        const values = Object.values(data);

        this.charts.concentration = new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.5)',
                        'rgba(16, 185, 129, 0.5)',
                        'rgba(239, 68, 68, 0.5)',
                        'rgba(245, 158, 11, 0.5)',
                        'rgba(139, 92, 246, 0.5)',
                    ],
                    borderColor: [
                        '#3b82f6',
                        '#10b981',
                        '#ef4444',
                        '#f59e0b',
                        '#8b5cf6',
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed.toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    // ===== UI UPDATE METHODS =====

    updateTradeStats(data) {
        document.getElementById('totalTradesMetric').textContent = data.total_trades || 0;
        document.getElementById('avgDurationMetric').textContent = data.avg_duration_hours
            ? `${data.avg_duration_hours.toFixed(1)}h`
            : '-';
        document.getElementById('medianDurationMetric').textContent = data.median_duration_hours
            ? `${data.median_duration_hours.toFixed(1)}h`
            : '-';

        if (data.best_trades && data.best_trades.length > 0) {
            const bestTrade = data.best_trades[0];
            document.getElementById('bestTradeMetric').textContent = `$${bestTrade.pnl.toFixed(2)}`;
        }
    }

    updateBestWorstTrades(best, worst) {
        const tbody = document.getElementById('bestWorstTradesTable');
        if (!tbody) return;

        let html = '';

        best.slice(0, 3).forEach(trade => {
            html += `
                <tr>
                    <td><span class="badge bg-success">Best</span></td>
                    <td>${trade.symbol}</td>
                    <td class="text-success">$${trade.pnl.toFixed(2)}</td>
                </tr>
            `;
        });

        worst.slice(0, 3).forEach(trade => {
            html += `
                <tr>
                    <td><span class="badge bg-danger">Worst</span></td>
                    <td>${trade.symbol}</td>
                    <td class="text-danger">$${trade.pnl.toFixed(2)}</td>
                </tr>
            `;
        });

        tbody.innerHTML = html || '<tr><td colspan="3" class="text-center">No trades found</td></tr>';
    }

    updateModelSummary(summary) {
        document.getElementById('avgMAEMetric').textContent = summary.avg_mae
            ? summary.avg_mae.toFixed(4)
            : '-';
        document.getElementById('avgRMSEMetric').textContent = summary.avg_rmse
            ? summary.avg_rmse.toFixed(4)
            : '-';
        document.getElementById('avgMAPEMetric').textContent = summary.avg_mape
            ? `${summary.avg_mape.toFixed(2)}%`
            : '-';
        document.getElementById('avgICMetric').textContent = summary.avg_ic
            ? summary.avg_ic.toFixed(3)
            : '-';
    }

    updateRiskMetrics(data) {
        document.getElementById('varMetric').textContent = data.var_95
            ? `$${data.var_95.toFixed(2)}`
            : '-';
        document.getElementById('currentDDMetric').textContent = data.current_drawdown
            ? `${data.current_drawdown.toFixed(2)}%`
            : '-';
        document.getElementById('maxDDMetric').textContent = data.max_drawdown
            ? `${data.max_drawdown.toFixed(2)}%`
            : '-';
        document.getElementById('exposureMetric').textContent = data.total_exposure
            ? `$${data.total_exposure.toFixed(2)}`
            : '-';
    }

    updateRiskAdjustmentsTable(adjustments) {
        const tbody = document.getElementById('riskAdjustmentsTable');
        if (!tbody) return;

        if (adjustments.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center">No risk adjustments</td></tr>';
            return;
        }

        tbody.innerHTML = adjustments.map(adj => `
            <tr>
                <td>${adj.parameter_name}</td>
                <td>${adj.adjustment_factor.toFixed(2)}x</td>
                <td>${adj.trigger_reason}</td>
                <td>${new Date(adj.timestamp).toLocaleString()}</td>
            </tr>
        `).join('');
    }

    updateCorrelationHeatmap(data) {
        const container = document.getElementById('correlationHeatmap');
        if (!container || !data.symbols || data.symbols.length === 0) return;

        // Simple text-based correlation matrix
        let html = '<div class="table-responsive"><table class="table table-sm text-center">';
        html += '<thead><tr><th></th>';
        data.symbols.forEach(s => {
            html += `<th>${s}</th>`;
        });
        html += '</tr></thead><tbody>';

        data.matrix.forEach((row, i) => {
            html += `<tr><th>${data.symbols[i]}</th>`;
            row.forEach(val => {
                const intensity = Math.abs(val);
                const color = val > 0
                    ? `rgba(16, 185, 129, ${intensity})`
                    : `rgba(239, 68, 68, ${intensity})`;
                html += `<td style="background-color: ${color}">${val.toFixed(2)}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table></div>';

        container.innerHTML = html;
    }

    updateSystemMetrics(data) {
        document.getElementById('dbLatencyMetric').textContent = data.database_latency_ms
            ? `${data.database_latency_ms.toFixed(1)}ms`
            : '-';
        document.getElementById('apiStatusMetric').textContent = data.api_status || '-';
        document.getElementById('memoryMetric').textContent = data.memory_usage_percent
            ? `${data.memory_usage_percent.toFixed(1)}%`
            : '-';
        document.getElementById('uptimeMetric').textContent = data.uptime_minutes
            ? `${Math.floor(data.uptime_minutes / 60)}h ${Math.floor(data.uptime_minutes % 60)}m`
            : '-';

        document.getElementById('errorRateValue').textContent = data.error_rate_hourly
            ? `${data.error_rate_hourly.toFixed(1)}%`
            : '0%';
    }

    updateRecentErrorsTable(errors) {
        const tbody = document.getElementById('recentErrorsTable');
        if (!tbody) return;

        if (errors.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center">No recent errors</td></tr>';
            return;
        }

        tbody.innerHTML = errors.map(err => `
            <tr>
                <td>${err.message}</td>
                <td><span class="badge bg-${err.severity === 'ERROR' ? 'danger' : 'warning'}">${err.severity}</span></td>
                <td>${new Date(err.timestamp).toLocaleString()}</td>
            </tr>
        `).join('');
    }

    // ===== EXPORT METHODS =====

    async exportData(type) {
        try {
            let url;
            switch(type) {
                case 'trades':
                    const tradeDays = document.getElementById('tradeTimeRange')?.value || 30;
                    url = `/api/export/trades?days=${tradeDays}`;
                    break;
                case 'performance':
                    const perfDays = document.getElementById('perfTimeRange')?.value || 30;
                    url = `/api/export/performance?days=${perfDays}`;
                    break;
                case 'positions':
                    url = '/api/export/positions';
                    break;
                default:
                    console.error('Unknown export type:', type);
                    return;
            }

            // Download file
            window.location.href = url;
        } catch (error) {
            console.error('Error exporting data:', error);
            alert('Failed to export data. Please try again.');
        }
    }

    // ===== THEME METHODS =====

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(this.currentTheme);
        localStorage.setItem('theme', this.currentTheme);
    }

    applyTheme(theme) {
        document.body.setAttribute('data-theme', theme);
        const icon = document.querySelector('#themeToggle i');
        if (icon) {
            icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }

        // Update chart colors if needed
        if (theme === 'light') {
            this.updateChartColorsForLight();
        }
    }

    updateChartColorsForLight() {
        // Optional: adjust chart colors for light theme
        // Implementation depends on requirements
    }
}

// Initialize enhanced dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.enhancedDashboard = new EnhancedDashboard();
});
