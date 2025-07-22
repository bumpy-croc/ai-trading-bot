document.addEventListener('DOMContentLoaded', () => {
    const tableBody = document.querySelector('#backtestTable tbody');
    const searchInput = document.getElementById('searchInput');
    const compareBtn = document.getElementById('compareBtn');
    const compareTitle = document.getElementById('compareTitle');
    const comparisonContainer = document.getElementById('comparisonContainer');

    let backtests = [];
    let selectedFiles = new Set();

    // Fetch all backtests
    fetch('/api/backtests')
        .then(res => res.json())
        .then(data => {
            backtests = data;
            renderTable(backtests);
        });

    function renderTable(list) {
        tableBody.innerHTML = '';
        list.forEach(bt => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><input type="checkbox" data-file="${bt.file}" /></td>
                <td>${bt.timestamp || ''}</td>
                <td>${bt.strategy}</td>
                <td>${bt.symbol}</td>
                <td>${bt.duration_years}</td>
                <td>${bt.total_trades}</td>
                <td>${formatNum(bt.win_rate)}</td>
                <td>${formatNum(bt.total_return)}</td>
                <td>${formatNum(bt.annualized_return)}</td>
                <td>${formatNum(bt.max_drawdown)}</td>
                <td>${formatNum(bt.sharpe_ratio)}</td>
            `;
            tableBody.appendChild(tr);
        });

        // Attach checkbox listeners
        tableBody.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', e => {
                const file = e.target.getAttribute('data-file');
                if (e.target.checked) {
                    selectedFiles.add(file);
                } else {
                    selectedFiles.delete(file);
                }
                compareBtn.disabled = selectedFiles.size !== 2;
            });
        });
    }

    function formatNum(val) {
        return val === undefined || val === null ? '' : Number(val).toFixed(2);
    }

    // Search filtering
    searchInput.addEventListener('input', () => {
        const term = searchInput.value.toLowerCase();
        const filtered = backtests.filter(bt => {
            return (
                (bt.strategy || '').toLowerCase().includes(term) ||
                (bt.symbol || '').toLowerCase().includes(term) ||
                (bt.file || '').toLowerCase().includes(term)
            );
        });
        renderTable(filtered);
    });

    // Compare button click
    compareBtn.addEventListener('click', () => {
        if (selectedFiles.size !== 2) return;
        const [first, second] = Array.from(selectedFiles);
        fetch(`/api/compare?first=${encodeURIComponent(first)}&second=${encodeURIComponent(second)}`)
            .then(res => res.json())
            .then(data => {
                showComparison(data);
            });
    });

    function showComparison(data) {
        compareTitle.style.display = 'block';
        // build comparison table
        const diff = data.diff || {};
        let html = '<table><thead><tr><th>Metric</th><th>First</th><th>Second</th></tr></thead><tbody>';
        Object.keys(diff).forEach(key => {
            html += `<tr><td>${key}</td><td>${formatNum(diff[key].first)}</td><td>${formatNum(diff[key].second)}</td></tr>`;
        });
        html += '</tbody></table>';
        comparisonContainer.innerHTML = html;
    }
});