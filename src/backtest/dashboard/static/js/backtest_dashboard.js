document.addEventListener('DOMContentLoaded', () => {
    const tableBody = document.querySelector('#backtestTable tbody');
    const searchInput = document.getElementById('searchInput');
    const compareBtn = document.getElementById('compareBtn');
    const compareTitle = document.getElementById('compareTitle');
    const comparisonContainer = document.getElementById('comparisonContainer');

    let backtests = [];
    const selectedFiles = new Set();

    // Fetch backtests
    fetch('/api/backtests')
        .then(r => {
            if (!r.ok) {
                throw new Error(`Failed to fetch backtests: ${r.status} ${r.statusText}`);
            }
            return r.json();
        })
        .then(data => {
            backtests = data;
            renderTable(backtests);
        })
        .catch(error => {
            console.error(error);
            tableBody.innerHTML = '<tr><td colspan="11">Failed to load backtests. Please try again later.</td></tr>';
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
                <td>${format(bt.win_rate)}</td>
                <td>${format(bt.total_return)}</td>
                <td>${format(bt.annualized_return)}</td>
                <td>${format(bt.max_drawdown)}</td>
                <td>${format(bt.sharpe_ratio)}</td>`;
            tableBody.appendChild(tr);
        });
        tableBody.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', e => {
                const file = e.target.getAttribute('data-file');
                e.target.checked ? selectedFiles.add(file) : selectedFiles.delete(file);
                compareBtn.disabled = selectedFiles.size !== 2;
            });
        });
    }

    function format(v) { return v == null ? '' : Number(v).toFixed(2); }

    // Search
    searchInput.addEventListener('input', () => {
        const t = searchInput.value.toLowerCase();
        renderTable(backtests.filter(bt => (bt.strategy || '').toLowerCase().includes(t) || (bt.symbol || '').toLowerCase().includes(t) || (bt.file || '').toLowerCase().includes(t)));
    });

    // Compare
    compareBtn.addEventListener('click', () => {
        if (selectedFiles.size !== 2) return;
        const [f, s] = [...selectedFiles];
        fetch(`/api/compare?first=${encodeURIComponent(f)}&second=${encodeURIComponent(s)}`)
            .then(r => r.json())
            .then(showComparison);
    });

    function showComparison(data) {
        compareTitle.style.display = 'block';
        const diff = data.diff || {};
        let html = '<table><thead><tr><th>Metric</th><th>First</th><th>Second</th></tr></thead><tbody>';
        Object.keys(diff).forEach(k => {
            html += `<tr><td>${k}</td><td>${format(diff[k].first)}</td><td>${format(diff[k].second)}</td></tr>`;
        });
        comparisonContainer.innerHTML = html + '</tbody></table>';
    }
});