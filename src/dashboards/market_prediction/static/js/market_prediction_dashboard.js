async function fetchPredictions() {
  const symbol = document.getElementById('symbolSelect')?.value || 'BTCUSDT';
  try {
    const resp = await fetch(`/api/predictions?symbol=${symbol}`);
    const data = await resp.json();
    if (data.error) {
      console.error(data.error);
      return;
    }
    populateTable(data.predictions);
    populateSentiment(data.sentiment);
    const ts = new Date(data.generated_at || Date.now());
    document.getElementById('lastUpdated').innerText = `Last updated: ${ts.toLocaleString()}`;
  } catch (err) {
    console.error('Failed to fetch predictions', err);
  }
}

function populateTable(preds) {
  const tbody = document.querySelector('#predictionTable tbody');
  tbody.innerHTML = '';
  preds.forEach((p) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${p.horizon_days}</td>
      <td>${p.predicted_price.toLocaleString()}</td>
      <td>${p.pct_change}%</td>
      <td>${(p.confidence * 100).toFixed(0)}%</td>
      <td>${p.recommendation}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateSentiment(sent) {
  const div = document.getElementById('sentimentContainer');
  if (!sent || Object.keys(sent).length === 0) {
    div.innerText = 'No sentiment data available.';
    return;
  }
  div.innerHTML = `
    <p>Index Value: ${sent.index_value || 'N/A'} (${sent.classification || 'Unknown'})</p>
    <p>Normalized Score: ${(sent.sentiment_primary * 100).toFixed(1)}%</p>
    <p>Timestamp: ${new Date(sent.timestamp).toLocaleString()}</p>
  `;
}

// Initial fetch and periodic refresh every minute
fetchPredictions();
setInterval(fetchPredictions, 60 * 1000);

// Refresh immediately when symbol changes
const sel = document.getElementById('symbolSelect');
if (sel) {
  sel.addEventListener('change', fetchPredictions);
}