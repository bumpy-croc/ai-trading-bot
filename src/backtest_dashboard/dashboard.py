import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from flask import Flask, jsonify, render_template, request  # type: ignore

# Reuse project logger configuration
logger = logging.getLogger(__name__)


class BacktestDashboard:
    """Simple dashboard to visualise historical backtest runs stored as JSON files."""

    def __init__(self, logs_dir: 'Union[str, Path]' = 'logs/backtest'):
        self.logs_dir = Path(logs_dir)
        # Ensure directory exists so that the UI loads even when no logs are present
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Flask application – templates/static live in a sibling directory of this file
        self.app = Flask(
            __name__,
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static')
        )
        self._setup_routes()

    # ------------------------------------------------------------------
    # Routes & helpers
    # ------------------------------------------------------------------
    def _setup_routes(self):
        @self.app.route('/')
        def index():  # noqa: D401 – simple route
            return render_template('backtest_dashboard.html')

        @self.app.route('/api/backtests')
        def list_backtests():
            """Return a JSON list with summary of every backtest log file."""
            summaries = self._load_backtest_summaries()
            return jsonify(summaries)

        @self.app.route('/api/backtests/<string:filename>')
        def get_backtest(filename: str):
            data = self._load_single_backtest(filename)
            if data is None:
                return jsonify({'error': 'not found'}), 404
            return jsonify(data)

        @self.app.route('/api/compare')
        def compare_backtests():
            first = request.args.get('first')
            second = request.args.get('second')
            if not first or not second:
                return jsonify({'error': 'first and second parameters required'}), 400
            first_data = self._load_single_backtest(first)
            second_data = self._load_single_backtest(second)
            if not first_data or not second_data:
                return jsonify({'error': 'one or both backtests not found'}), 404
            comparison = {
                'first': first_data,
                'second': second_data,
                'diff': self._compute_diff(first_data.get('results', {}), second_data.get('results', {}))
            }
            return jsonify(comparison)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_backtest_summaries(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries with key information for each backtest log."""
        summaries: List[Dict[str, Any]] = []
        for file_path in sorted(self.logs_dir.glob('*.json')):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                summary = {
                    'file': file_path.name,
                    'timestamp': data.get('timestamp'),
                    'strategy': data.get('strategy'),
                    'symbol': data.get('symbol'),
                    'timeframe': data.get('timeframe'),
                    'duration_years': data.get('duration_years'),
                    'total_trades': data.get('results', {}).get('total_trades'),
                    'win_rate': data.get('results', {}).get('win_rate'),
                    'total_return': data.get('results', {}).get('total_return'),
                    'annualized_return': data.get('results', {}).get('annualized_return'),
                    'max_drawdown': data.get('results', {}).get('max_drawdown'),
                    'sharpe_ratio': data.get('results', {}).get('sharpe_ratio'),
                }
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to read backtest log {file_path}: {e}")
        # Sort newest first
        summaries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return summaries

    def _load_single_backtest(self, filename: str) -> Optional[Dict[str, Any]]:
        path = self.logs_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load backtest file {filename}: {e}")
            return None

    def _compute_diff(self, first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
        keys = set(first.keys()) | set(second.keys())
        diff = {}
        for k in keys:
            diff[k] = {
                'first': first.get(k),
                'second': second.get(k),
            }
        return diff

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(self, host: str = '0.0.0.0', port: int = 8001, debug: bool = False):
        logger.info(f"Starting BacktestDashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# Allow running directly: python -m backtest_dashboard.dashboard
if __name__ == '__main__':
    dashboard = BacktestDashboard()
    dashboard.run(debug=True)