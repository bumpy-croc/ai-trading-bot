from __future__ import annotations

"""Backtest dashboard web server.

This module was relocated from ``src/backtest_dashboard`` to
``src/backtest/dashboard``. Functionality is identical: run with::

    python -m backtest.dashboard.dashboard
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from flask import Flask, jsonify, render_template, request  # type: ignore

logger = logging.getLogger(__name__)


class BacktestDashboard:
    """Simple dashboard to visualise historical backtest runs stored as JSON files."""

    def __init__(self, logs_dir: Union[str, Path] = 'src/backtesting/runs'):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Template/static folders live next to this file
        base_path = Path(__file__).parent
        self.app = Flask(
            __name__,
            template_folder=str(base_path / 'templates'),
            static_folder=str(base_path / 'static')
        )
        self._setup_routes()

    # -------------------------- routes ---------------------------------
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('backtest_dashboard.html')

        @self.app.route('/api/backtests')
        def list_backtests():
            return jsonify(self._load_backtest_summaries())

        @self.app.route('/api/backtests/<string:filename>')
        def get_backtest(filename: str):
            data = self._load_single_backtest(filename)
            return (jsonify(data) if data else (jsonify({'error': 'not found'}), 404))

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
            return jsonify({
                'first': first_data,
                'second': second_data,
                'diff': self._compute_diff(first_data.get('results', {}), second_data.get('results', {}))
            })

    # ------------------------ helpers ----------------------------------
    def _load_backtest_summaries(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for fp in sorted(self.logs_dir.glob('*.json')):
            try:
                with open(fp, 'r') as f:
                    data = json.load(f)
                summaries.append({
                    'file': fp.name,
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
                })
            except Exception as exc:
                logger.warning(f'Could not read backtest log {fp}: {exc}')
        summaries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return summaries

    def _load_single_backtest(self, filename: str) -> Optional[Dict[str, Any]]:
        path = self.logs_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f'Failed to load {filename}: {exc}')
            return None

    @staticmethod
    def _compute_diff(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
        keys = set(first) | set(second)
        return {k: {'first': first.get(k), 'second': second.get(k)} for k in keys}

    # ------------------------- run ------------------------------------
    def run(self, host: str = '0.0.0.0', port: int = 8001, debug: bool = False):
        logger.info(f'BacktestDashboard available at http://{host}:{port}')
        self.app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    BacktestDashboard().run(debug=True)