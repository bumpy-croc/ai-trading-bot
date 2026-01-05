from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BalanceTracker:
    starting_balance: float
    current_balance: float
    peak_balance: float
    total_pnl: float = 0.0
    max_drawdown: float = 0.0  # as fraction (0..1)

    @classmethod
    def start(cls, initial_balance: float) -> BalanceTracker:
        return cls(
            starting_balance=initial_balance,
            current_balance=initial_balance,
            peak_balance=initial_balance,
            total_pnl=0.0,
            max_drawdown=0.0,
        )

    def apply_pnl(self, pnl_dollars: float) -> None:
        new_balance = self.current_balance + pnl_dollars
        # Protect against negative balance (prevents division errors in position sizing)
        if new_balance < 0:
            logger.critical(
                "CRITICAL: Balance would go negative (%.2f + %.2f = %.2f) - clamping to 0",
                self.current_balance,
                pnl_dollars,
                new_balance,
            )
            new_balance = 0.0
        self.current_balance = new_balance
        self.total_pnl += pnl_dollars
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    @property
    def total_return_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return (self.current_balance - self.starting_balance) / self.starting_balance * 100.0

    @property
    def current_drawdown_pct(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance * 100.0
