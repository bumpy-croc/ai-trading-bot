from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BalanceTracker:
    starting_balance: float
    current_balance: float
    peak_balance: float
    total_pnl: float = 0.0
    max_drawdown: float = 0.0  # as fraction (0..1)

    @classmethod
    def start(cls, initial_balance: float) -> "BalanceTracker":
        return cls(
            starting_balance=initial_balance,
            current_balance=initial_balance,
            peak_balance=initial_balance,
            total_pnl=0.0,
            max_drawdown=0.0,
        )

    def apply_pnl(self, pnl_dollars: float) -> None:
        self.current_balance += pnl_dollars
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