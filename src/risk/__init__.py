from .risk_manager import PortfolioRiskManager, RiskParameters

# Backward compatibility alias
RiskManager = PortfolioRiskManager

__all__ = ["PortfolioRiskManager", "RiskManager", "RiskParameters"]
