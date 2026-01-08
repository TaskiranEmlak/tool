# HFT Trading Tools - Divergence Detector
"""
CVD/Fiyat uyumsuzluklarını tespit eden modül.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from core.event_bus import EventBus, event_bus


@dataclass
class DivergenceAlert:
    """Divergence uyarısı"""
    symbol: str
    type: str  # "bullish_divergence", "bearish_divergence"
    description: str
    strength: float  # 0-1
    price: float
    cvd_value: float
    timestamp: datetime


class DivergenceDetector:
    """
    Fiyat-CVD Divergence tespiti.
    
    Bullish: Fiyat düşüyor ama CVD yükseliyor (absorpsiyon)
    Bearish: Fiyat yükseliyor ama CVD düşüyor (tükeniş)
    """
    
    def __init__(self):
        self.event_bus = event_bus
        self._last_alert_time = {}  # Cooldown için
    
    def check(self, symbol: str, price_change: float, cvd_change: float,
              current_price: float, current_cvd: float) -> Optional[DivergenceAlert]:
        """
        Divergence kontrolü.
        
        Args:
            symbol: Sembol
            price_change: Fiyat değişimi (%)
            cvd_change: CVD değişimi (normalize)
            current_price: Mevcut fiyat
            current_cvd: Mevcut CVD
            
        Returns:
            DivergenceAlert veya None
        """
        now = datetime.now()
        
        # Cooldown kontrolü (60 saniye)
        if symbol in self._last_alert_time:
            if (now - self._last_alert_time[symbol]).seconds < 60:
                return None
        
        # Minimum hareket eşiği
        if abs(price_change) < 0.3:  # %0.3'ten az hareket
            return None
        
        alert = None
        
        # Bullish Divergence
        if price_change < -0.3 and cvd_change > 0.1:
            strength = min(abs(cvd_change) * 2, 1.0)
            alert = DivergenceAlert(
                symbol=symbol,
                type="bullish_divergence",
                description=f"Fiyat düşerken alış baskısı artıyor (Absorpsiyon)",
                strength=strength,
                price=current_price,
                cvd_value=current_cvd,
                timestamp=now
            )
        
        # Bearish Divergence
        elif price_change > 0.3 and cvd_change < -0.1:
            strength = min(abs(cvd_change) * 2, 1.0)
            alert = DivergenceAlert(
                symbol=symbol,
                type="bearish_divergence",
                description=f"Fiyat yükselirken satış baskısı artıyor (Tükeniş)",
                strength=strength,
                price=current_price,
                cvd_value=current_cvd,
                timestamp=now
            )
        
        if alert:
            self._last_alert_time[symbol] = now
        
        return alert
