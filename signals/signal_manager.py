# HFT Trading Tools - Sinyal Yöneticisi
"""
Tüm göstergelerden gelen verileri birleştirip sinyal üreten ana modül.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.event_bus import EventBus, Event, EventType, event_bus
from core.database import Database
from indicators.cvd import CVDCalculator, CVDData
from indicators.obi import OBICalculator, OBIData
from indicators.liquidation import LiquidationHeatmap, HeatmapData
from config.settings import settings


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    NEUTRAL = "neutral"


@dataclass
class TradingSignal:
    """Trading sinyali"""
    symbol: str
    direction: SignalDirection
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Gerekçe
    reasons: List[str] = field(default_factory=list)
    
    # Gösterge değerleri
    cvd_value: float = 0.0
    cvd_trend: str = "neutral"
    obi_value: float = 0.0
    obi_signal: str = "neutral"
    near_liquidation: bool = False
    
    # Meta
    timestamp: datetime = field(default_factory=datetime.now)
    id: int = 0
    status: str = "active"


class SignalManager:
    """
    Ana sinyal yöneticisi.
    
    CVD, OBI ve Likidasyon verilerini birleştirerek sinyal üretir.
    """
    
    def __init__(self, db: Database = None):
        self.db = db
        self.event_bus = event_bus
        
        # Göstergeler
        self.cvd_calc = CVDCalculator(db)
        self.obi_calc = OBICalculator(db)
        self.liq_heatmap = LiquidationHeatmap(db)
        
        # Aktif sinyaller
        self._active_signals: Dict[str, TradingSignal] = {}
        
        # Sinyal geçmişi
        self._signal_history: List[TradingSignal] = []
        
        # Son sinyal zamanları (cooldown için)
        self._last_signal_time: Dict[str, datetime] = {}
        
        # Parametreler
        self.min_confidence = settings.SIGNAL_MIN_CONFIDENCE  # 0.6
        self.cooldown_seconds = settings.SIGNAL_COOLDOWN_SECONDS  # 60
        
        # Risk/Ödül oranları
        self.default_sl_percent = 0.5  # %0.5 stop loss
        self.default_tp_percent = 1.0  # %1 take profit (1:2 R/R)
    
    def generate_signal(self, symbol: str, current_price: float,
                        bids: List[Tuple[float, float]],
                        asks: List[Tuple[float, float]]) -> Optional[TradingSignal]:
        """
        Tüm göstergeleri analiz edip sinyal üret.
        
        Args:
            symbol: Sembol
            current_price: Mevcut fiyat
            bids: Order book alış emirleri
            asks: Order book satış emirleri
            
        Returns:
            TradingSignal veya None
        """
        now = datetime.now()
        
        # Cooldown kontrolü
        if symbol in self._last_signal_time:
            elapsed = (now - self._last_signal_time[symbol]).seconds
            if elapsed < self.cooldown_seconds:
                return None
        
        # OBI hesapla
        obi = self.obi_calc.calculate(symbol, bids, asks)
        
        # CVD al (trade callback'ten güncelleniyor)
        cvd = self.cvd_calc.get_cvd(symbol)
        
        # Heatmap hesapla
        heatmap = self.liq_heatmap.calculate_heatmap(symbol, current_price)
        
        # Likidasyon bölgesine yakın mı?
        near_liq, liq_level = self.liq_heatmap.is_approaching_liquidation_zone(
            symbol, current_price, threshold_percent=1.0
        )
        
        # Sinyal analizi
        direction = SignalDirection.NEUTRAL
        confidence = 0.0
        reasons = []
        
        # ====== LONG Sinyal Koşulları ======
        long_score = 0.0
        
        # 1. OBI pozitif (güçlü alış desteği)
        if obi.weighted_obi > 0.3:
            long_score += 0.3
            reasons.append(f"OBI pozitif: {obi.weighted_obi:.2f}")
        elif obi.weighted_obi > 0.5:
            long_score += 0.4
            reasons.append(f"OBI çok güçlü: {obi.weighted_obi:.2f}")
        
        # 2. CVD yükseliyor (alış baskısı)
        if cvd and cvd.trend == "bullish":
            long_score += 0.25
            reasons.append(f"CVD yükseliyor (trend: {cvd.trend_strength:.2f})")
        
        # 3. CVD ve OBI aynı yönde (uyum)
        if cvd and obi.weighted_obi > 0.2 and cvd.cvd_value > 0:
            long_score += 0.2
            reasons.append("CVD-OBI uyumu (ikisi de pozitif)")
        
        # 4. Likidasyon bölgesinden çıkış (mean reversion)
        if near_liq and liq_level and liq_level.side == "long":
            # Long likidasyonlar patladı, yukarı dönüş beklenir
            long_score += 0.25
            reasons.append(f"Long likidasyon bölgesi: ${liq_level.price:,.0f}")
        
        # ====== SHORT Sinyal Koşulları ======
        short_score = 0.0
        
        # 1. OBI negatif (güçlü satış baskısı)
        if obi.weighted_obi < -0.3:
            short_score += 0.3
            reasons.append(f"OBI negatif: {obi.weighted_obi:.2f}")
        elif obi.weighted_obi < -0.5:
            short_score += 0.4
            reasons.append(f"OBI çok negatif: {obi.weighted_obi:.2f}")
        
        # 2. CVD düşüyor (satış baskısı)
        if cvd and cvd.trend == "bearish":
            short_score += 0.25
            reasons.append(f"CVD düşüyor (trend: {cvd.trend_strength:.2f})")
        
        # 3. CVD ve OBI aynı yönde
        if cvd and obi.weighted_obi < -0.2 and cvd.cvd_value < 0:
            short_score += 0.2
            reasons.append("CVD-OBI uyumu (ikisi de negatif)")
        
        # 4. Likidasyon bölgesinden çıkış
        if near_liq and liq_level and liq_level.side == "short":
            short_score += 0.25
            reasons.append(f"Short likidasyon bölgesi: ${liq_level.price:,.0f}")
        
        # Yön belirleme
        if long_score > short_score and long_score >= self.min_confidence:
            direction = SignalDirection.LONG
            confidence = min(long_score, 1.0)
        elif short_score > long_score and short_score >= self.min_confidence:
            direction = SignalDirection.SHORT
            confidence = min(short_score, 1.0)
        else:
            return None  # Yeterli güven yok
        
        # SL/TP hesapla
        if direction == SignalDirection.LONG:
            stop_loss = current_price * (1 - self.default_sl_percent / 100)
            take_profit = current_price * (1 + self.default_tp_percent / 100)
        else:
            stop_loss = current_price * (1 + self.default_sl_percent / 100)
            take_profit = current_price * (1 - self.default_tp_percent / 100)
        
        # Sinyal oluştur
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            cvd_value=cvd.cvd_value if cvd else 0,
            cvd_trend=cvd.trend if cvd else "neutral",
            obi_value=obi.weighted_obi,
            obi_signal=obi.signal,
            near_liquidation=near_liq,
            timestamp=now
        )
        
        # Veritabanına kaydet
        if self.db:
            signal.id = self.db.insert_signal(
                symbol=symbol,
                direction=direction.value,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason="; ".join(reasons),
                timestamp=now
            )
        
        # Aktif sinyale ekle
        self._active_signals[symbol] = signal
        self._signal_history.append(signal)
        self._last_signal_time[symbol] = now
        
        # Event yayınla
        event_type = EventType.SIGNAL_LONG if direction == SignalDirection.LONG else EventType.SIGNAL_SHORT
        self.event_bus.publish_sync(Event(
            type=event_type,
            symbol=symbol,
            data={
                "direction": direction.value,
                "confidence": confidence,
                "entry": current_price,
                "sl": stop_loss,
                "tp": take_profit,
                "reasons": reasons
            },
            timestamp=now
        ))
        
        return signal
    
    def check_signal_exit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Aktif sinyalin çıkış kontrolü.
        
        Returns:
            "tp", "sl", veya None
        """
        if symbol not in self._active_signals:
            return None
        
        signal = self._active_signals[symbol]
        
        if signal.direction == SignalDirection.LONG:
            if current_price >= signal.take_profit:
                return "tp"
            elif current_price <= signal.stop_loss:
                return "sl"
        elif signal.direction == SignalDirection.SHORT:
            if current_price <= signal.take_profit:
                return "tp"
            elif current_price >= signal.stop_loss:
                return "sl"
        
        return None
    
    def close_signal(self, symbol: str, exit_type: str, exit_price: float):
        """Sinyali kapat"""
        if symbol not in self._active_signals:
            return
        
        signal = self._active_signals[symbol]
        
        # PnL hesapla
        if signal.direction == SignalDirection.LONG:
            pnl_percent = ((exit_price - signal.entry_price) / signal.entry_price) * 100
        else:
            pnl_percent = ((signal.entry_price - exit_price) / signal.entry_price) * 100
        
        signal.status = f"closed_{exit_type}"
        
        # DB güncelle
        if self.db and signal.id:
            self.db.close_signal(signal.id, pnl_percent)
        
        # Event yayınla
        self.event_bus.publish_sync(Event(
            type=EventType.SIGNAL_CLOSE,
            symbol=symbol,
            data={
                "exit_type": exit_type,
                "entry": signal.entry_price,
                "exit": exit_price,
                "pnl_percent": pnl_percent
            },
            timestamp=datetime.now()
        ))
        
        del self._active_signals[symbol]
    
    def get_active_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Aktif sinyal döndür"""
        return self._active_signals.get(symbol)
    
    def get_all_active_signals(self) -> Dict[str, TradingSignal]:
        """Tüm aktif sinyalleri döndür"""
        return self._active_signals.copy()
    
    def get_signal_history(self, limit: int = 50) -> List[TradingSignal]:
        """Sinyal geçmişi"""
        return self._signal_history[-limit:]
    
    def get_indicators(self, symbol: str) -> Dict:
        """Sembol için tüm gösterge değerlerini döndür"""
        cvd = self.cvd_calc.get_cvd(symbol)
        obi = self.obi_calc.get_obi(symbol)
        heatmap = self.liq_heatmap.get_heatmap(symbol)
        
        return {
            "cvd": {
                "value": cvd.cvd_value if cvd else 0,
                "delta": cvd.delta if cvd else 0,
                "trend": cvd.trend if cvd else "neutral",
                "buy_volume": cvd.buy_volume if cvd else 0,
                "sell_volume": cvd.sell_volume if cvd else 0
            } if cvd else None,
            "obi": {
                "value": obi.obi_value if obi else 0,
                "weighted": obi.weighted_obi if obi else 0,
                "signal": obi.signal if obi else "neutral",
                "spread": obi.spread if obi else 0,
                "bid_volume": obi.bid_volume if obi else 0,
                "ask_volume": obi.ask_volume if obi else 0
            } if obi else None,
            "heatmap": {
                "magnet_zone": heatmap.magnet_zone if heatmap else None,
                "level_count": len(heatmap.levels) if heatmap else 0
            } if heatmap else None
        }
