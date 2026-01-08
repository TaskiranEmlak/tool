# HFT Trading Tools - Cumulative Volume Delta (CVD)
"""
Kümülatif Hacim Deltası hesaplama modülü.
Piyasa agresifliğini ölçer: Alış baskısı vs Satış baskısı.

Formül: CVD_t = CVD_{t-1} + (V_buy - V_sell)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from core.event_bus import EventBus, Event, EventType, event_bus
from core.database import Database, Trade
from config.settings import settings


@dataclass
class CVDData:
    """CVD veri yapısı"""
    symbol: str
    cvd_value: float = 0.0
    delta: float = 0.0  # Son delta (buy - sell)
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Trend bilgisi
    trend: str = "neutral"  # "bullish", "bearish", "neutral"
    trend_strength: float = 0.0  # 0-1 arası


@dataclass
class DivergenceSignal:
    """Delta Divergence sinyali"""
    symbol: str
    divergence_type: str  # "bullish" veya "bearish"
    price_direction: str  # "up" veya "down"
    cvd_direction: str  # "up" veya "down"
    strength: float  # 0-1 arası
    timestamp: datetime


class CVDCalculator:
    """
    Kümülatif Hacim Deltası hesaplayıcı.
    
    Özellikler:
    - Multiple timeframe CVD (5m, 15m, 1h)
    - Delta Divergence tespiti
    - Borsalar arası CVD karşılaştırma
    - Tükeniş (exhaustion) tespiti
    """
    
    def __init__(self, db: Database = None):
        self.db = db
        self.event_bus = event_bus
        
        # Her sembol için CVD verileri
        self._cvd_data: Dict[str, CVDData] = {}
        
        # Rolling windows (deque ile hızlı)
        self._trade_windows: Dict[str, Dict[int, deque]] = {}  # symbol -> {window_sec -> trades}
        
        # Divergence tespiti için geçmiş
        self._price_history: Dict[str, deque] = {}  # Son 100 fiyat
        self._cvd_history: Dict[str, deque] = {}    # Son 100 CVD
        
        # Window süreleri (saniye)
        self.windows = [
            settings.CVD_WINDOW_5M,   # 300
            settings.CVD_WINDOW_15M,  # 900
            settings.CVD_WINDOW_1H    # 3600
        ]
    
    def _init_symbol(self, symbol: str):
        """Sembol için veri yapılarını başlat"""
        if symbol not in self._cvd_data:
            self._cvd_data[symbol] = CVDData(symbol=symbol)
            self._trade_windows[symbol] = {w: deque() for w in self.windows}
            self._price_history[symbol] = deque(maxlen=100)
            self._cvd_history[symbol] = deque(maxlen=100)
    
    def process_trade(self, trade: Trade) -> CVDData:
        """
        Yeni trade'i işle ve CVD güncelle.
        
        Args:
            trade: Trade verisi
            
        Returns:
            Güncel CVD verisi
        """
        symbol = trade.symbol
        self._init_symbol(symbol)
        
        now = datetime.now()
        cvd = self._cvd_data[symbol]
        
        # Delta hesapla
        delta = trade.quantity if trade.side == "buy" else -trade.quantity
        
        # Her window için güncelle
        for window_sec in self.windows:
            window = self._trade_windows[symbol][window_sec]
            
            # Yeni trade'i ekle
            window.append((trade.timestamp, delta))
            
            # Eski trade'leri temizle
            cutoff = now - timedelta(seconds=window_sec)
            while window and window[0][0] < cutoff:
                window.popleft()
        
        # Ana CVD değerini güncelle (5 dakikalık window)
        main_window = self._trade_windows[symbol][settings.CVD_WINDOW_5M]
        
        buy_volume = sum(d for _, d in main_window if d > 0)
        sell_volume = sum(-d for _, d in main_window if d < 0)
        
        cvd.cvd_value = buy_volume - sell_volume
        cvd.delta = delta
        cvd.buy_volume = buy_volume
        cvd.sell_volume = sell_volume
        cvd.timestamp = now
        
        # Trend hesapla
        cvd.trend, cvd.trend_strength = self._calculate_trend(symbol)
        
        # Geçmişe kaydet
        self._price_history[symbol].append(trade.price)
        self._cvd_history[symbol].append(cvd.cvd_value)
        
        # Event yayınla
        self.event_bus.publish_sync(Event(
            type=EventType.CVD_UPDATE,
            symbol=symbol,
            data={
                "cvd": cvd.cvd_value,
                "delta": cvd.delta,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "trend": cvd.trend,
                "trend_strength": cvd.trend_strength
            },
            timestamp=now
        ))
        
        # Veritabanına kaydet (her 100 trade'de bir)
        if self.db and len(main_window) % 100 == 0:
            self.db.insert_cvd(
                symbol=symbol,
                cvd_value=cvd.cvd_value,
                delta=cvd.delta,
                window_seconds=settings.CVD_WINDOW_5M,
                timestamp=now
            )
        
        return cvd
    
    def _calculate_trend(self, symbol: str) -> Tuple[str, float]:
        """
        CVD trendini hesapla.
        
        Returns:
            (trend, strength) - trend: "bullish"/"bearish"/"neutral", strength: 0-1
        """
        cvd_history = self._cvd_history.get(symbol)
        if not cvd_history or len(cvd_history) < 10:
            return "neutral", 0.0
        
        # Son 10 ve önceki 10 CVD ortalaması
        recent = list(cvd_history)[-10:]
        older = list(cvd_history)[-20:-10] if len(cvd_history) >= 20 else recent
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        # Trend yönü
        diff = recent_avg - older_avg
        
        # Normalleştir (max delta tahmini)
        max_delta = max(abs(max(cvd_history)), abs(min(cvd_history)), 1)
        strength = min(abs(diff) / max_delta, 1.0)
        
        if diff > 0 and strength > 0.1:
            return "bullish", strength
        elif diff < 0 and strength > 0.1:
            return "bearish", strength
        else:
            return "neutral", strength
    
    def detect_divergence(self, symbol: str) -> Optional[DivergenceSignal]:
        """
        Delta Divergence tespit et.
        
        Bullish Divergence: Fiyat düşüyor, CVD yükseliyor (absorpsiyon)
        Bearish Divergence: Fiyat yükseliyor, CVD düşüyor (tükeniş)
        
        Returns:
            DivergenceSignal veya None
        """
        price_history = self._price_history.get(symbol)
        cvd_history = self._cvd_history.get(symbol)
        
        if not price_history or not cvd_history or len(price_history) < 20:
            return None
        
        # Son 20 veri
        prices = list(price_history)[-20:]
        cvds = list(cvd_history)[-20:]
        
        # Fiyat trendi (ilk yarı vs son yarı)
        price_first = sum(prices[:10]) / 10
        price_last = sum(prices[10:]) / 10
        price_change = (price_last - price_first) / price_first
        
        # CVD trendi
        cvd_first = sum(cvds[:10]) / 10
        cvd_last = sum(cvds[10:]) / 10
        cvd_change = cvd_last - cvd_first
        
        # Normalleştir
        max_cvd = max(abs(max(cvds)), abs(min(cvds)), 1)
        cvd_change_norm = cvd_change / max_cvd
        
        # Divergence kontrolü
        # Threshold: Fiyat %0.5'ten fazla hareket etmeli
        if abs(price_change) < 0.005:
            return None
        
        # Bullish Divergence: Fiyat düşüyor, CVD yükseliyor
        if price_change < -0.005 and cvd_change_norm > 0.1:
            strength = min(abs(cvd_change_norm) * 2, 1.0)
            return DivergenceSignal(
                symbol=symbol,
                divergence_type="bullish",
                price_direction="down",
                cvd_direction="up",
                strength=strength,
                timestamp=datetime.now()
            )
        
        # Bearish Divergence: Fiyat yükseliyor, CVD düşüyor
        if price_change > 0.005 and cvd_change_norm < -0.1:
            strength = min(abs(cvd_change_norm) * 2, 1.0)
            return DivergenceSignal(
                symbol=symbol,
                divergence_type="bearish",
                price_direction="up",
                cvd_direction="down",
                strength=strength,
                timestamp=datetime.now()
            )
        
        return None
    
    def get_cvd(self, symbol: str) -> Optional[CVDData]:
        """Sembol için CVD verisi döndür"""
        return self._cvd_data.get(symbol)
    
    def get_cvd_for_window(self, symbol: str, window_seconds: int) -> float:
        """Belirli bir window için CVD döndür"""
        self._init_symbol(symbol)
        window = self._trade_windows[symbol].get(window_seconds)
        if not window:
            return 0.0
        return sum(d for _, d in window)
    
    def get_multi_timeframe_cvd(self, symbol: str) -> Dict[str, float]:
        """Tüm timeframe'ler için CVD"""
        return {
            "5m": self.get_cvd_for_window(symbol, settings.CVD_WINDOW_5M),
            "15m": self.get_cvd_for_window(symbol, settings.CVD_WINDOW_15M),
            "1h": self.get_cvd_for_window(symbol, settings.CVD_WINDOW_1H)
        }
    
    def reset(self, symbol: str = None):
        """CVD verilerini sıfırla"""
        if symbol:
            if symbol in self._cvd_data:
                del self._cvd_data[symbol]
            if symbol in self._trade_windows:
                del self._trade_windows[symbol]
        else:
            self._cvd_data.clear()
            self._trade_windows.clear()
