# HFT Trading Tools - Order Book Imbalance (OBI)
"""
Emir Defteri Dengesizliği hesaplama modülü.
Kısa vadeli fiyat hareketlerinin en güçlü öncü göstergesi.

Formül: ρ = (V_bid - V_ask) / (V_bid + V_ask)
Ağırlıklı: w_i = e^(-k * (i-1))
"""

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from core.event_bus import EventBus, Event, EventType, event_bus
from core.database import Database
from config.settings import settings


@dataclass
class OBIData:
    """OBI veri yapısı"""
    symbol: str
    obi_value: float = 0.0           # Standart OBI (-1 ile +1 arası)
    weighted_obi: float = 0.0        # Ağırlıklı OBI
    bid_volume: float = 0.0          # Toplam alış hacmi
    ask_volume: float = 0.0          # Toplam satış hacmi
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    spread_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Sinyal durumu
    signal: str = "neutral"          # "strong_buy", "buy", "neutral", "sell", "strong_sell"
    signal_strength: float = 0.0     # 0-1 arası


@dataclass
class OBIPressure:
    """Order Book baskı analizi"""
    symbol: str
    buy_pressure: float = 0.0        # Alış tarafı baskısı (0-1)
    sell_pressure: float = 0.0       # Satış tarafı baskısı (0-1)
    dominant_side: str = "neutral"   # "bid", "ask", "neutral"
    wall_detected: bool = False      # Büyük duvar tespit edildi mi
    wall_side: str = ""              # "bid" veya "ask"
    wall_price: float = 0.0
    wall_size: float = 0.0


class OBICalculator:
    """
    Emir Defteri Dengesizliği hesaplayıcı.
    
    Özellikler:
    - Standart OBI hesaplama
    - Ağırlıklı OBI (decay factor ile)
    - Büyük duvar tespiti
    - Baskı analizi
    - Spoofing şüphesi tespiti
    """
    
    def __init__(self, db: Database = None):
        self.db = db
        self.event_bus = event_bus
        
        # Her sembol için OBI verileri
        self._obi_data: Dict[str, OBIData] = {}
        
        # OBI geçmişi (trend için)
        self._obi_history: Dict[str, deque] = {}
        
        # Parametreler
        self.depth = settings.OBI_DEPTH                    # Kaç seviye analiz edilecek
        self.decay_factor = settings.OBI_DECAY_FACTOR      # Ağırlık azalma katsayısı
        self.strong_threshold = settings.OBI_STRONG_THRESHOLD  # Güçlü sinyal eşiği
        
        # Duvar tespiti için (ortalama hacmin 5 katı)
        self.wall_multiplier = 5.0
        
        # Ağırlıkları önceden hesapla (performans için)
        self._weights = [math.exp(-self.decay_factor * i) for i in range(self.depth)]
        self._weight_sum = sum(self._weights)
    
    def _init_symbol(self, symbol: str):
        """Sembol için veri yapılarını başlat"""
        if symbol not in self._obi_data:
            self._obi_data[symbol] = OBIData(symbol=symbol)
            self._obi_history[symbol] = deque(maxlen=100)
    
    def calculate(self, symbol: str, bids: List[Tuple[float, float]], 
                  asks: List[Tuple[float, float]]) -> OBIData:
        """
        Order book verisiyle OBI hesapla.
        
        Args:
            symbol: Sembol
            bids: Alış emirleri [(price, qty), ...] en yüksekten aşağı sıralı
            asks: Satış emirleri [(price, qty), ...] en düşükten yukarı sıralı
            
        Returns:
            OBI verisi
        """
        self._init_symbol(symbol)
        now = datetime.now()
        obi = self._obi_data[symbol]
        
        if not bids or not asks:
            return obi
        
        # En iyi fiyatlar
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0
        
        # Toplam hacimleri hesapla
        bid_volume = sum(qty for _, qty in bids[:self.depth])
        ask_volume = sum(qty for _, qty in asks[:self.depth])
        total_volume = bid_volume + ask_volume
        
        # Standart OBI
        if total_volume > 0:
            standard_obi = (bid_volume - ask_volume) / total_volume
        else:
            standard_obi = 0.0
        
        # Ağırlıklı OBI hesapla
        weighted_bid = 0.0
        weighted_ask = 0.0
        
        for i, (price, qty) in enumerate(bids[:self.depth]):
            if i < len(self._weights):
                weighted_bid += qty * self._weights[i]
        
        for i, (price, qty) in enumerate(asks[:self.depth]):
            if i < len(self._weights):
                weighted_ask += qty * self._weights[i]
        
        weighted_total = weighted_bid + weighted_ask
        if weighted_total > 0:
            weighted_obi = (weighted_bid - weighted_ask) / weighted_total
        else:
            weighted_obi = 0.0
        
        # Sinyal durumu
        signal, signal_strength = self._calculate_signal(weighted_obi)
        
        # OBI verisini güncelle
        obi.obi_value = standard_obi
        obi.weighted_obi = weighted_obi
        obi.bid_volume = bid_volume
        obi.ask_volume = ask_volume
        obi.best_bid = best_bid
        obi.best_ask = best_ask
        obi.spread = spread
        obi.spread_percent = spread_percent
        obi.signal = signal
        obi.signal_strength = signal_strength
        obi.timestamp = now
        
        # Geçmişe kaydet
        self._obi_history[symbol].append(weighted_obi)
        
        # Event yayınla
        self.event_bus.publish_sync(Event(
            type=EventType.OBI_UPDATE,
            symbol=symbol,
            data={
                "obi": standard_obi,
                "weighted_obi": weighted_obi,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "spread": spread,
                "spread_percent": spread_percent,
                "signal": signal,
                "signal_strength": signal_strength
            },
            timestamp=now
        ))
        
        # Veritabanına kaydet (throttled)
        if self.db and len(self._obi_history[symbol]) % 50 == 0:
            self.db.insert_obi(
                symbol=symbol,
                obi_value=standard_obi,
                weighted_obi=weighted_obi,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                timestamp=now
            )
        
        return obi
    
    def _calculate_signal(self, weighted_obi: float) -> Tuple[str, float]:
        """
        OBI değerinden sinyal üret.
        
        Returns:
            (signal, strength)
        """
        strength = abs(weighted_obi)
        
        if weighted_obi > 0.5:
            return "strong_buy", strength
        elif weighted_obi > self.strong_threshold:  # 0.3
            return "buy", strength
        elif weighted_obi < -0.5:
            return "strong_sell", strength
        elif weighted_obi < -self.strong_threshold:
            return "sell", strength
        else:
            return "neutral", strength
    
    def detect_walls(self, symbol: str, bids: List[Tuple[float, float]], 
                     asks: List[Tuple[float, float]]) -> OBIPressure:
        """
        Büyük duvar (wall) tespiti.
        
        Args:
            symbol: Sembol
            bids/asks: Order book verileri
            
        Returns:
            OBIPressure analizi
        """
        pressure = OBIPressure(symbol=symbol)
        
        if not bids or not asks:
            return pressure
        
        # Ortalama hacim
        all_volumes = [qty for _, qty in bids[:self.depth]] + [qty for _, qty in asks[:self.depth]]
        if not all_volumes:
            return pressure
        
        avg_volume = sum(all_volumes) / len(all_volumes)
        wall_threshold = avg_volume * self.wall_multiplier
        
        # Bid duvarları kontrol
        bid_wall_price = 0.0
        bid_wall_size = 0.0
        for price, qty in bids[:self.depth]:
            if qty > wall_threshold and qty > bid_wall_size:
                bid_wall_price = price
                bid_wall_size = qty
        
        # Ask duvarları kontrol
        ask_wall_price = 0.0
        ask_wall_size = 0.0
        for price, qty in asks[:self.depth]:
            if qty > wall_threshold and qty > ask_wall_size:
                ask_wall_price = price
                ask_wall_size = qty
        
        # Hangi taraf dominant?
        total_bid = sum(qty for _, qty in bids[:self.depth])
        total_ask = sum(qty for _, qty in asks[:self.depth])
        total = total_bid + total_ask
        
        if total > 0:
            pressure.buy_pressure = total_bid / total
            pressure.sell_pressure = total_ask / total
        
        if pressure.buy_pressure > 0.6:
            pressure.dominant_side = "bid"
        elif pressure.sell_pressure > 0.6:
            pressure.dominant_side = "ask"
        else:
            pressure.dominant_side = "neutral"
        
        # En büyük duvarı belirle
        if bid_wall_size > 0 or ask_wall_size > 0:
            pressure.wall_detected = True
            if bid_wall_size > ask_wall_size:
                pressure.wall_side = "bid"
                pressure.wall_price = bid_wall_price
                pressure.wall_size = bid_wall_size
            else:
                pressure.wall_side = "ask"
                pressure.wall_price = ask_wall_price
                pressure.wall_size = ask_wall_size
        
        return pressure
    
    def get_obi(self, symbol: str) -> Optional[OBIData]:
        """Sembol için OBI verisi döndür"""
        return self._obi_data.get(symbol)
    
    def get_obi_trend(self, symbol: str, lookback: int = 20) -> str:
        """
        OBI trendini hesapla.
        
        Returns:
            "up", "down", veya "sideways"
        """
        history = self._obi_history.get(symbol)
        if not history or len(history) < lookback:
            return "sideways"
        
        recent = list(history)[-lookback:]
        first_half = sum(recent[:lookback//2]) / (lookback//2)
        second_half = sum(recent[lookback//2:]) / (lookback//2)
        
        diff = second_half - first_half
        
        if diff > 0.1:
            return "up"
        elif diff < -0.1:
            return "down"
        else:
            return "sideways"
    
    def is_buy_signal(self, symbol: str) -> Tuple[bool, float]:
        """
        Alış sinyali kontrolü.
        
        Returns:
            (is_signal, confidence)
        """
        obi = self.get_obi(symbol)
        if not obi:
            return False, 0.0
        
        if obi.signal in ["buy", "strong_buy"]:
            return True, obi.signal_strength
        return False, 0.0
    
    def is_sell_signal(self, symbol: str) -> Tuple[bool, float]:
        """
        Satış sinyali kontrolü.
        
        Returns:
            (is_signal, confidence)
        """
        obi = self.get_obi(symbol)
        if not obi:
            return False, 0.0
        
        if obi.signal in ["sell", "strong_sell"]:
            return True, obi.signal_strength
        return False, 0.0
    
    def reset(self, symbol: str = None):
        """OBI verilerini sıfırla"""
        if symbol:
            if symbol in self._obi_data:
                del self._obi_data[symbol]
            if symbol in self._obi_history:
                del self._obi_history[symbol]
        else:
            self._obi_data.clear()
            self._obi_history.clear()
