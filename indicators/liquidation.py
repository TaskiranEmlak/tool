# HFT Trading Tools - Likidasyon Heatmap
"""
Likidasyon seviyelerini tahmin eden ve ısı haritası oluşturan modül.
"Mıknatıs Teorisi": Fiyat, likidite yoğun bölgelere çekilir.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.event_bus import EventBus, Event, EventType, event_bus
from core.database import Database, Liquidation
from config.settings import settings


@dataclass
class LiquidationLevel:
    """Likidasyon seviyesi"""
    price: float
    total_volume: float          # Toplam likidasyon hacmi
    count: int                   # Kaç likidasyon oldu
    side: str                    # "long" veya "short"
    intensity: float = 0.0       # 0-1 arası yoğunluk
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class HeatmapData:
    """Heatmap veri yapısı"""
    symbol: str
    levels: List[LiquidationLevel]
    strongest_long_liq: Optional[LiquidationLevel] = None   # En güçlü long likidasyon bölgesi
    strongest_short_liq: Optional[LiquidationLevel] = None  # En güçlü short likidasyon bölgesi
    magnet_zone: Optional[Tuple[float, float]] = None       # Mıknatıs bölgesi (alt, üst)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EstimatedLiquidation:
    """Tahmini likidasyon seviyesi (OI bazlı)"""
    entry_price: float
    leverage: int
    liquidation_price_long: float
    liquidation_price_short: float
    oi_volume: float


class LiquidationHeatmap:
    """
    Likidasyon Isı Haritası oluşturucu.
    
    Özellikler:
    - Gerçek likidasyon verisi toplama
    - Tahmini likidasyon seviyesi hesaplama (OI + Kaldıraç)
    - Isı haritası matrisi
    - Mıknatıs bölgesi tespiti
    - Zaman aşımı (decay)
    """
    
    def __init__(self, db: Database = None):
        self.db = db
        self.event_bus = event_bus
        
        # Her sembol için likidasyon verileri
        self._liquidations: Dict[str, Dict[float, LiquidationLevel]] = {}
        
        # Heatmap cache
        self._heatmap_cache: Dict[str, HeatmapData] = {}
        
        # Parametreler
        self.cluster_percent = settings.LIQUIDATION_CLUSTER_PERCENT / 100  # 0.005
        self.decay_hours = settings.LIQUIDATION_DECAY_HOURS  # 24
        
        # Yaygın kaldıraç seviyeleri
        self.common_leverages = [10, 20, 25, 50, 75, 100, 125]
    
    def _price_to_cluster(self, price: float, reference_price: float) -> float:
        """
        Fiyatı küme merkezine yuvarla.
        
        Args:
            price: Ham fiyat
            reference_price: Referans fiyat (cluster boyutu hesabı için)
            
        Returns:
            Küme merkezi fiyatı
        """
        cluster_size = reference_price * self.cluster_percent
        return round(price / cluster_size) * cluster_size
    
    def add_liquidation(self, liq: Liquidation, reference_price: float):
        """
        Yeni likidasyon ekle.
        
        Args:
            liq: Likidasyon verisi
            reference_price: Kümeleme için referans fiyat
        """
        symbol = liq.symbol
        
        if symbol not in self._liquidations:
            self._liquidations[symbol] = {}
        
        # Fiyatı kümeye ata
        cluster_price = self._price_to_cluster(liq.price, reference_price)
        
        if cluster_price not in self._liquidations[symbol]:
            self._liquidations[symbol][cluster_price] = LiquidationLevel(
                price=cluster_price,
                total_volume=0.0,
                count=0,
                side=liq.side
            )
        
        level = self._liquidations[symbol][cluster_price]
        level.total_volume += liq.quantity * liq.price  # USD değeri
        level.count += 1
        level.last_update = liq.timestamp
        
        # Event yayınla
        self.event_bus.publish_sync(Event(
            type=EventType.LIQUIDATION,
            symbol=symbol,
            data={
                "price": liq.price,
                "cluster_price": cluster_price,
                "volume": liq.quantity * liq.price,
                "side": liq.side
            },
            timestamp=liq.timestamp
        ))
    
    def _apply_decay(self, symbol: str):
        """Zaman aşımı uygula - eski likidasyonların önemini azalt"""
        now = datetime.now()
        cutoff = now - timedelta(hours=self.decay_hours)
        
        if symbol not in self._liquidations:
            return
        
        to_remove = []
        for price, level in self._liquidations[symbol].items():
            if level.last_update < cutoff:
                to_remove.append(price)
            else:
                # Decay factor hesapla
                age_hours = (now - level.last_update).total_seconds() / 3600
                decay_factor = math.exp(-age_hours / self.decay_hours)
                level.intensity = level.total_volume * decay_factor
        
        for price in to_remove:
            del self._liquidations[symbol][price]
    
    def calculate_heatmap(self, symbol: str, current_price: float) -> HeatmapData:
        """
        Isı haritası hesapla.
        
        Args:
            symbol: Sembol
            current_price: Mevcut fiyat
            
        Returns:
            HeatmapData
        """
        self._apply_decay(symbol)
        
        if symbol not in self._liquidations:
            return HeatmapData(symbol=symbol, levels=[])
        
        levels = list(self._liquidations[symbol].values())
        
        if not levels:
            return HeatmapData(symbol=symbol, levels=[])
        
        # Yoğunlukları normalize et
        max_volume = max(l.total_volume for l in levels)
        if max_volume > 0:
            for level in levels:
                level.intensity = level.total_volume / max_volume
        
        # Fiyata göre sırala
        levels.sort(key=lambda x: x.price)
        
        # En güçlü seviyeleri bul
        strongest_long = None
        strongest_short = None
        
        for level in levels:
            if level.side == "long":
                if strongest_long is None or level.intensity > strongest_long.intensity:
                    strongest_long = level
            else:
                if strongest_short is None or level.intensity > strongest_short.intensity:
                    strongest_short = level
        
        # Mıknatıs bölgesi (en yoğun alan)
        magnet_zone = None
        if levels:
            # En yoğun 3 seviyenin ortalaması
            sorted_by_intensity = sorted(levels, key=lambda x: x.intensity, reverse=True)[:3]
            if sorted_by_intensity:
                avg_price = sum(l.price for l in sorted_by_intensity) / len(sorted_by_intensity)
                zone_size = current_price * 0.02  # %2 aralık
                magnet_zone = (avg_price - zone_size, avg_price + zone_size)
        
        heatmap = HeatmapData(
            symbol=symbol,
            levels=levels,
            strongest_long_liq=strongest_long,
            strongest_short_liq=strongest_short,
            magnet_zone=magnet_zone,
            timestamp=datetime.now()
        )
        
        self._heatmap_cache[symbol] = heatmap
        
        # Event yayınla
        self.event_bus.publish_sync(Event(
            type=EventType.HEATMAP_UPDATE,
            symbol=symbol,
            data={
                "level_count": len(levels),
                "magnet_zone": magnet_zone,
                "strongest_long": strongest_long.price if strongest_long else None,
                "strongest_short": strongest_short.price if strongest_short else None
            },
            timestamp=datetime.now()
        ))
        
        return heatmap
    
    def estimate_liquidation_levels(self, entry_price: float, 
                                    oi_volume: float = 0) -> List[EstimatedLiquidation]:
        """
        Tahmini likidasyon seviyeleri hesapla.
        
        Likidasyon formülü (isolated margin):
        - Long: entry_price * (1 - 1/leverage)
        - Short: entry_price * (1 + 1/leverage)
        
        Args:
            entry_price: Giriş fiyatı
            oi_volume: Açık pozisyon hacmi (opsiyonel)
            
        Returns:
            Farklı kaldıraç seviyeleri için likidasyon fiyatları
        """
        estimates = []
        
        for leverage in self.common_leverages:
            liq_long = entry_price * (1 - 1/leverage)
            liq_short = entry_price * (1 + 1/leverage)
            
            estimates.append(EstimatedLiquidation(
                entry_price=entry_price,
                leverage=leverage,
                liquidation_price_long=liq_long,
                liquidation_price_short=liq_short,
                oi_volume=oi_volume
            ))
        
        return estimates
    
    def get_nearby_liquidations(self, symbol: str, current_price: float, 
                                 range_percent: float = 2.0) -> List[LiquidationLevel]:
        """
        Mevcut fiyata yakın likidasyon seviyelerini getir.
        
        Args:
            symbol: Sembol
            current_price: Mevcut fiyat
            range_percent: Aralık yüzdesi
            
        Returns:
            Yakın likidasyon seviyeleri
        """
        if symbol not in self._liquidations:
            return []
        
        range_factor = range_percent / 100
        lower_bound = current_price * (1 - range_factor)
        upper_bound = current_price * (1 + range_factor)
        
        nearby = [
            level for level in self._liquidations[symbol].values()
            if lower_bound <= level.price <= upper_bound
        ]
        
        # Yoğunluğa göre sırala
        nearby.sort(key=lambda x: x.intensity, reverse=True)
        
        return nearby
    
    def is_approaching_liquidation_zone(self, symbol: str, current_price: float,
                                         threshold_percent: float = 1.0) -> Tuple[bool, Optional[LiquidationLevel]]:
        """
        Fiyat likidasyon bölgesine yaklaşıyor mu kontrol et.
        
        Returns:
            (is_approaching, nearest_level)
        """
        nearby = self.get_nearby_liquidations(symbol, current_price, threshold_percent)
        
        if nearby:
            # En yoğun seviye
            strongest = max(nearby, key=lambda x: x.intensity)
            if strongest.intensity > 0.5:  # Yeterince güçlü
                return True, strongest
        
        return False, None
    
    def get_heatmap_matrix(self, symbol: str, current_price: float,
                           levels: int = 50, range_percent: float = 5.0) -> List[Dict]:
        """
        Görselleştirme için heatmap matrisi oluştur.
        
        Args:
            symbol: Sembol
            current_price: Mevcut fiyat
            levels: Kaç fiyat seviyesi
            range_percent: Toplam aralık yüzdesi
            
        Returns:
            [{price, intensity, side, volume}, ...]
        """
        range_factor = range_percent / 100
        lower = current_price * (1 - range_factor)
        upper = current_price * (1 + range_factor)
        step = (upper - lower) / levels
        
        matrix = []
        
        for i in range(levels):
            price = lower + (i * step)
            
            # Bu fiyata en yakın likidasyon seviyesini bul
            intensity = 0.0
            volume = 0.0
            side = "none"
            
            if symbol in self._liquidations:
                for liq_price, level in self._liquidations[symbol].items():
                    if abs(liq_price - price) < step:
                        intensity = max(intensity, level.intensity)
                        volume += level.total_volume
                        side = level.side
            
            matrix.append({
                "price": price,
                "intensity": intensity,
                "side": side,
                "volume": volume
            })
        
        return matrix
    
    def get_heatmap(self, symbol: str) -> Optional[HeatmapData]:
        """Cache'lenmiş heatmap döndür"""
        return self._heatmap_cache.get(symbol)
    
    def reset(self, symbol: str = None):
        """Likidasyon verilerini sıfırla"""
        if symbol:
            if symbol in self._liquidations:
                del self._liquidations[symbol]
            if symbol in self._heatmap_cache:
                del self._heatmap_cache[symbol]
        else:
            self._liquidations.clear()
            self._heatmap_cache.clear()
