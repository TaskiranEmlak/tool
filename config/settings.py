# HFT Trading Tools - Temel Konfigürasyon
"""
Kripto piyasaları için yüksek frekanslı ticaret araçları.
CVD, OBI, Likidasyon Heatmap ve sinyal üretimi.
"""

from dataclasses import dataclass
from typing import List, Dict
import os

@dataclass
class Settings:
    """Ana konfigürasyon sınıfı"""
    
    # İzlenecek semboller
    SYMBOLS: List[str] = None
    
    # Borsa WebSocket URL'leri
    EXCHANGES: Dict[str, str] = None
    
    # OBI Parametreleri
    OBI_DEPTH: int = 10  # Kaç kademe order book
    OBI_DECAY_FACTOR: float = 0.5  # Ağırlık azalma katsayısı
    OBI_STRONG_THRESHOLD: float = 0.3  # Güçlü sinyal eşiği
    
    # CVD Parametreleri  
    CVD_WINDOW_5M: int = 300  # 5 dakikalık pencere (saniye)
    CVD_WINDOW_15M: int = 900  # 15 dakikalık pencere
    CVD_WINDOW_1H: int = 3600  # 1 saatlik pencere
    
    # Likidasyon Parametreleri
    LIQUIDATION_CLUSTER_PERCENT: float = 0.5  # Kümeleme aralığı %
    LIQUIDATION_DECAY_HOURS: int = 24  # Zaman aşımı
    
    # Sinyal Parametreleri
    SIGNAL_MIN_CONFIDENCE: float = 0.6
    SIGNAL_COOLDOWN_SECONDS: int = 60  # Aynı yönde sinyal arası bekleme
    
    # Dashboard
    DASHBOARD_REFRESH_MS: int = 100  # Yenileme hızı
    SOUND_ENABLED: bool = True
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = [
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
                "BNBUSDT",
                "XRPUSDT"
            ]
        
        if self.EXCHANGES is None:
            self.EXCHANGES = {
                "binance": {
                    "ws_base": "wss://stream.binance.com:9443/ws",
                    "ws_futures": "wss://fstream.binance.com/ws",
                    "rest_base": "https://api.binance.com",
                    "rest_futures": "https://fapi.binance.com"
                },
                "bybit": {
                    "ws_base": "wss://stream.bybit.com/v5/public/linear",
                    "rest_base": "https://api.bybit.com"
                }
            }


# Global settings instance
settings = Settings()


# Yardımcı fonksiyonlar
def get_binance_trade_stream(symbol: str) -> str:
    """Binance trade WebSocket stream URL'i"""
    return f"{settings.EXCHANGES['binance']['ws_base']}/{symbol.lower()}@aggTrade"


def get_binance_depth_stream(symbol: str, levels: int = 10) -> str:
    """Binance order book WebSocket stream URL'i"""
    return f"{settings.EXCHANGES['binance']['ws_base']}/{symbol.lower()}@depth{levels}@100ms"


def get_binance_liquidation_stream() -> str:
    """Binance likidasyon stream URL'i (futures)"""
    return f"{settings.EXCHANGES['binance']['ws_futures']}/!forceOrder@arr"


def get_symbol_pairs() -> List[str]:
    """Aktif sembol listesi"""
    return settings.SYMBOLS
