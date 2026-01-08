# Ultimate Scalping Dashboard - Helper Tools
"""
Gözcü araçları: Sesli uyarı, Korelasyon, Market Rejimi, Risk Hesap
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np
import threading


# ===========================
# SESLI UYARI (Whale Hunter)
# ===========================

class VoiceAlert:
    """
    Sesli uyarı sistemi.
    Önemli olaylarda konuşarak uyarır.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._engine = None
        self._init_engine()
    
    def _init_engine(self):
        """TTS motorunu başlat"""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 180)  # Hız
            self._engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"[Voice] TTS baslatilamadi: {e}")
            self.enabled = False
    
    def speak(self, text: str):
        """Sesli uyarı ver"""
        if not self.enabled or not self._engine:
            return
        
        def _speak():
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except:
                pass
        
        # Background thread'de çalıştır
        threading.Thread(target=_speak, daemon=True).start()
    
    def whale_alert(self, symbol: str, direction: str):
        """Balina uyarısı"""
        if direction == "buy":
            self.speak(f"{symbol.replace('USDT', '')} de devasa alım var!")
        else:
            self.speak(f"{symbol.replace('USDT', '')} de büyük satış!")
    
    def signal_alert(self, symbol: str, direction: str, score: int):
        """Sinyal uyarısı"""
        dir_text = "Long" if direction == "up" else "Short"
        self.speak(f"{symbol.replace('USDT', '')} {dir_text} sinyali, skor {score}")


# ===========================
# KORELASYON MATRİSİ
# ===========================

class CorrelationTracker:
    """
    BTC ile korelasyon takibi.
    Coin BTC'den bağımsız mı hareket ediyor?
    """
    
    def __init__(self, window: int = 60):
        self.window = window  # 60 saniyelik pencere
        self.btc_prices: deque = deque(maxlen=window)
        self.coin_prices: Dict[str, deque] = {}
    
    def update_btc(self, price: float):
        """BTC fiyatını güncelle"""
        self.btc_prices.append(price)
    
    def update_coin(self, symbol: str, price: float):
        """Coin fiyatını güncelle"""
        if symbol not in self.coin_prices:
            self.coin_prices[symbol] = deque(maxlen=self.window)
        self.coin_prices[symbol].append(price)
    
    def get_correlation(self, symbol: str) -> Tuple[float, str]:
        """
        Korelasyon hesapla.
        
        Returns:
            (korelasyon_katsayısı, yorum)
        """
        if symbol not in self.coin_prices:
            return 0.0, "Veri yok"
        
        btc = list(self.btc_prices)
        coin = list(self.coin_prices[symbol])
        
        min_len = min(len(btc), len(coin))
        if min_len < 10:
            return 0.0, "Yetersiz veri"
        
        btc = btc[-min_len:]
        coin = coin[-min_len:]
        
        # Pearson korelasyon
        try:
            btc_returns = np.diff(btc) / btc[:-1]
            coin_returns = np.diff(coin) / coin[:-1]
            
            corr = np.corrcoef(btc_returns, coin_returns)[0, 1]
            
            if np.isnan(corr):
                return 0.0, "Hesaplanamadı"
            
            # Yorum
            if abs(corr) < 0.3:
                comment = "Bagimsiz hareket (Guclu)"
            elif abs(corr) < 0.6:
                comment = "Kismen bagimli"
            elif abs(corr) < 0.8:
                comment = "BTC ile uyumlu"
            else:
                comment = "BTC kopyasi (Riskli)"
            
            return float(corr), comment
            
        except Exception as e:
            return 0.0, "Hata"


# ===========================
# MARKET REJİMİ (Trafik Işığı)
# ===========================

@dataclass
class MarketRegime:
    """Market rejim durumu"""
    regime: str          # "bullish", "bearish", "neutral"
    color: str           # "green", "red", "yellow"
    up_percent: float    # Yükselen coin %
    down_percent: float  # Düşen coin %
    comment: str         # Türkçe yorum
    btc_price: float = 0
    btc_change: float = 0
    dominance: float = 0


class MarketRegimeDetector:
    """
    Piyasa rejimi tespit.
    Top 50 coin'e bakarak genel yönü belirler.
    """
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self._last_regime: Optional[MarketRegime] = None
    
    async def detect(self) -> MarketRegime:
        """Market rejimini tespit et"""
        try:
            async with aiohttp.ClientSession() as session:
                # 24h ticker
                url = f"{self.base_url}/fapi/v1/ticker/24hr"
                async with session.get(url) as response:
                    tickers = await response.json()
                
                # USDT çiftlerini filtrele ve hacme göre sırala
                usdt_pairs = [t for t in tickers if t["symbol"].endswith("USDT") 
                             and "USDC" not in t["symbol"]]
                usdt_pairs.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
                
                # Top 50
                top_50 = usdt_pairs[:50]
                
                # Yön analizi
                up_count = 0
                down_count = 0
                btc_price = 0
                btc_change = 0
                
                for t in top_50:
                    change = float(t.get("priceChangePercent", 0))
                    if change > 0:
                        up_count += 1
                    else:
                        down_count += 1
                    
                    if t["symbol"] == "BTCUSDT":
                        btc_price = float(t.get("lastPrice", 0))
                        btc_change = change
                
                up_pct = (up_count / 50) * 100
                down_pct = (down_count / 50) * 100
                
                # Rejim belirleme
                if up_pct >= 60:
                    regime = "bullish"
                    color = "green"
                    comment = "Piyasa yukari, LONG firsatlarina odaklan"
                elif down_pct >= 60:
                    regime = "bearish"
                    color = "red"
                    comment = "Piyasa asagi, SHORT firsatlarina odaklan"
                else:
                    regime = "neutral"
                    color = "yellow"
                    comment = "Piyasa kararsiz, dikkatli ol veya bekle"
                
                self._last_regime = MarketRegime(
                    regime=regime,
                    color=color,
                    up_percent=up_pct,
                    down_percent=down_pct,
                    comment=comment,
                    btc_price=btc_price,
                    btc_change=btc_change
                )
                
                return self._last_regime
                
        except Exception as e:
            return MarketRegime(
                regime="unknown",
                color="gray",
                up_percent=0,
                down_percent=0,
                comment=f"Hata: {e}"
            )
    
    def get_last(self) -> Optional[MarketRegime]:
        """Son rejimi döndür"""
        return self._last_regime


# ===========================
# RİSK HESAPLAYICI
# ===========================

@dataclass
class RiskCalculation:
    """Risk hesap sonucu"""
    symbol: str
    atr_percent: float       # ATR yüzdesi
    suggested_sl: float      # Önerilen stop-loss %
    max_position: float      # Max pozisyon ($)
    leverage_safe: int       # Güvenli kaldıraç
    comment: str


class RiskCalculator:
    """
    Dinamik risk hesaplayıcı.
    ATR bazlı pozisyon büyüklüğü önerir.
    """
    
    def __init__(self, account_size: float = 1000):
        self.account_size = account_size
        self.risk_per_trade = 0.02  # Trade başına %2 risk
        self.base_url = "https://fapi.binance.com"
    
    def set_account_size(self, size: float):
        """Hesap büyüklüğünü güncelle"""
        self.account_size = size
    
    async def calculate(self, symbol: str) -> RiskCalculation:
        """Risk hesapla"""
        try:
            async with aiohttp.ClientSession() as session:
                # Son 14 mum (1 saat = 14 x 4dk veya 5m x 14)
                url = f"{self.base_url}/fapi/v1/klines"
                params = {"symbol": symbol, "interval": "5m", "limit": 14}
                
                async with session.get(url, params=params) as response:
                    klines = await response.json()
                
                if len(klines) < 14:
                    return RiskCalculation(
                        symbol=symbol, atr_percent=0, suggested_sl=1.0,
                        max_position=self.account_size, leverage_safe=1,
                        comment="Yetersiz veri"
                    )
                
                # ATR hesapla
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                closes = [float(k[4]) for k in klines]
                
                tr_list = []
                for i in range(1, len(klines)):
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1])
                    )
                    tr_list.append(tr)
                
                atr = sum(tr_list) / len(tr_list)
                current_price = closes[-1]
                atr_percent = (atr / current_price) * 100
                
                # Stop-loss önerisi (1.5 ATR)
                suggested_sl = atr_percent * 1.5
                
                # Max pozisyon
                risk_amount = self.account_size * self.risk_per_trade
                max_position = risk_amount / (suggested_sl / 100)
                
                # Güvenli kaldıraç
                leverage_safe = max(1, int(100 / (suggested_sl * 2)))
                
                # Yorum
                if suggested_sl < 0.5:
                    comment = f"Dusuk volatilite, SL %{suggested_sl:.2f}"
                elif suggested_sl < 1.0:
                    comment = f"Normal volatilite, SL %{suggested_sl:.2f}"
                else:
                    comment = f"Yuksek volatilite! SL %{suggested_sl:.2f}, dikkatli ol"
                
                return RiskCalculation(
                    symbol=symbol,
                    atr_percent=atr_percent,
                    suggested_sl=suggested_sl,
                    max_position=max_position,
                    leverage_safe=leverage_safe,
                    comment=comment
                )
                
        except Exception as e:
            return RiskCalculation(
                symbol=symbol, atr_percent=0, suggested_sl=1.0,
                max_position=self.account_size, leverage_safe=1,
                comment=f"Hata: {e}"
            )
