# HFT Trading Tools - Market Data (Funding, OI, L/S Ratio)
"""
Binance Futures'tan ek piyasa verilerini çeken modül.

API Endpoints:
- Funding Rate: /fapi/v1/fundingRate
- Open Interest: /fapi/v1/openInterest  
- Long/Short Ratio: /futures/data/topLongShortAccountRatio
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarketMetrics:
    """Piyasa metrikleri"""
    symbol: str
    
    # Funding Rate
    funding_rate: float = 0.0           # Mevcut oran (örn: 0.0001 = %0.01)
    funding_rate_annual: float = 0.0    # Yıllık (x3 x 365)
    next_funding_time: Optional[datetime] = None
    
    # Open Interest
    open_interest: float = 0.0          # Açık pozisyon (coin cinsinden)
    open_interest_usdt: float = 0.0     # USDT cinsinden
    oi_change_5m: float = 0.0           # Son 5dk değişim %
    
    # Long/Short Ratio
    long_short_ratio: float = 1.0       # Long/Short oranı
    long_percent: float = 50.0          # Long %
    short_percent: float = 50.0         # Short %
    
    # Analiz
    signal: str = ""                    # "bullish", "bearish", "neutral"
    signal_strength: float = 0.0        # 0-100
    reasons: List[str] = None
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MarketDataFetcher:
    """
    Piyasa verisi çekici.
    Funding Rate, Open Interest, Long/Short Ratio.
    """
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        
        # OI geçmişi (5dk değişim için)
        self._oi_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Cache
        self._cache: Dict[str, MarketMetrics] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 30  # 30 saniye cache
    
    async def get_funding_rate(self, symbol: str) -> Tuple[float, Optional[datetime]]:
        """
        Funding rate çek.
        
        Returns:
            (funding_rate, next_funding_time)
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/fapi/v1/premiumIndex"
                params = {"symbol": symbol}
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if "lastFundingRate" in data:
                        rate = float(data["lastFundingRate"])
                        next_time = datetime.fromtimestamp(
                            int(data.get("nextFundingTime", 0)) / 1000
                        ) if data.get("nextFundingTime") else None
                        
                        return rate, next_time
                    
                    return 0.0, None
                    
        except Exception as e:
            print(f"[MarketData] Funding rate hatası ({symbol}): {e}")
            return 0.0, None
    
    async def get_open_interest(self, symbol: str) -> Tuple[float, float]:
        """
        Open Interest çek.
        
        Returns:
            (oi_coin, oi_usdt)
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/fapi/v1/openInterest"
                params = {"symbol": symbol}
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if "openInterest" in data:
                        oi = float(data["openInterest"])
                        
                        # USDT değeri için fiyat çek
                        price_url = f"{self.base_url}/fapi/v1/ticker/price"
                        async with session.get(price_url, params={"symbol": symbol}) as price_resp:
                            price_data = await price_resp.json()
                            price = float(price_data.get("price", 0))
                            oi_usdt = oi * price
                        
                        return oi, oi_usdt
                    
                    return 0.0, 0.0
                    
        except Exception as e:
            print(f"[MarketData] OI hatası ({symbol}): {e}")
            return 0.0, 0.0
    
    async def get_long_short_ratio(self, symbol: str) -> Tuple[float, float, float]:
        """
        Top Trader Long/Short Ratio çek.
        
        Returns:
            (ratio, long_percent, short_percent)
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/futures/data/topLongShortAccountRatio"
                params = {"symbol": symbol, "period": "5m", "limit": 1}
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data and len(data) > 0:
                        item = data[0]
                        ratio = float(item.get("longShortRatio", 1.0))
                        long_pct = float(item.get("longAccount", 0.5)) * 100
                        short_pct = float(item.get("shortAccount", 0.5)) * 100
                        
                        return ratio, long_pct, short_pct
                    
                    return 1.0, 50.0, 50.0
                    
        except Exception as e:
            print(f"[MarketData] L/S Ratio hatası ({symbol}): {e}")
            return 1.0, 50.0, 50.0
    
    async def get_all_metrics(self, symbol: str) -> MarketMetrics:
        """
        Tüm metrikleri çek ve analiz et.
        
        Returns:
            MarketMetrics objesi
        """
        # Cache kontrolü
        if symbol in self._cache:
            cache_age = (datetime.now() - self._cache_time.get(symbol, datetime.min)).total_seconds()
            if cache_age < self._cache_ttl:
                return self._cache[symbol]
        
        # Paralel çek
        funding_task = self.get_funding_rate(symbol)
        oi_task = self.get_open_interest(symbol)
        ls_task = self.get_long_short_ratio(symbol)
        
        (funding_rate, next_funding), (oi, oi_usdt), (ls_ratio, long_pct, short_pct) = await asyncio.gather(
            funding_task, oi_task, ls_task
        )
        
        # OI değişim hesapla
        oi_change = self._calculate_oi_change(symbol, oi)
        
        # Metrics oluştur
        metrics = MarketMetrics(
            symbol=symbol,
            funding_rate=funding_rate,
            funding_rate_annual=funding_rate * 3 * 365 * 100,  # Yıllık %
            next_funding_time=next_funding,
            open_interest=oi,
            open_interest_usdt=oi_usdt,
            oi_change_5m=oi_change,
            long_short_ratio=ls_ratio,
            long_percent=long_pct,
            short_percent=short_pct
        )
        
        # Sinyal analizi
        self._analyze_metrics(metrics)
        
        # Cache'e kaydet
        self._cache[symbol] = metrics
        self._cache_time[symbol] = datetime.now()
        
        return metrics
    
    def _calculate_oi_change(self, symbol: str, current_oi: float) -> float:
        """OI değişim yüzdesini hesapla"""
        now = datetime.now()
        
        if symbol not in self._oi_history:
            self._oi_history[symbol] = []
        
        # Geçmişe ekle
        self._oi_history[symbol].append((now, current_oi))
        
        # 5 dakikadan eski kayıtları sil
        cutoff = now - timedelta(minutes=5)
        self._oi_history[symbol] = [
            (t, v) for t, v in self._oi_history[symbol] if t > cutoff
        ]
        
        # Değişim hesapla
        history = self._oi_history[symbol]
        if len(history) >= 2:
            old_oi = history[0][1]
            if old_oi > 0:
                return ((current_oi - old_oi) / old_oi) * 100
        
        return 0.0
    
    def _analyze_metrics(self, metrics: MarketMetrics):
        """Metrikleri analiz et ve sinyal üret"""
        score = 0
        reasons = []
        
        # 1. Funding Rate Analizi
        # Çok yüksek funding → Short sinyali (herkes long, terse dönüş beklenir)
        # Çok düşük funding → Long sinyali (herkes short, terse dönüş beklenir)
        if metrics.funding_rate > 0.0005:  # %0.05+ (yüksek)
            score -= 20
            reasons.append(f"⚠️ Funding yüksek ({metrics.funding_rate*100:.3f}%) - Short baskısı")
        elif metrics.funding_rate < -0.0003:  # -%0.03 (negatif)
            score += 20
            reasons.append(f"✅ Funding negatif ({metrics.funding_rate*100:.3f}%) - Long fırsatı")
        elif metrics.funding_rate > 0.0003:
            score -= 10
            reasons.append(f"Funding pozitif ({metrics.funding_rate*100:.3f}%)")
        
        # 2. Long/Short Ratio Analizi
        # Çok yüksek L/S → Herkes long, tersine dönüş beklenir
        # Çok düşük L/S → Herkes short, tersine dönüş beklenir
        if metrics.long_percent > 65:
            score -= 25
            reasons.append(f"⚠️ Aşırı Long ({metrics.long_percent:.0f}%) - Kalabalık pozisyon")
        elif metrics.long_percent < 35:
            score += 25
            reasons.append(f"✅ Aşırı Short ({metrics.short_percent:.0f}%) - Sıkışma potansiyeli")
        elif metrics.long_percent > 55:
            score -= 10
        elif metrics.long_percent < 45:
            score += 10
        
        # 3. Open Interest Değişimi
        # OI artıyor + Funding pozitif = Long pozisyonlar artıyor
        # OI artıyor + Funding negatif = Short pozisyonlar artıyor
        if abs(metrics.oi_change_5m) > 2:
            if metrics.oi_change_5m > 0:
                reasons.append(f"📈 OI artıyor (+{metrics.oi_change_5m:.1f}%) - Yeni pozisyonlar açılıyor")
                if metrics.funding_rate < 0:
                    score += 15
                    reasons.append("  → Short squeeze potansiyeli")
            else:
                reasons.append(f"📉 OI azalıyor ({metrics.oi_change_5m:.1f}%) - Pozisyonlar kapanıyor")
        
        # Sinyal belirleme
        if score >= 25:
            metrics.signal = "bullish"
        elif score <= -25:
            metrics.signal = "bearish"
        else:
            metrics.signal = "neutral"
        
        metrics.signal_strength = abs(score)
        metrics.reasons = reasons
    
    def get_cached(self, symbol: str) -> Optional[MarketMetrics]:
        """Cache'ten al"""
        return self._cache.get(symbol)


async def test_market_data():
    """Test"""
    fetcher = MarketDataFetcher()
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    print("\n" + "="*60)
    print("MARKET DATA TEST")
    print("="*60)
    
    for symbol in symbols:
        print(f"\n📊 {symbol}")
        print("-" * 40)
        
        metrics = await fetcher.get_all_metrics(symbol)
        
        print(f"  Funding Rate: {metrics.funding_rate*100:.4f}% (Yıllık: {metrics.funding_rate_annual:.1f}%)")
        print(f"  Open Interest: ${metrics.open_interest_usdt/1e9:.2f}B")
        print(f"  OI Değişim (5m): {metrics.oi_change_5m:+.2f}%")
        print(f"  Long/Short: {metrics.long_percent:.1f}% / {metrics.short_percent:.1f}%")
        print(f"  Sinyal: {metrics.signal.upper()} (güç: {metrics.signal_strength})")
        
        if metrics.reasons:
            print(f"  Gerekçeler:")
            for r in metrics.reasons:
                print(f"    {r}")


if __name__ == "__main__":
    asyncio.run(test_market_data())
