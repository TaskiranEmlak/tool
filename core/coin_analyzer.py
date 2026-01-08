# HFT Trading Tools - Coin Analyzer
"""
Seçilen coin için detaylı analiz yapan modül.
Türkçe yorumlarla birlikte tüm verileri döndürür.
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TimeframeAnalysis:
    """Tek timeframe analizi"""
    timeframe: str          # "15m", "5m", "3m", "1m"
    trend: str              # "up", "down", "neutral"
    change_percent: float   # Değişim %
    candle_count: int       # Kaç mum analiz edildi
    commentary: str         # Türkçe yorum


@dataclass 
class DepthLevel:
    """Order book seviyesi"""
    price: float
    quantity: float
    total_usdt: float


@dataclass
class CoinAnalysis:
    """Tam coin analizi"""
    symbol: str
    price: float
    
    # 24h veriler
    change_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float
    
    # Timeframe analizleri
    tf_15m: Optional[TimeframeAnalysis] = None
    tf_5m: Optional[TimeframeAnalysis] = None
    tf_3m: Optional[TimeframeAnalysis] = None
    tf_1m: Optional[TimeframeAnalysis] = None
    tf_summary: str = ""  # Özet yorum
    
    # Alış/Satış baskısı
    bid_pressure: float = 0.0   # 0-100
    ask_pressure: float = 0.0   # 0-100
    pressure_comment: str = ""
    
    # Order book
    top_bids: List[DepthLevel] = field(default_factory=list)
    top_asks: List[DepthLevel] = field(default_factory=list)
    bid_wall: Optional[str] = None   # Duvar varsa açıklama
    ask_wall: Optional[str] = None
    
    # Market verileri
    funding_rate: float = 0.0
    funding_comment: str = ""
    long_percent: float = 50.0
    short_percent: float = 50.0
    ls_comment: str = ""
    open_interest: float = 0.0
    oi_change: float = 0.0
    oi_comment: str = ""
    
    # Genel sonuç
    risk_level: str = ""        # "düşük", "orta", "yüksek"
    final_verdict: str = ""     # Son yorum
    
    timestamp: datetime = field(default_factory=datetime.now)


class CoinAnalyzer:
    """
    Coin için detaylı analiz yapan sınıf.
    """
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self._all_symbols: List[str] = []
    
    async def get_all_symbols(self) -> List[Tuple[str, float, float]]:
        """
        Tüm futures sembollerini getir.
        
        Returns:
            [(symbol, price, change_24h), ...]
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/fapi/v1/ticker/24hr"
                async with session.get(url) as response:
                    tickers = await response.json()
                    
                    result = []
                    for t in tickers:
                        symbol = t.get("symbol", "")
                        if symbol.endswith("USDT") and "USDC" not in symbol:
                            price = float(t.get("lastPrice", 0))
                            change = float(t.get("priceChangePercent", 0))
                            result.append((symbol, price, change))
                    
                    # Hacme göre sırala
                    result.sort(key=lambda x: abs(x[2]), reverse=True)
                    self._all_symbols = [r[0] for r in result]
                    return result
                    
        except Exception as e:
            print(f"[Analyzer] Sembol listesi hatası: {e}")
            return []
    
    async def analyze(self, symbol: str) -> CoinAnalysis:
        """
        Coin için tam analiz yap.
        """
        analysis = CoinAnalysis(symbol=symbol, price=0, change_24h=0, 
                               volume_24h=0, high_24h=0, low_24h=0)
        
        async with aiohttp.ClientSession() as session:
            # 1. 24h ticker
            await self._fetch_ticker(session, symbol, analysis)
            
            # 2. Timeframe analizleri
            await self._analyze_timeframes(session, symbol, analysis)
            
            # 3. Order book
            await self._analyze_orderbook(session, symbol, analysis)
            
            # 4. Market verileri
            await self._fetch_market_data(session, symbol, analysis)
            
            # 5. Final değerlendirme
            self._generate_verdict(analysis)
        
        return analysis
    
    async def _fetch_ticker(self, session, symbol: str, analysis: CoinAnalysis):
        """24h ticker verisi"""
        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr?symbol={symbol}"
            async with session.get(url) as response:
                data = await response.json()
                analysis.price = float(data.get("lastPrice", 0))
                analysis.change_24h = float(data.get("priceChangePercent", 0))
                analysis.volume_24h = float(data.get("quoteVolume", 0))
                analysis.high_24h = float(data.get("highPrice", 0))
                analysis.low_24h = float(data.get("lowPrice", 0))
        except:
            pass
    
    async def _analyze_timeframes(self, session, symbol: str, analysis: CoinAnalysis):
        """Tüm timeframe'leri analiz et"""
        timeframes = [("15m", 4), ("5m", 4), ("3m", 4), ("1m", 6)]
        
        for tf, limit in timeframes:
            try:
                url = f"{self.base_url}/fapi/v1/klines"
                params = {"symbol": symbol, "interval": tf, "limit": limit}
                
                async with session.get(url, params=params) as response:
                    klines = await response.json()
                    
                    if len(klines) >= 3:
                        tf_analysis = self._analyze_single_tf(tf, klines)
                        
                        if tf == "15m":
                            analysis.tf_15m = tf_analysis
                        elif tf == "5m":
                            analysis.tf_5m = tf_analysis
                        elif tf == "3m":
                            analysis.tf_3m = tf_analysis
                        elif tf == "1m":
                            analysis.tf_1m = tf_analysis
            except:
                pass
        
        # TF özeti oluştur
        analysis.tf_summary = self._generate_tf_summary(analysis)
    
    def _analyze_single_tf(self, tf: str, klines: list) -> TimeframeAnalysis:
        """Tek timeframe analizi"""
        closes = [float(k[4]) for k in klines]
        opens = [float(k[1]) for k in klines]
        
        first_open = opens[0]
        last_close = closes[-1]
        change = ((last_close - first_open) / first_open) * 100 if first_open > 0 else 0
        
        # Trend belirleme
        up_candles = sum(1 for i in range(len(closes)) if closes[i] > opens[i])
        down_candles = len(closes) - up_candles
        
        if change > 0.2 and up_candles >= len(closes) // 2:
            trend = "up"
            commentary = f"{tf} yukari trend ({change:+.2f}%)"
        elif change < -0.2 and down_candles >= len(closes) // 2:
            trend = "down"
            commentary = f"{tf} asagi trend ({change:+.2f}%)"
        else:
            trend = "neutral"
            commentary = f"{tf} yatay/kararsiz"
        
        return TimeframeAnalysis(
            timeframe=tf,
            trend=trend,
            change_percent=change,
            candle_count=len(klines),
            commentary=commentary
        )
    
    def _generate_tf_summary(self, analysis: CoinAnalysis) -> str:
        """Timeframe özet yorumu"""
        trends = []
        if analysis.tf_15m:
            trends.append(("15m", analysis.tf_15m.trend))
        if analysis.tf_5m:
            trends.append(("5m", analysis.tf_5m.trend))
        if analysis.tf_3m:
            trends.append(("3m", analysis.tf_3m.trend))
        if analysis.tf_1m:
            trends.append(("1m", analysis.tf_1m.trend))
        
        up_count = sum(1 for _, t in trends if t == "up")
        down_count = sum(1 for _, t in trends if t == "down")
        
        if up_count >= 3:
            return "GUCLU YUKSELIS: Cogu timeframe yukari gosteriyor"
        elif down_count >= 3:
            return "GUCLU DUSUS: Cogu timeframe asagi gosteriyor"
        elif up_count == 2 and down_count == 0:
            return "UYUMLU YUKSELIS: Yukselis trendi var"
        elif down_count == 2 and up_count == 0:
            return "UYUMLU DUSUS: Dusus trendi var"
        else:
            return "KARISIK: Timeframe'ler farkli yonlerde, dikkatli ol"
    
    async def _analyze_orderbook(self, session, symbol: str, analysis: CoinAnalysis):
        """Order book analizi"""
        try:
            url = f"{self.base_url}/fapi/v1/depth?symbol={symbol}&limit=20"
            async with session.get(url) as response:
                data = await response.json()
                
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                
                # İlk 10 seviye
                for b in bids[:10]:
                    price = float(b[0])
                    qty = float(b[1])
                    analysis.top_bids.append(DepthLevel(price, qty, price * qty))
                
                for a in asks[:10]:
                    price = float(a[0])
                    qty = float(a[1])
                    analysis.top_asks.append(DepthLevel(price, qty, price * qty))
                
                # Toplam hacim
                bid_total = sum(b[1] for b in bids[:10])
                ask_total = sum(a[1] for a in asks[:10])
                total = bid_total + ask_total
                
                if total > 0:
                    analysis.bid_pressure = (bid_total / total) * 100
                    analysis.ask_pressure = (ask_total / total) * 100
                
                # Yorum
                if analysis.bid_pressure > 65:
                    analysis.pressure_comment = "GUCLU ALIS BASKISI: Alicilar agirlikta"
                elif analysis.bid_pressure > 55:
                    analysis.pressure_comment = "Hafif alis baskisi"
                elif analysis.ask_pressure > 65:
                    analysis.pressure_comment = "GUCLU SATIS BASKISI: Saticilar agirlikta"
                elif analysis.ask_pressure > 55:
                    analysis.pressure_comment = "Hafif satis baskisi"
                else:
                    analysis.pressure_comment = "Dengeli: Alis ve satis esit"
                
                # Duvar tespiti
                if analysis.top_bids:
                    max_bid = max(analysis.top_bids, key=lambda x: x.total_usdt)
                    avg_bid = sum(b.total_usdt for b in analysis.top_bids) / len(analysis.top_bids)
                    if max_bid.total_usdt > avg_bid * 3:
                        analysis.bid_wall = f"ALIS DUVARI: ${max_bid.price:,.2f} seviyesinde buyuk emir"
                
                if analysis.top_asks:
                    max_ask = max(analysis.top_asks, key=lambda x: x.total_usdt)
                    avg_ask = sum(a.total_usdt for a in analysis.top_asks) / len(analysis.top_asks)
                    if max_ask.total_usdt > avg_ask * 3:
                        analysis.ask_wall = f"SATIS DUVARI: ${max_ask.price:,.2f} seviyesinde buyuk emir"
                        
        except Exception as e:
            pass
    
    async def _fetch_market_data(self, session, symbol: str, analysis: CoinAnalysis):
        """Market verileri (Funding, L/S, OI)"""
        try:
            # Funding
            url = f"{self.base_url}/fapi/v1/premiumIndex?symbol={symbol}"
            async with session.get(url) as response:
                data = await response.json()
                analysis.funding_rate = float(data.get("lastFundingRate", 0))
                
                rate_pct = analysis.funding_rate * 100
                if rate_pct > 0.05:
                    analysis.funding_comment = f"YUKSEK FUNDING ({rate_pct:.3f}%): Long'cular odeme yapiyor, short baskisi olabilir"
                elif rate_pct > 0.01:
                    analysis.funding_comment = f"Normal funding ({rate_pct:.3f}%)"
                elif rate_pct < -0.01:
                    analysis.funding_comment = f"NEGATIF FUNDING ({rate_pct:.3f}%): Short'cular odeme yapiyor, long firsati"
                else:
                    analysis.funding_comment = f"Notr funding ({rate_pct:.3f}%)"
            
            # Long/Short Ratio
            url = f"{self.base_url}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
            async with session.get(url) as response:
                data = await response.json()
                if data:
                    analysis.long_percent = float(data[0].get("longAccount", 0.5)) * 100
                    analysis.short_percent = float(data[0].get("shortAccount", 0.5)) * 100
                    
                    if analysis.long_percent > 70:
                        analysis.ls_comment = f"DIKKAT! Cok fazla long ({analysis.long_percent:.0f}%). Tersine donebilir!"
                    elif analysis.long_percent > 60:
                        analysis.ls_comment = f"Long agirlikli ({analysis.long_percent:.0f}%)"
                    elif analysis.short_percent > 70:
                        analysis.ls_comment = f"DIKKAT! Cok fazla short ({analysis.short_percent:.0f}%). Short squeeze olabilir!"
                    elif analysis.short_percent > 60:
                        analysis.ls_comment = f"Short agirlikli ({analysis.short_percent:.0f}%)"
                    else:
                        analysis.ls_comment = f"Dengeli pozisyonlar ({analysis.long_percent:.0f}%/{analysis.short_percent:.0f}%)"
            
            # Open Interest
            url = f"{self.base_url}/fapi/v1/openInterest?symbol={symbol}"
            async with session.get(url) as response:
                data = await response.json()
                oi = float(data.get("openInterest", 0))
                analysis.open_interest = oi * analysis.price
                analysis.oi_comment = f"Acik pozisyon: ${analysis.open_interest/1e9:.2f}B"
                
        except Exception as e:
            pass
    
    def _generate_verdict(self, analysis: CoinAnalysis):
        """Final değerlendirme"""
        risk_points = 0
        reasons = []
        
        # L/S kontrolü
        if analysis.long_percent > 70:
            risk_points += 2
            reasons.append("Long cok kalabalik")
        elif analysis.short_percent > 70:
            risk_points += 2
            reasons.append("Short cok kalabalik")
        
        # Funding kontrolü
        if abs(analysis.funding_rate) > 0.0005:
            risk_points += 1
            reasons.append("Funding yuksek")
        
        # TF uyumsuzluk
        if analysis.tf_summary and "KARISIK" in analysis.tf_summary:
            risk_points += 1
            reasons.append("TF'ler uyumsuz")
        
        # Risk seviyesi
        if risk_points >= 3:
            analysis.risk_level = "YUKSEK"
        elif risk_points >= 1:
            analysis.risk_level = "ORTA"
        else:
            analysis.risk_level = "DUSUK"
        
        # Final yorum
        if analysis.risk_level == "YUKSEK":
            analysis.final_verdict = f"RISKLI! {', '.join(reasons)}. Islem yapmadan once bekle."
        elif analysis.risk_level == "ORTA":
            analysis.final_verdict = f"DIKKATLI OL: {', '.join(reasons) if reasons else 'Bazi belirsizlikler var'}."
        else:
            if analysis.tf_summary and "YUKSELIS" in analysis.tf_summary:
                analysis.final_verdict = "LONG ICIN UYGUN gorunuyor. Trend ve veriler olumlu."
            elif analysis.tf_summary and "DUSUS" in analysis.tf_summary:
                analysis.final_verdict = "SHORT ICIN UYGUN gorunuyor. Trend ve veriler olumsuz."
            else:
                analysis.final_verdict = "NOTR. Belirgin bir sinyal yok, bekle."
