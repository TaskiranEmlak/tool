# HFT Trading Tools - Volatilite Scanner
"""
Binance FUTURES tarayıp en volatil coinleri bulan modül.
1 dakikalık mum değişimine göre sıralama yapar.
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CoinVolatility:
    """Coin volatilite verisi"""
    symbol: str
    price: float
    prev_close: float              # Önceki mumun kapanışı
    price_change_percent: float    # 1m mum değişimi %
    volume_24h: float              # 24h hacim
    high_1m: float                 # 1m yüksek
    low_1m: float                  # 1m düşük
    timestamp: datetime


class VolatilityScanner:
    """
    Binance FUTURES tarayıcı.
    1 dakikalık mum değişimine göre en volatil coinleri bulur.
    """
    
    def __init__(self):
        # FUTURES API
        self.base_url = "https://fapi.binance.com"
        
        # Parametreler
        self.min_volume_usdt = 10_000_000   # Min 10M$ 24h hacim
        self.min_change_percent = 0.3       # Min %0.3 1m değişim
        self.top_n = 15                     # En volatil 15 coin
        self.scan_interval = 10             # Her 10 saniyede bir tara
        
        # Cache
        self._all_symbols: List[str] = []
        self._last_scan: Optional[datetime] = None
        self._results: List[CoinVolatility] = []
    
    async def get_all_futures_pairs(self) -> List[str]:
        """Tüm USDT perpetual çiftlerini getir"""
        if self._all_symbols:
            return self._all_symbols
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fapi/v1/exchangeInfo"
            async with session.get(url) as response:
                data = await response.json()
                
                symbols = []
                for s in data.get("symbols", []):
                    if (s.get("quoteAsset") == "USDT" and 
                        s.get("contractType") == "PERPETUAL" and
                        s.get("status") == "TRADING"):
                        symbols.append(s["symbol"])
                
                self._all_symbols = symbols
                print(f"[Scanner] {len(symbols)} USDT Perpetual çifti bulundu")
                return symbols
    
    async def get_24h_tickers(self) -> Dict[str, Dict]:
        """24 saatlik ticker verilerini çek (hacim filtresi için)"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            async with session.get(url) as response:
                tickers = await response.json()
                return {t["symbol"]: t for t in tickers}
    
    async def get_1m_klines_batch(self, symbols: List[str]) -> Dict[str, List]:
        """Birden fazla sembol için 1m mum verisi çek"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            # Paralel istekler
            tasks = []
            for symbol in symbols:
                url = f"{self.base_url}/fapi/v1/klines"
                params = {"symbol": symbol, "interval": "1m", "limit": 2}
                tasks.append(self._fetch_kline(session, symbol, url, params))
            
            klines_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, klines in zip(symbols, klines_list):
                if not isinstance(klines, Exception) and klines:
                    results[symbol] = klines
        
        return results
    
    async def _fetch_kline(self, session, symbol, url, params):
        """Tek kline isteği"""
        try:
            async with session.get(url, params=params) as response:
                return await response.json()
        except:
            return None
    
    async def scan_1m_volatility(self) -> List[CoinVolatility]:
        """
        1 dakikalık mum değişimine göre tarama.
        
        Returns:
            En volatil coinler (1m değişime göre sıralı)
        """
        print(f"[Scanner] 1m mum taraması başlatılıyor... ({datetime.now().strftime('%H:%M:%S')})")
        
        # 1. Tüm futures çiftlerini al
        symbols = await self.get_all_futures_pairs()
        
        # 2. 24h ticker (hacim filtresi için)
        tickers = await self.get_24h_tickers()
        
        # 3. Yeterli hacimli coinleri filtrele
        high_volume_symbols = []
        for s in symbols:
            ticker = tickers.get(s, {})
            volume = float(ticker.get("quoteVolume", 0))
            if volume >= self.min_volume_usdt:
                high_volume_symbols.append(s)
        
        print(f"[Scanner] {len(high_volume_symbols)} yüksek hacimli coin kontrol ediliyor...")
        
        # 4. 1m kline verilerini çek (batch)
        klines_data = await self.get_1m_klines_batch(high_volume_symbols)
        
        # 5. Volatilite hesapla
        volatile_coins = []
        
        for symbol, klines in klines_data.items():
            if not klines or len(klines) < 2:
                continue
            
            try:
                # Önceki mum ve mevcut mum
                prev_candle = klines[0]
                curr_candle = klines[1]
                
                prev_close = float(prev_candle[4])  # Önceki kapanış
                curr_close = float(curr_candle[4])  # Şimdiki fiyat
                curr_high = float(curr_candle[2])
                curr_low = float(curr_candle[3])
                
                if prev_close == 0:
                    continue
                
                # 1m değişim yüzdesi
                change_percent = ((curr_close - prev_close) / prev_close) * 100
                
                # Hacim
                ticker = tickers.get(symbol, {})
                volume_24h = float(ticker.get("quoteVolume", 0))
                
                # Minimum değişim filtresi
                if abs(change_percent) >= self.min_change_percent:
                    volatile_coins.append(CoinVolatility(
                        symbol=symbol,
                        price=curr_close,
                        prev_close=prev_close,
                        price_change_percent=change_percent,
                        volume_24h=volume_24h,
                        high_1m=curr_high,
                        low_1m=curr_low,
                        timestamp=datetime.now()
                    ))
            
            except (IndexError, ValueError, TypeError) as e:
                continue
        
        # 6. Volatiliteye göre sırala (mutlak değer)
        volatile_coins.sort(key=lambda x: abs(x.price_change_percent), reverse=True)
        
        # 7. En volatil N coin
        self._results = volatile_coins[:self.top_n]
        self._last_scan = datetime.now()
        
        print(f"[Scanner] {len(volatile_coins)} volatil coin, top {len(self._results)} seçildi")
        
        # İlk 5'i göster
        for i, coin in enumerate(self._results[:5], 1):
            icon = "🟢" if coin.price_change_percent > 0 else "🔴"
            print(f"   {i}. {icon} {coin.symbol}: {coin.price_change_percent:+.2f}% (1m)")
        
        return self._results
    
    # Eski metodu koruyalım (24h)
    async def scan_volatility(self) -> List[CoinVolatility]:
        """1m taramaya yönlendir"""
        return await self.scan_1m_volatility()
    
    def get_results(self) -> List[CoinVolatility]:
        """Son tarama sonuçlarını döndür"""
        return self._results
    
    def get_symbols(self) -> List[str]:
        """Seçilen sembolleri döndür"""
        return [c.symbol for c in self._results]


async def test_scanner():
    """Scanner test"""
    scanner = VolatilityScanner()
    
    print("\n" + "="*50)
    print("FUTURES 1M VOLATILITE TARAMASI")
    print("="*50)
    
    results = await scanner.scan_1m_volatility()
    
    print("\n📊 Detaylı Sonuçlar:")
    print("-" * 60)
    
    for i, coin in enumerate(results, 1):
        icon = "🟢" if coin.price_change_percent > 0 else "🔴"
        print(f"{i:2}. {icon} {coin.symbol:12} | "
              f"Değişim: {coin.price_change_percent:+6.2f}% | "
              f"Fiyat: ${coin.price:>10,.4f} | "
              f"Önceki: ${coin.prev_close:>10,.4f}")


if __name__ == "__main__":
    asyncio.run(test_scanner())
