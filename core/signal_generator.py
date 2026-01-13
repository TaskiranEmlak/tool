# Signal Generator - Gerçek Zamanlı Sinyal Üretici
"""
Binance API'den veri çekip strateji ile sinyal üretir.
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hmac
import hashlib
import time
from urllib.parse import urlencode

from core.proven_strategy import get_strategy, TradingSignal, SignalType, SignalStrength

# Config'den API key'leri al
try:
    from config.api_keys import API_KEY, API_SECRET
except ImportError:
    API_KEY = ""
    API_SECRET = ""


class BinanceDataFetcher:
    """Binance'den OHLCV verisi çeker"""
    
    BASE_URL = "https://api.binance.com"
    
    # Varsayılan coinler (fallback)
    DEFAULT_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
    ]
    
    # Dinamik coin listesi (volatiliteye göre güncellenir)
    SYMBOLS = DEFAULT_SYMBOLS.copy()
    
    def __init__(self):
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_top_volatile_coins(self, limit: int = 30) -> List[Dict]:
        """
        Son 24 saatte en volatil coinleri bul
        Returns list of dicts with symbol and volatility
        """
        session = await self._get_session()
        
        # 24 saat değişim verisi çek
        url = f"{self.BASE_URL}/api/v3/ticker/24hr"
        
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # USDT çiftlerini filtrele
                    usdt_pairs = []
                    for item in data:
                        symbol = item.get('symbol', '')
                        if symbol.endswith('USDT') and not symbol.endswith('DOWNUSDT') and not symbol.endswith('UPUSDT'):
                            try:
                                quote_volume = float(item.get('quoteVolume', 0))
                                price_change_pct = float(item.get('priceChangePercent', 0))
                                last_price = float(item.get('lastPrice', 0))
                                
                                # Min volume ve fiyat filtresi
                                if quote_volume > 10_000_000 and last_price > 0.0001:
                                    usdt_pairs.append({
                                        'symbol': symbol,
                                        'change': price_change_pct,
                                        'volume': quote_volume
                                    })
                            except:
                                continue
                    
                    # Volatiliteye göre sırala (absolute value)
                    usdt_pairs.sort(key=lambda x: abs(x['change']), reverse=True)
                    
                    top_coins = usdt_pairs[:limit]
                    print(f"[DataFetcher] En volatil {len(top_coins)} coin bulundu")
                    for i, coin in enumerate(top_coins[:5]):
                        print(f"  {i+1}. {coin['symbol']}: {coin['change']:.1f}% değişim")
                    
                    # Ayrıca SYMBOLS'u güncelle
                    self.SYMBOLS = [c['symbol'] for c in top_coins]
                    
                    return top_coins
        except Exception as e:
            print(f"[DataFetcher] Volatil coin tarama hatası: {e}")
        
        return [{'symbol': s, 'change': 0} for s in self.DEFAULT_SYMBOLS]
    
    async def update_coin_list(self, limit: int = 30):
        """Coin listesini güncelle"""
        coins = await self.fetch_top_volatile_coins(limit)
        if coins:
            self.SYMBOLS = [c['symbol'] for c in coins]
            self._volatile_cache = coins  # Cache volatility data
            print(f"[DataFetcher] Coin listesi güncellendi: {len(self.SYMBOLS)} coin")
    
    def get_cached_volatility(self) -> List[Dict]:
        """Get cached volatility data"""
        return getattr(self, '_volatile_cache', [])
    
    async def fetch_klines(self, symbol: str, interval: str = "1h", limit: int = 250) -> pd.DataFrame:
        """
        Mum verisi çek
        
        interval: 1m, 5m, 15m, 1h, 4h, 1d
        """
        session = await self._get_session()
        
        url = f"{self.BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    # Numeric dönüşümler
                    df['open'] = pd.to_numeric(df['open'])
                    df['high'] = pd.to_numeric(df['high'])
                    df['low'] = pd.to_numeric(df['low'])
                    df['close'] = pd.to_numeric(df['close'])
                    df['volume'] = pd.to_numeric(df['volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    df.attrs['symbol'] = symbol
                    
                    return df
                else:
                    print(f"[DataFetcher] {symbol} hata: {resp.status}")
                    return pd.DataFrame()
        except Exception as e:
            print(f"[DataFetcher] {symbol} exception: {e}")
            return pd.DataFrame()
    
    async def fetch_all_symbols(self, interval: str = "1h") -> Dict[str, pd.DataFrame]:
        """Tüm sembollerin verisini çek"""
        results = {}
        
        for symbol in self.SYMBOLS:
            df = await self.fetch_klines(symbol, interval)
            if not df.empty:
                results[symbol] = df
            await asyncio.sleep(0.1)  # Rate limiting
        
        return results
    
    async def get_current_prices(self) -> Dict[str, float]:
        """Anlık fiyatları çek"""
        session = await self._get_session()
        
        url = f"{self.BASE_URL}/api/v3/ticker/price"
        
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    prices = {item['symbol']: float(item['price']) for item in data}
                    return {s: prices.get(s, 0) for s in self.SYMBOLS}
        except Exception as e:
            print(f"[DataFetcher] Price error: {e}")
        
        return {}


class SignalGenerator:
    """
    Gerçek zamanlı sinyal üretici
    
    - Binance'den veri çeker
    - Strateji ile analiz eder
    - Sinyal üretir
    """
    
    def __init__(self):
        self.data_fetcher = BinanceDataFetcher()
        self.strategy = get_strategy()
        self.active_signals: Dict[str, TradingSignal] = {}
        self.is_running = False
        self.check_interval = 300  # 5 dakika
    
    async def generate_signals(self, interval: str = "1h") -> List[TradingSignal]:
        """Tüm semboller için sinyal üret"""
        signals = []
        
        print(f"[SignalGen] {len(self.data_fetcher.SYMBOLS)} coin analiz ediliyor...")
        
        data = await self.data_fetcher.fetch_all_symbols(interval)
        
        for symbol, df in data.items():
            try:
                signal = self.strategy.analyze(df)
                
                if signal and signal.strength in [SignalStrength.STRONG, SignalStrength.MODERATE]:
                    # Aynı sembol için zaten sinyal var mı kontrol
                    if symbol not in self.active_signals:
                        self.active_signals[symbol] = signal
                        signals.append(signal)
                        print(f"[SINYAL] {signal.signal_type.value} {symbol} - Güven: {signal.confidence}%")
            except Exception as e:
                print(f"[SignalGen] {symbol} analiz hatası: {e}")
        
        return signals
    
    async def run_loop(self):
        """Sürekli sinyal tarama döngüsü"""
        self.is_running = True
        print("[SignalGen] Sinyal tarama başladı")
        
        while self.is_running:
            try:
                signals = await self.generate_signals()
                
                if signals:
                    print(f"[SignalGen] {len(signals)} yeni sinyal!")
                else:
                    print("[SignalGen] Sinyal yok, bekleniyor...")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                print(f"[SignalGen] Döngü hatası: {e}")
                await asyncio.sleep(60)
        
        await self.data_fetcher.close()
    
    def stop(self):
        """Taramayı durdur"""
        self.is_running = False
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Aktif sinyalleri getir"""
        return list(self.active_signals.values())
    
    def clear_signal(self, symbol: str):
        """Sinyal temizle (pozisyon kapatıldığında)"""
        if symbol in self.active_signals:
            del self.active_signals[symbol]


# Singleton
_generator_instance = None

def get_signal_generator() -> SignalGenerator:
    """Signal generator singleton"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SignalGenerator()
    return _generator_instance


# Test
if __name__ == "__main__":
    async def test():
        gen = get_signal_generator()
        
        print("\n=== VERİ ÇEKME TESTİ ===")
        df = await gen.data_fetcher.fetch_klines("BTCUSDT", "1h", 250)
        print(f"BTC Data: {len(df)} mum")
        print(df.tail(3))
        
        print("\n=== SİNYAL ÜRETİM TESTİ ===")
        signals = await gen.generate_signals()
        
        if signals:
            for s in signals:
                print(f"\n{s.signal_type.value} {s.symbol}")
                print(f"  Entry: {s.entry_price:.2f}")
                print(f"  SL: {s.stop_loss:.2f}")
                print(f"  TP: {s.take_profit:.2f}")
                print(f"  Güven: {s.confidence}%")
                print(f"  Sebep: {s.reason}")
        else:
            print("Sinyal bulunamadı")
        
        await gen.data_fetcher.close()
    
    asyncio.run(test())
