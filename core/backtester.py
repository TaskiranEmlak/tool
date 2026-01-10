# HFT Trading Tools - Backtest Modülü
"""
Historical data ile tahmin performansını test eden modül.
20 dakikalık fiyat hareketlerini tahmin etme doğruluğunu ölçer.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import os


@dataclass
class BacktestResult:
    """Tek bir backtest noktasının sonucu"""
    timestamp: datetime
    symbol: str
    entry_price: float
    predicted_direction: str  # "up" veya "down"
    predicted_move_percent: float
    confidence: float
    
    # Gerçekleşen sonuç
    actual_price_20m: float = 0.0
    actual_change_percent: float = 0.0
    actual_direction: str = ""
    
    # Başarı
    direction_correct: bool = False
    pnl_percent: float = 0.0
    
    # Sinyal detayları
    signal_scores: Dict = field(default_factory=dict)


@dataclass
class BacktestMetrics:
    """Backtest özet metrikleri"""
    total_signals: int = 0
    correct_predictions: int = 0
    win_rate: float = 0.0
    
    # PnL
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    # Risk metrikleri
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    
    # Detaylar
    up_signals: int = 0
    down_signals: int = 0
    up_win_rate: float = 0.0
    down_win_rate: float = 0.0
    
    # Optimize ağırlıklar
    optimized_weights: Dict = field(default_factory=dict)


class Backtester:
    """
    Historical data ile tahmin performansı testi.
    
    Kullanım:
        bt = Backtester(days=7)
        results = await bt.run(['BTCUSDT', 'ETHUSDT'])
    """
    
    def __init__(self, days: int = 7):
        self.base_url = "https://fapi.binance.com"
        self.days = days
        
        # Mevcut ağırlıklar (varsayılan)
        self.weights = {
            "volume": 15,
            "momentum": 15,
            "obi": 10,
            "market": 25,
            "multi_tf": 20,
            "btc_lag": 15
        }
        
        # Sonuçlar
        self.results: List[BacktestResult] = []
        self.metrics: Optional[BacktestMetrics] = None
        
        # Cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    async def fetch_historical_klines(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """
        Binance'tan historical OHLCV verisi çek.
        
        Args:
            symbol: BTCUSDT, ETHUSDT vs.
            interval: 5m, 15m, 1h vs.
            
        Returns:
            DataFrame: open, high, low, close, volume, timestamp
        """
        cache_key = f"{symbol}_{interval}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        print(f"[Backtest] {symbol} {self.days} günlük veri çekiliyor...")
        
        all_klines = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=self.days)).timestamp() * 1000)
        
        async with aiohttp.ClientSession() as session:
            current_start = start_time
            
            while current_start < end_time:
                url = f"{self.base_url}/fapi/v1/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "limit": 1000
                }
                
                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            klines = await resp.json()
                            if not klines:
                                break
                            all_klines.extend(klines)
                            current_start = klines[-1][0] + 1
                        else:
                            print(f"[Backtest] API hatası: {resp.status}")
                            break
                except Exception as e:
                    print(f"[Backtest] Veri çekme hatası: {e}")
                    break
                
                # Rate limit
                await asyncio.sleep(0.1)
        
        if not all_klines:
            return pd.DataFrame()
        
        # DataFrame oluştur
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Tip dönüşümleri
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Ekstra özellikler hesapla
        df['returns'] = df['close'].pct_change() * 100
        df['volume_sma'] = df['volume'].rolling(12).mean()  # 1 saat ortalama
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum (son 4 mumun toplam değişimi)
        df['momentum'] = df['returns'].rolling(4).sum()
        
        # 20 dakika sonraki fiyat (4 mum sonrası)
        df['price_20m'] = df['close'].shift(-4)
        df['change_20m'] = ((df['price_20m'] - df['close']) / df['close']) * 100
        
        self._data_cache[cache_key] = df
        print(f"[Backtest] {symbol}: {len(df)} mum yüklendi")
        
        return df
    
    async def fetch_funding_history(self, symbol: str) -> pd.DataFrame:
        """Funding rate geçmişini çek"""
        print(f"[Backtest] {symbol} funding history çekiliyor...")
        
        all_funding = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=self.days)).timestamp() * 1000)
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fapi/v1/fundingRate"
            params = {
                "symbol": symbol,
                "startTime": start_time,
                "endTime": end_time,
                "limit": 1000
            }
            
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        all_funding = await resp.json()
            except Exception as e:
                print(f"[Backtest] Funding hatası: {e}")
        
        if not all_funding:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_funding)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        
        return df
    
    def calculate_signal_scores(self, row: pd.Series, df: pd.DataFrame, idx: int) -> Dict:
        """
        Tek bir zaman noktasında sinyal skorlarını hesapla.
        
        Mevcut predictor.py mantığını simüle eder.
        """
        scores = {}
        
        # 1. Volume Score (0-100)
        volume_ratio = row.get('volume_ratio', 1.0)
        if volume_ratio >= 3.0:
            scores['volume'] = 100
        elif volume_ratio >= 2.0:
            scores['volume'] = 75
        elif volume_ratio >= 1.5:
            scores['volume'] = 50
        elif volume_ratio >= 1.2:
            scores['volume'] = 25
        else:
            scores['volume'] = 0
        
        # 2. Momentum Score (0-100)
        momentum = row.get('momentum', 0)
        if momentum > 0.5:
            scores['momentum'] = min(100, momentum * 50)
        elif momentum < -0.5:
            scores['momentum'] = min(100, abs(momentum) * 50)
        else:
            scores['momentum'] = 0
        
        # 3. OBI Score (simüle - gerçek order book yok)
        # Historical'da OBI yok, momentum'dan tahmin et
        scores['obi'] = scores['momentum'] * 0.7
        
        # 4. Market Score (funding varsa)
        # Simplified - funding yüksekse short baskısı
        scores['market'] = 50  # Nötr varsayım
        
        # 5. Multi-TF Score
        # Son 3 ve 5 mumun yönüne bak
        if idx >= 5:
            last_3 = df.iloc[idx-3:idx]['returns'].sum()
            last_5 = df.iloc[idx-5:idx]['returns'].sum()
            
            if last_3 > 0.2 and last_5 > 0.3:
                scores['multi_tf'] = 100  # Güçlü uptrend
            elif last_3 < -0.2 and last_5 < -0.3:
                scores['multi_tf'] = 100  # Güçlü downtrend
            elif (last_3 > 0) == (last_5 > 0):
                scores['multi_tf'] = 50   # Uyumlu ama zayıf
            else:
                scores['multi_tf'] = 0    # Çakışma
        else:
            scores['multi_tf'] = 0
        
        # 6. BTC Lag Score (sadece altcoinler için)
        scores['btc_lag'] = 0  # Basitlik için
        
        return scores
    
    def calculate_total_score(self, scores: Dict) -> Tuple[float, str, float]:
        """
        Toplam skor ve tahmin yönü hesapla.
        
        Returns:
            (total_score, direction, confidence)
        """
        total = sum(
            scores.get(key, 0) * weight / 100
            for key, weight in self.weights.items()
        )
        
        # Yön belirleme
        momentum = scores.get('momentum', 0)
        multi_tf = scores.get('multi_tf', 0)
        
        # Momentum ve TF'den yön çıkar
        direction_score = 0
        if momentum > 30:
            direction_score += 1
        elif momentum < -30:
            direction_score -= 1
        
        if multi_tf > 50:
            # TF yönüne bak (bu basitleştirilmiş)
            direction_score += 1 if momentum >= 0 else -1
        
        direction = "up" if direction_score >= 0 else "down"
        
        # Confidence
        confidence = min(total / 80, 1.0)
        
        return total, direction, confidence
    
    async def simulate_predictions(self, symbol: str) -> List[BacktestResult]:
        """
        Tüm historical data üzerinde tahmin simülasyonu yap.
        """
        df = await self.fetch_historical_klines(symbol)
        
        if df.empty:
            return []
        
        results = []
        
        # Her 5 dakikada bir sinyal üret (20dk sonrasını tahmin)
        for idx in range(10, len(df) - 4):  # İlk 10 ve son 4 mumu atla
            row = df.iloc[idx]
            
            # 20dk sonraki fiyat yoksa atla
            if pd.isna(row['price_20m']):
                continue
            
            # Sinyal skorları hesapla
            scores = self.calculate_signal_scores(row, df, idx)
            total_score, direction, confidence = self.calculate_total_score(scores)
            
            # Sinyal esigi (daha fazla veri icin dusuk tutuyoruz)
            if total_score < 25:
                continue
            
            # Gerçek sonuç
            actual_change = row['change_20m']
            actual_direction = "up" if actual_change > 0 else "down"
            direction_correct = (direction == actual_direction)
            
            # PnL (yön doğruysa kar, yanlışsa zarar)
            if direction_correct:
                pnl = abs(actual_change)
            else:
                pnl = -abs(actual_change)
            
            result = BacktestResult(
                timestamp=row['timestamp'],
                symbol=symbol,
                entry_price=row['close'],
                predicted_direction=direction,
                predicted_move_percent=total_score / 60,  # Score to % mapping
                confidence=confidence,
                actual_price_20m=row['price_20m'],
                actual_change_percent=actual_change,
                actual_direction=actual_direction,
                direction_correct=direction_correct,
                pnl_percent=pnl,
                signal_scores=scores
            )
            
            results.append(result)
        
        return results
    
    def calculate_metrics(self, results: List[BacktestResult]) -> BacktestMetrics:
        """
        Backtest sonuçlarından metrikler hesapla.
        """
        if not results:
            return BacktestMetrics()
        
        metrics = BacktestMetrics()
        metrics.total_signals = len(results)
        
        # Win/Loss
        correct = [r for r in results if r.direction_correct]
        metrics.correct_predictions = len(correct)
        metrics.win_rate = len(correct) / len(results) * 100
        
        # PnL
        pnls = [r.pnl_percent for r in results]
        metrics.total_pnl = sum(pnls)
        metrics.avg_pnl_per_trade = np.mean(pnls)
        metrics.max_profit = max(pnls) if pnls else 0
        metrics.max_loss = min(pnls) if pnls else 0
        
        # Sharpe Ratio (basitleştirilmiş)
        if len(pnls) > 1 and np.std(pnls) > 0:
            metrics.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))
        
        # Max Drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Profit Factor
        profits = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p < 0))
        metrics.profit_factor = profits / losses if losses > 0 else float('inf')
        
        # Up/Down breakdown
        up_results = [r for r in results if r.predicted_direction == "up"]
        down_results = [r for r in results if r.predicted_direction == "down"]
        
        metrics.up_signals = len(up_results)
        metrics.down_signals = len(down_results)
        
        if up_results:
            metrics.up_win_rate = len([r for r in up_results if r.direction_correct]) / len(up_results) * 100
        if down_results:
            metrics.down_win_rate = len([r for r in down_results if r.direction_correct]) / len(down_results) * 100
        
        return metrics
    
    async def optimize_weights(self, symbols: List[str]) -> Dict[str, float]:
        """
        Grid search ile en iyi ağırlıkları bul.
        """
        from itertools import product
        
        print("[Backtest] Ağırlık optimizasyonu başlıyor...")
        
        # Tüm verileri önce çek
        all_data = {}
        for symbol in symbols:
            df = await self.fetch_historical_klines(symbol)
            if not df.empty:
                all_data[symbol] = df
        
        if not all_data:
            return self.weights
        
        best_weights = self.weights.copy()
        best_win_rate = 0
        
        # Basit grid search (5'er adımlarla)
        weight_options = [5, 10, 15, 20, 25, 30]
        
        # Sadece ana 3 ağırlığı optimize et (hız için)
        for vol_w in weight_options:
            for mom_w in weight_options:
                for market_w in weight_options:
                    # Diğer ağırlıkları oranla
                    remaining = 100 - vol_w - mom_w - market_w
                    if remaining < 15:
                        continue
                    
                    test_weights = {
                        "volume": vol_w,
                        "momentum": mom_w,
                        "obi": 10,
                        "market": market_w,
                        "multi_tf": remaining - 10,
                        "btc_lag": 5
                    }
                    
                    # Bu ağırlıklarla test et
                    self.weights = test_weights
                    
                    all_results = []
                    for symbol, df in all_data.items():
                        results = await self.simulate_predictions(symbol)
                        all_results.extend(results)
                    
                    if all_results:
                        metrics = self.calculate_metrics(all_results)
                        
                        # Win rate + Sharpe kombinasyonu
                        score = metrics.win_rate + metrics.sharpe_ratio * 5
                        
                        if score > best_win_rate:
                            best_win_rate = score
                            best_weights = test_weights.copy()
        
        print(f"[Backtest] Optimize ağırlıklar: {best_weights}")
        print(f"[Backtest] En iyi skor: {best_win_rate:.2f}")
        
        return best_weights
    
    async def run(self, symbols: List[str], optimize: bool = False) -> BacktestMetrics:
        """
        Tam backtest çalıştır.
        
        Args:
            symbols: Test edilecek semboller
            optimize: True ise ağırlık optimizasyonu da yap
            
        Returns:
            BacktestMetrics objesi
        """
        print(f"\n{'='*50}")
        print(f"[Backtest] {self.days} günlük test başlıyor")
        print(f"[Backtest] Semboller: {symbols}")
        print(f"{'='*50}\n")
        
        self.results = []
        
        # Her sembol için backtest
        for symbol in symbols:
            results = await self.simulate_predictions(symbol)
            self.results.extend(results)
            print(f"[Backtest] {symbol}: {len(results)} sinyal")
        
        # Metrikler
        self.metrics = self.calculate_metrics(self.results)
        
        # Ağırlık optimizasyonu (isteğe bağlı)
        if optimize and self.results:
            optimized = await self.optimize_weights(symbols)
            self.metrics.optimized_weights = optimized
        
        # Sonuçları yazdır
        self._print_results()
        
        # Sonuçları kaydet
        self._save_results()
        
        return self.metrics
    
    def _print_results(self):
        """Sonuçları formatlı yazdır"""
        m = self.metrics
        
        print(f"\n{'='*50}")
        print("      BACKTEST SONUÇLARI")
        print(f"{'='*50}")
        print(f"""
[GENEL]
   Toplam Sinyal: {m.total_signals}
   Dogru Tahmin: {m.correct_predictions}
   Win Rate: {m.win_rate:.1f}%

[PNL]
   Toplam PnL: {m.total_pnl:+.2f}%
   Ortalama PnL: {m.avg_pnl_per_trade:+.3f}%
   Max Kar: {m.max_profit:+.2f}%
   Max Zarar: {m.max_loss:+.2f}%

[RISK METRIKLERI]
   Sharpe Ratio: {m.sharpe_ratio:.2f}
   Max Drawdown: {m.max_drawdown:.2f}%
   Profit Factor: {m.profit_factor:.2f}

[YON ANALIZI]
   UP Sinyalleri: {m.up_signals} (Win: {m.up_win_rate:.1f}%)
   DOWN Sinyalleri: {m.down_signals} (Win: {m.down_win_rate:.1f}%)
""")
        
        if m.optimized_weights:
            print(f"\n[OPTIMIZE AGIRLIKLAR]")
            for k, v in m.optimized_weights.items():
                print(f"   {k}: {v}")
        
        print(f"{'='*50}\n")
    
    def _save_results(self):
        """Sonuçları dosyaya kaydet"""
        os.makedirs("data/backtest", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metrikleri kaydet
        metrics_path = f"data/backtest/metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "days": self.days,
                "total_signals": self.metrics.total_signals,
                "win_rate": self.metrics.win_rate,
                "total_pnl": self.metrics.total_pnl,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "max_drawdown": self.metrics.max_drawdown,
                "optimized_weights": self.metrics.optimized_weights
            }, f, indent=2)
        
        # Detaylı sonuçları CSV olarak kaydet
        if self.results:
            df = pd.DataFrame([
                {
                    "timestamp": r.timestamp.isoformat(),
                    "symbol": r.symbol,
                    "entry_price": r.entry_price,
                    "predicted_direction": r.predicted_direction,
                    "actual_direction": r.actual_direction,
                    "actual_change": r.actual_change_percent,
                    "correct": r.direction_correct,
                    "pnl": r.pnl_percent
                }
                for r in self.results
            ])
            csv_path = f"data/backtest/results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"[Backtest] Sonuçlar kaydedildi: {csv_path}")
        
        # Optimize ağırlıkları ayrı kaydet
        if self.metrics.optimized_weights:
            weights_path = "data/backtest/optimized_weights.json"
            with open(weights_path, 'w') as f:
                json.dump(self.metrics.optimized_weights, f, indent=2)
            print(f"[Backtest] Optimize agirliklar: {weights_path}")


async def fetch_top_volatile_coins(count: int = 5) -> List[str]:
    """
    En cok fiyat degisimi yasayan coinleri bul.
    
    Args:
        count: Kac coin dondurmek istiyorsun
        
    Returns:
        Symbol listesi ['BTCUSDT', 'ETHUSDT', ...]
    """
    print(f"[Backtest] En volatil {count} coin araniyor...")
    
    async with aiohttp.ClientSession() as session:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"[Backtest] API hatasi: {resp.status}")
                    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
                
                data = await resp.json()
                
                # USDT paritelerini filtrele ve mutlak degisime gore sirala
                usdt_pairs = [
                    {
                        "symbol": d["symbol"],
                        "change": abs(float(d.get("priceChangePercent", 0))),
                        "volume": float(d.get("quoteVolume", 0))
                    }
                    for d in data
                    if d["symbol"].endswith("USDT") and float(d.get("quoteVolume", 0)) > 10_000_000  # Min $10M hacim
                ]
                
                # Oncelik: Yuksek volatilite + yuksek hacim
                # Volatilite skoru = abs(change) * log(volume)
                import math
                for p in usdt_pairs:
                    p["score"] = p["change"] * math.log10(max(p["volume"], 1))
                
                # Skora gore sirala
                usdt_pairs.sort(key=lambda x: x["score"], reverse=True)
                
                # En iyi coinleri dondur (major coinler her zaman dahil)
                major_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
                volatile_coins = [p["symbol"] for p in usdt_pairs if p["symbol"] not in major_coins][:count]
                
                # Major + Volatil kombinasyonu
                top_coins = major_coins + volatile_coins[:count-len(major_coins)]
                top_coins = top_coins[:count]  # Max count
                
                print(f"[Backtest] En volatil coinler: {top_coins}")
                return top_coins
                
        except Exception as e:
            print(f"[Backtest] Coin arama hatasi: {e}")
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]


async def run_backtest(symbols: List[str] = None, days: int = 3, optimize: bool = False):
    """Hizli backtest calistir"""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]
    
    bt = Backtester(days=days)
    return await bt.run(symbols, optimize=optimize)


async def run_smart_backtest(days: int = 3, coin_count: int = 5, optimize: bool = True, train_ml: bool = True):
    """
    Akilli backtest - En volatil coinleri otomatik bulur ve egitir.
    
    Args:
        days: Kac gunluk veri
        coin_count: Kac coin test edilsin
        optimize: Agirlik optimizasyonu yapilsin mi
        train_ml: LightGBM modeli egitilsin mi
        
    Returns:
        BacktestMetrics
        
    Kullanim:
        import asyncio
        from core.backtester import run_smart_backtest
        asyncio.run(run_smart_backtest(days=3, coin_count=5))
    """
    print(f"\n{'='*60}")
    print(f"     AKILLI BACKTEST")
    print(f"     {days} gun, en volatil {coin_count} coin")
    print(f"{'='*60}\n")
    
    # En volatil coinleri bul
    top_coins = await fetch_top_volatile_coins(coin_count)
    
    # Backtest calistir
    bt = Backtester(days=days)
    metrics = await bt.run(top_coins, optimize=optimize)
    
    # ML egitimi (istege bagli)
    if train_ml and bt.results:
        print("\n[Backtest] LightGBM egitimi basliyor...")
        try:
            from core.ai_predictor import LightGBMPredictor
            
            # Backtest sonuclarindan DataFrame olustur
            training_data = pd.DataFrame([
                {
                    "obi": r.signal_scores.get('obi', 0) / 100,
                    "volume_ratio": r.signal_scores.get('volume', 0) / 50,
                    "momentum_score": r.signal_scores.get('momentum', 0),
                    "funding_rate": 0,  # Historical'da yok
                    "long_percent": 50,
                    "oi_change_5m": 0,
                    "tf_5m": 1 if r.predicted_direction == "up" else -1,
                    "tf_1m": 1 if r.predicted_direction == "up" else -1,
                    "btc_lag": 0,
                    "hour_of_day": r.timestamp.hour if hasattr(r.timestamp, 'hour') else 12,
                    "target_change_percent": r.actual_change_percent
                }
                for r in bt.results
            ])
            
            if len(training_data) >= 50:
                lgb = LightGBMPredictor()
                result = lgb.train(df=training_data)
                
                if result.get("success"):
                    print(f"[Backtest] ML egitimi basarili!")
                    print(f"           Yon dogrulugu: {result.get('direction_accuracy', 0):.1f}%")
                else:
                    print(f"[Backtest] ML egitimi basarisiz: {result.get('error')}")
            else:
                print(f"[Backtest] Yetersiz veri ({len(training_data)} < 50), ML egitilmedi")
                
        except Exception as e:
            print(f"[Backtest] ML egitim hatasi: {e}")
    
    print(f"\n{'='*60}")
    print(f"     TAMAMLANDI!")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    # Akilli backtest calistir
    asyncio.run(run_smart_backtest(days=3, coin_count=5, optimize=True, train_ml=True))

