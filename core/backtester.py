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
    Deep Learning (LSTM) + Rule-based ensemble sistem.
    
    Kullanım:
        bt = Backtester(days=7, use_deep_learning=True)
        results = await bt.run(['BTCUSDT', 'ETHUSDT'])
    """
    
    def __init__(self, days: int = 7, use_deep_learning: bool = True):
        self.base_url = "https://fapi.binance.com"
        self.days = days
        self.use_deep_learning = use_deep_learning
        
        # === TRADING PARAMETRELERI (GERCEKCI) ===
        self.COMMISSION = 0.001  # %0.1 (Taker fee + slippage)
        self.TP_PERCENT = 0.006  # %0.6 Take Profit
        self.SL_PERCENT = 0.003  # %0.3 Stop Loss
        self.MIN_PROFIT_THRESHOLD = 0.002  # %0.2 - komisyon ustunde kar
        
        # Backtest agirliklari - GERCEK VERIYE DAYALI
        # NOT: OBI (Order Book Imbalance) backtest'te KULLANILAMAZ
        # cunku gecmis order book verisi yok. RSI'dan OBI turetmek yaniltici.
        self.weights = {
            "volume": 25,      # Hacim analizi - GERCEK
            "momentum": 25,    # Fiyat momentumu - GERCEK
            "rsi": 20,         # RSI indikatoru - GERCEK
            "ema_trend": 20,   # EMA trendi - GERCEK
            "stoch": 10        # Stochastic - GERCEK
            # OBI, market, btc_lag CIKARILDI - backtest'te gercek verisi yok
        }
        
        # Deep Learning predictor
        self.deep_predictor = None
        if use_deep_learning:
            try:
                from core.deep_learning_predictor import LSTMPredictor
                self.deep_predictor = LSTMPredictor()
                if self.deep_predictor.is_trained:
                    print("[Backtest] Deep Learning (LSTM) modeli yüklendi")
                else:
                    print("[Backtest] LSTM modeli henüz eğitilmemiş - kural tabanlı sistem kullanılacak")
            except Exception as e:
                print(f"[Backtest] Deep Learning yüklenemedi: {e}")
        
        # Ensemble ağırlıkları
        self.ensemble_weights = {
            "rule_based": 0.35,  # Teknik indikatörler
            "deep_learning": 0.65  # LSTM tahminleri
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
        
        # Tip donusumleri
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Ekstra ozellikler hesapla
        df['returns'] = df['close'].pct_change() * 100
        df['volume_sma'] = df['volume'].rolling(12).mean()  # 1 saat ortalama
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum (son 4 mumun toplam degisimi)
        df['momentum'] = df['returns'].rolling(4).sum()
        
        # === YENI INDIKATORLER ===
        
        # RSI (14 period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 0.0001)))
        
        # MACD (12, 26, 9)
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic (14 period)
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14).replace(0, 0.0001)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Bollinger Bands (20, 2)
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 0.0001)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100  # Volatilite gostergesi
        
        # EMA Trend (9, 21, 50)
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()  # Trend filtresi icin
        df['ema_trend'] = np.where(df['ema9'] > df['ema21'], 1, -1)
        
        # 20 dakika sonraki fiyat (4 mum sonrasi)
        df['price_20m'] = df['close'].shift(-4)
        df['change_20m'] = ((df['price_20m'] - df['close']) / df['close']) * 100
        
        self._data_cache[cache_key] = df
        print(f"[Backtest] {symbol}: {len(df)} mum yuklendi (RSI/MACD/STOCH aktif)")
        
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
        Tek bir zaman noktasinda sinyal skorlarini hesapla.
        RSI, MACD, Stochastic ile gelistirilmis versiyon.
        """
        scores = {}
        
        # 1. Volume Score (0-100)
        volume_ratio = row.get('volume_ratio', 1.0)
        if pd.isna(volume_ratio):
            volume_ratio = 1.0
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
        
        # 2. RSI Score (0-100) - Oversold/Overbought sinyalleri
        # DAHA SECICI ESIKLER
        rsi = row.get('rsi', 50)
        if pd.isna(rsi):
            rsi = 50
        
        if rsi < 25:
            scores['rsi'] = 100  # Cok oversold - guclu LONG
            scores['rsi_direction'] = 2  # Cift agirlik
        elif rsi < 35:
            scores['rsi'] = 70   # Oversold - LONG firsati
            scores['rsi_direction'] = 1
        elif rsi > 75:
            scores['rsi'] = 100  # Cok overbought - guclu SHORT
            scores['rsi_direction'] = -2
        elif rsi > 65:
            scores['rsi'] = 70   # Overbought - SHORT firsati
            scores['rsi_direction'] = -1
        else:
            scores['rsi'] = 10   # Notr bolge - sinyal yok
            scores['rsi_direction'] = 0
        
        # 3. MACD Score (0-100) - Trend momentum
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        macd_hist = row.get('macd_hist', 0)
        
        if pd.isna(macd) or pd.isna(macd_signal):
            macd, macd_signal, macd_hist = 0, 0, 0
        
        # MACD cross-over kontrolu
        if idx >= 1:
            prev_macd = df.iloc[idx-1].get('macd', 0)
            prev_signal = df.iloc[idx-1].get('macd_signal', 0)
            
            # Bullish cross (MACD signal'i yukari kesiyor)
            if prev_macd <= prev_signal and macd > macd_signal:
                scores['macd'] = 100
                scores['macd_direction'] = 1
            # Bearish cross (MACD signal'i asagi kesiyor)
            elif prev_macd >= prev_signal and macd < macd_signal:
                scores['macd'] = 100
                scores['macd_direction'] = -1
            elif macd > macd_signal:
                scores['macd'] = 60
                scores['macd_direction'] = 1
            elif macd < macd_signal:
                scores['macd'] = 60
                scores['macd_direction'] = -1
            else:
                scores['macd'] = 20
                scores['macd_direction'] = 0
        else:
            scores['macd'] = 20
            scores['macd_direction'] = 0
        
        # 4. Stochastic Score (0-100)
        stoch_k = row.get('stoch_k', 50)
        stoch_d = row.get('stoch_d', 50)
        
        if pd.isna(stoch_k) or pd.isna(stoch_d):
            stoch_k, stoch_d = 50, 50
        
        if stoch_k < 20 and stoch_d < 20:
            scores['stochastic'] = 100
            scores['stoch_direction'] = 1  # Oversold
        elif stoch_k > 80 and stoch_d > 80:
            scores['stochastic'] = 100
            scores['stoch_direction'] = -1  # Overbought
        elif stoch_k < 30:
            scores['stochastic'] = 60
            scores['stoch_direction'] = 1
        elif stoch_k > 70:
            scores['stochastic'] = 60
            scores['stoch_direction'] = -1
        else:
            scores['stochastic'] = 20
            scores['stoch_direction'] = 0
        
        # 5. EMA Trend (0-100)
        ema_trend = row.get('ema_trend', 0)
        momentum = row.get('momentum', 0)
        
        if pd.isna(ema_trend):
            ema_trend = 0
        if pd.isna(momentum):
            momentum = 0
        
        scores['ema_trend'] = 80 if ema_trend != 0 else 30
        scores['ema_direction'] = int(ema_trend) if not pd.isna(ema_trend) else 0
        
        # 6. Momentum (eski mantik)
        if abs(momentum) > 0.5:
            scores['momentum'] = min(100, abs(momentum) * 50)
        else:
            scores['momentum'] = 0
        scores['momentum_direction'] = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
        
        # NOT: OBI (Order Book Imbalance) backtest'te KULLANILMIYOR
        # Gercek OBI icin canli order book verisi gerekli
        # RSI'dan OBI turetmek YANILTICI oldugu icin kaldirildi
        scores['obi'] = 0  # Backtest'te yok
        scores['market'] = 0  # Canli veri gerekli
        scores['multi_tf'] = scores['ema_trend']
        scores['btc_lag'] = 0  # Canli veri gerekli
        
        return scores
    
    def calculate_total_score(self, scores: Dict) -> Tuple[float, str, float]:
        """
        Toplam skor ve tahmin yonu hesapla.
        RSI, MACD, Stochastic ile gelistirilmis yon belirleme.
        
        Returns:
            (total_score, direction, confidence)
        """
        # Toplam skor (eski agirliklar + yeniler)
        base_score = sum(
            scores.get(key, 0) * weight / 100
            for key, weight in self.weights.items()
        )
        
        # Yeni indikator skorlari (bonus)
        rsi_score = scores.get('rsi', 0) * 0.15
        macd_score = scores.get('macd', 0) * 0.15
        stoch_score = scores.get('stochastic', 0) * 0.10
        
        total = base_score + rsi_score + macd_score + stoch_score
        
        # === YON BELIRLEME (SIMETRIK ESIKLER) ===
        # Her indikatorun yonu
        rsi_dir = scores.get('rsi_direction', 0)
        macd_dir = scores.get('macd_direction', 0)
        stoch_dir = scores.get('stoch_direction', 0)
        ema_dir = scores.get('ema_direction', 0)
        mom_dir = scores.get('momentum_direction', 0)
        
        # Agirlikli yon skoru (toplam max: ~7)
        direction_score = (
            rsi_dir * 2.0 +      # RSI onemli (-4 to +4)
            macd_dir * 2.0 +     # MACD onemli (-2 to +2)
            stoch_dir * 1.5 +    # Stochastic (-1.5 to +1.5)
            ema_dir * 1.0 +      # EMA trend (-1 to +1)
            mom_dir * 0.5        # Momentum (-0.5 to +0.5)
        )
        
        # === SIMETRIK YON BELIRLEME ===
        # Ayni esikler - her iki yon icin esit davran
        # En az 2 indikator uyumlu olmali
        
        if direction_score >= 1.5:
            # UP sinyali - pozitif consensus
            direction = "up"
        elif direction_score <= -1.5:
            # DOWN sinyali - negatif consensus
            direction = "down"
        elif direction_score > 0.5:
            # Zayif UP - EMA ve MACD onaylarsa kabul et
            if ema_dir > 0 and (macd_dir > 0 or rsi_dir > 0):
                direction = "up"
            else:
                direction = "neutral"  # Belirsiz
        elif direction_score < -0.5:
            # Zayif DOWN - EMA ve MACD onaylarsa kabul et
            if ema_dir < 0 and (macd_dir < 0 or rsi_dir < 0):
                direction = "down"
            else:
                direction = "neutral"  # Belirsiz
        else:
            direction = "neutral"  # Sinyal yok
        
        # Confidence (yon uyumu ne kadar guclu)
        confidence = min(abs(direction_score) / 7.0, 1.0)
        
        # === TREND FILTRESI (DUZELTILMIS) ===
        # Trend tersine islem icin guclu onay iste
        if direction == "up" and ema_dir < 0:
            # Downtrend'de UP sinyali - RSI oversold ve MACD crossover gerekli
            rsi_val = scores.get('rsi', 50)
            macd_bullish = macd_dir > 0
            stoch_oversold = stoch_dir > 0
            
            if rsi_val >= 70 and (macd_bullish or stoch_oversold):
                # Guclu reversal sinyali var - kabul et ama skoru azalt
                total *= 0.85
            else:
                # Zayif sinyal - neutral yap
                direction = "neutral"
                confidence *= 0.5
                
        elif direction == "down" and ema_dir > 0:
            # Uptrend'de DOWN sinyali - RSI overbought ve MACD crossover gerekli
            rsi_val = scores.get('rsi', 50)
            macd_bearish = macd_dir < 0
            stoch_overbought = stoch_dir < 0
            
            if rsi_val >= 70 and (macd_bearish or stoch_overbought):
                # Guclu reversal sinyali var - kabul et ama skoru azalt
                total *= 0.85
            else:
                # Zayif sinyal - neutral yap
                direction = "neutral"
                confidence *= 0.5

        return total, direction, confidence
    
    async def simulate_predictions(self, symbol: str) -> List[BacktestResult]:
        """
        Tüm historical data üzerinde tahmin simülasyonu yap.
        Deep Learning (LSTM) + Rule-based ensemble kullanır.
        """
        df = await self.fetch_historical_klines(symbol)
        
        if df.empty:
            return []
        
        results = []
        
        # Deep Learning için özellikleri hesapla
        deep_features_df = None
        if self.deep_predictor and self.deep_predictor.is_trained:
            deep_features_df = self.deep_predictor.calculate_features(df.copy())
        
        # Her 5 dakikada bir sinyal üret (20dk sonrasını tahmin)
        for idx in range(60, len(df) - 4):  # LSTM için 60 mum gerekli
            row = df.iloc[idx]
            
            # 20dk sonraki fiyat yoksa atla
            if pd.isna(row['price_20m']):
                continue
            
            # === KURAL TABANLI SKOR ===
            scores = self.calculate_signal_scores(row, df, idx)
            rule_score, rule_direction, rule_confidence = self.calculate_total_score(scores)
            
            # === DEEP LEARNING TAHMİNİ ===
            dl_direction = "neutral"
            dl_confidence = 0.33
            dl_up_prob = 0.33
            dl_down_prob = 0.33
            
            if deep_features_df is not None and self.deep_predictor:
                try:
                    # Son 60 mum ile tahmin
                    df_slice = deep_features_df.iloc[max(0, idx-60):idx+1]
                    if len(df_slice) >= 60:
                        dl_pred = self.deep_predictor.predict(df_slice, symbol)
                        dl_direction = dl_pred.direction
                        dl_confidence = dl_pred.confidence
                        dl_up_prob = dl_pred.probabilities.get('up', 0.33)
                        dl_down_prob = dl_pred.probabilities.get('down', 0.33)
                        dl_neutral_prob = dl_pred.probabilities.get('neutral', 0.34)
                except Exception:
                    pass  # Hata durumunda kural tabanlı kullan
            
            # === ENSEMBLE KARAR (DUZELTILMIS) ===
            if self.deep_predictor and self.deep_predictor.is_trained:
                
                # KRITIK: LSTM NEUTRAL yuksekse sinyal verme
                # Model %60 NEUTRAL verisiyle egitildi, NEUTRAL = belirsizlik
                dl_neutral_prob = dl_pred.probabilities.get('neutral', 0.34) if 'dl_pred' in dir() else 0.34
                if dl_neutral_prob > 0.50:
                    continue  # Belirsiz durum, sinyal yok
                
                
                # Rule-based yon skorlari (normalize)
                rule_up = (rule_score / 100) if rule_direction == "up" else 0
                rule_down = (rule_score / 100) if rule_direction == "down" else 0
                
                # Ensemble skor - DENGELI (bias kaldirildi)
                up_score = (
                    rule_up * self.ensemble_weights["rule_based"] +
                    dl_up_prob * self.ensemble_weights["deep_learning"]
                )
                down_score = (
                    rule_down * self.ensemble_weights["rule_based"] +
                    dl_down_prob * self.ensemble_weights["deep_learning"]
                )
                
                # Fark esigi - yumusatildi
                score_diff = abs(up_score - down_score)
                
                if score_diff < 0.05:  # 0.12 -> 0.05 (Daha fazla sinyal icin)
                    # print(f"[Filter] Score diff too low: {score_diff:.3f}")
                    continue
                
                # YON KARARI - DENGELI
                if up_score > down_score:
                    # UP icin hem LSTM hem rule-based onay gerekli
                    if dl_direction in ["up", "neutral"] and rule_direction == "up":
                        direction = "up"
                        confidence = up_score / (up_score + down_score + 0.0001)
                        total_score = rule_score * 0.5 + dl_confidence * 100 * 0.5
                    else:
                        # print(f"[Filter] UP conflict: DL={dl_direction}, Rule={rule_direction}")
                        continue  # Onay yok, atla
                else:
                    # DOWN icin daha esnek
                    if dl_direction in ["down", "neutral"] or rule_direction == "down":
                        direction = "down"
                        confidence = down_score / (up_score + down_score + 0.0001)
                        total_score = rule_score * 0.5 + dl_confidence * 100 * 0.5
                    else:
                        # print(f"[Filter] DOWN conflict")
                        continue
            else:
                # Deep Learning yoksa sadece kural tabanlı
                direction = rule_direction
                confidence = rule_confidence
                total_score = rule_score
            
            # === TREND FİLTRESİ (EMA50) ===
            ema50 = row.get('ema_50', row['close'])
            trend_up = row['close'] > ema50
            
            # Trend tersi islem acma!
            if direction == "up" and not trend_up:
                # print(f"[Filter] Trend mismatch (UP signal in DOWN trend)")
                continue  # Dusus trendinde LONG acma
            if direction == "down" and trend_up:
                # print(f"[Filter] Trend mismatch (DOWN signal in UP trend)")
                continue  # Yukselis trendinde SHORT acma
            
            # === VOLATILITE FILTRESI ===
            bb_width = row.get('bb_width', 2.0)
            if bb_width < 0.3:  # 0.5 -> 0.3 (Biraz gevsetildi)
                # print(f"[Filter] Low volatility: {bb_width:.2f}")
                continue  # Yatay piyasada islem acma
            
            # === GUCLU FILTRELER ===
            if total_score < 50:  # 70 -> 50 (Formul degistigi icin dusuruldu)
                # print(f"[Filter] High score too low: {total_score:.1f}")
                continue
            
            if direction == "neutral":
                continue
            
            if confidence < 0.45: # 0.55 -> 0.45 (LSTM confidence genelde dusuktur)
                # print(f"[Filter] Low confidence: {confidence:.2f}")
                continue
            
            # === GERCEKCI TP/SL SIMULASYONU ===
            entry_price = row['close']
            
            # Gelecek 4 mumun (20dk) high/low'u
            future_slice = df.iloc[idx+1:idx+5]
            if len(future_slice) < 4:
                continue
            
            future_high = future_slice['high'].max()
            future_low = future_slice['low'].min()
            
            win = False
            pnl = 0.0
            
            if direction == "up":
                tp_price = entry_price * (1 + self.TP_PERCENT)
                sl_price = entry_price * (1 - self.SL_PERCENT)
                
                # Hangisi once oldu - TP mi SL mi?
                # Basit kontrol: low SL'e indiyse stop
                if future_low <= sl_price:
                    win = False
                    pnl = -self.SL_PERCENT - self.COMMISSION
                elif future_high >= tp_price:
                    win = True
                    pnl = self.TP_PERCENT - self.COMMISSION
                else:
                    # Ne TP ne SL - 20dk sonundaki fiyat
                    exit_price = future_slice.iloc[-1]['close']
                    raw_change = (exit_price - entry_price) / entry_price
                    pnl = raw_change - self.COMMISSION
                    win = pnl > self.MIN_PROFIT_THRESHOLD
            else:  # DOWN
                tp_price = entry_price * (1 - self.TP_PERCENT)
                sl_price = entry_price * (1 + self.SL_PERCENT)
                
                if future_high >= sl_price:
                    win = False
                    pnl = -self.SL_PERCENT - self.COMMISSION
                elif future_low <= tp_price:
                    win = True
                    pnl = self.TP_PERCENT - self.COMMISSION
                else:
                    exit_price = future_slice.iloc[-1]['close']
                    raw_change = (entry_price - exit_price) / entry_price
                    pnl = raw_change - self.COMMISSION
                    win = pnl > self.MIN_PROFIT_THRESHOLD
            
            # Actual values for logging
            actual_change = row['change_20m']
            actual_direction = "up" if actual_change > 0 else "down"
            
            # Skora DL bilgisi ekle
            scores['dl_direction'] = dl_direction
            scores['dl_confidence'] = dl_confidence
            scores['dl_up_prob'] = dl_up_prob
            scores['dl_down_prob'] = dl_down_prob
            scores['trend_up'] = trend_up
            scores['bb_width'] = bb_width
            
            result = BacktestResult(
                timestamp=row['timestamp'],
                symbol=symbol,
                entry_price=entry_price,
                predicted_direction=direction,
                predicted_move_percent=total_score / 60,
                confidence=confidence,
                actual_price_20m=row['price_20m'],
                actual_change_percent=pnl * 100,  # Gercek PnL (komisyon dahil)
                actual_direction=actual_direction,
                direction_correct=win,
                pnl_percent=pnl * 100,  # % olarak
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


async def fetch_top_volatile_coins(count: int = 10, include_major: bool = True) -> List[str]:
    """
    En cok fiyat degisimi yasayan coinleri bul.
    Major coinler her zaman dahil (daha guvenilir).
    
    Args:
        count: Kac coin dondurmek istiyorsun
        include_major: BTC/ETH/SOL zorunlu dahil edilsin mi (varsayilan: True)
        
    Returns:
        Symbol listesi ['XYZUSDT', ...]
    """
    print(f"[Backtest] En volatil {count} coin araniyor (major dahil)...")
    
    async with aiohttp.ClientSession() as session:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"[Backtest] API hatasi: {resp.status}")
                    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
                            "ADAUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT"][:count]
                
                data = await resp.json()
                
                # Major coinler - her zaman dahil (daha guvenilir)
                major_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
                # Top tier altcoinler - yuksek likidite
                tier2_coins = ["XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"]
                
                # USDT paritelerini filtrele
                usdt_pairs = []
                for d in data:
                    symbol = d["symbol"]
                    if not symbol.endswith("USDT"):
                        continue
                    
                    change = abs(float(d.get("priceChangePercent", 0)))
                    volume = float(d.get("quoteVolume", 0))
                    
                    is_major = symbol in major_coins
                    is_tier2 = symbol in tier2_coins
                    
                    # Filtreler (major'lar icin daha dusuk, digerleri icin yuksek):
                    if is_major:
                        # Major coinler her zaman al
                        min_vol = 100_000_000  # $100M
                    elif is_tier2:
                        # Tier2 coinler
                        min_vol = 50_000_000   # $50M
                    else:
                        # Diger altcoinler - cok yuksek hacim ve sinirli volatilite
                        min_vol = 100_000_000  # $100M - dusuk hacim = manipulasyon riski
                        if change < 2.0 or change > 15:  # %2-15 arasi volatilite
                            continue
                    
                    if volume < min_vol:
                        continue
                    
                    usdt_pairs.append({
                        "symbol": symbol,
                        "change": change,
                        "volume": volume,
                        "is_major": is_major,
                        "is_tier2": is_tier2
                    })
                
                # Skor hesapla - major ve tier2'ye bonus
                import math
                for p in usdt_pairs:
                    base_score = p["change"] * math.log10(max(p["volume"], 1))
                    if p["is_major"]:
                        base_score *= 2.0  # Major bonus
                    elif p["is_tier2"]:
                        base_score *= 1.5  # Tier2 bonus
                    p["score"] = base_score
                
                # Skora gore sirala
                usdt_pairs.sort(key=lambda x: x["score"], reverse=True)
                
                # Major coinleri her zaman en basa ekle
                top_coins = []
                if include_major:
                    for mc in major_coins:
                        if any(p["symbol"] == mc for p in usdt_pairs):
                            top_coins.append(mc)
                
                # Kalan slotlari en volatil coinlerle doldur
                for p in usdt_pairs:
                    if p["symbol"] not in top_coins and len(top_coins) < count:
                        top_coins.append(p["symbol"])
                
                if len(top_coins) < count:
                    # Yeterli coin bulunamadıysa fallback
                    fallback = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", 
                               "AVAXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT"]
                    for coin in fallback:
                        if coin not in top_coins and len(top_coins) < count:
                            top_coins.append(coin)
                
                print(f"[Backtest] En volatil coinler: {top_coins}")
                return top_coins
                
        except Exception as e:
            print(f"[Backtest] Coin arama hatasi: {e}")
            return ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
                    "LINKUSDT", "MATICUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT"][:count]


async def run_backtest(symbols: List[str] = None, days: int = 3, optimize: bool = False):
    """Hizli backtest calistir"""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]
    
    bt = Backtester(days=days)
    return await bt.run(symbols, optimize=optimize)


async def run_smart_backtest(days: int = 3, coin_count: int = 5, optimize: bool = True, 
                             train_ml: bool = True, use_deep_learning: bool = True,
                             train_lstm: bool = False):
    """
    Akilli backtest - En volatil coinleri otomatik bulur ve egitir.
    LSTM + LightGBM + Rule-based ensemble sistem.
    
    Args:
        days: Kac gunluk veri
        coin_count: Kac coin test edilsin
        optimize: Agirlik optimizasyonu yapilsin mi
        train_ml: LightGBM modeli egitilsin mi
        use_deep_learning: LSTM derin ogrenme kullanilsin mi
        train_lstm: LSTM modeli yeniden egitilsin mi
        
    Returns:
        BacktestMetrics
        
    Kullanim:
        import asyncio
        from core.backtester import run_smart_backtest
        asyncio.run(run_smart_backtest(days=3, coin_count=5))
    """
    print(f"\n{'='*60}")
    print(f"     AKILLI BACKTEST (Deep Learning)")
    print(f"     {days} gun, en volatil {coin_count} coin")
    print(f"{'='*60}\n")
    
    # LSTM egitimi (istege bagli)
    if train_lstm and use_deep_learning:
        print("[Backtest] LSTM modeli egitiliyor...")
        try:
            from core.deep_learning_predictor import train_lstm_model
            await train_lstm_model(days=days+2, symbols=None)
        except Exception as e:
            print(f"[Backtest] LSTM egitim hatasi: {e}")
    
    # En volatil coinleri bul
    top_coins = await fetch_top_volatile_coins(coin_count)
    
    # Backtest calistir
    bt = Backtester(days=days, use_deep_learning=use_deep_learning)
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
                    "rsi": r.signal_scores.get('rsi', 50),
                    "macd_direction": r.signal_scores.get('macd_direction', 0),
                    "stoch_direction": r.signal_scores.get('stoch_direction', 0),
                    "dl_confidence": r.signal_scores.get('dl_confidence', 0.33),
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
    if metrics:
        print(f"     Win Rate: {metrics.win_rate:.1f}%")
        print(f"     PnL: {metrics.total_pnl:+.2f}%")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    # Akilli backtest calistir
    asyncio.run(run_smart_backtest(days=3, coin_count=5, optimize=True, train_ml=True))

