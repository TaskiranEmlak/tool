# HFT Trading Tools - Tahmin Motoru (Predictor)
"""
15-20 dakika içinde %1-1.5 hareket edecek coinleri önceden tespit eden modül.

6 Sinyal:
1. Volume Spike - Hacim artışı
2. Momentum - Küçük ama tutarlı hareket
3. OBI Pressure - Order book baskısı birikiyor
4. Market Data - Funding, OI, Long/Short Ratio
5. BTC Lag - BTC'ye gecikmiş tepki
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from core.market_data import MarketDataFetcher


@dataclass
class CoinPrediction:
    """Coin tahmin verisi"""
    symbol: str
    current_price: float
    
    # Sinyaller (her biri 0-100 arası ham skor)
    volume_score: float = 0.0       # Hacim artışı
    momentum_score: float = 0.0     # Momentum gücü
    obi_score: float = 0.0          # OBI baskısı
    market_score: float = 0.0       # Funding + OI + L/S Ratio
    btc_lag_score: float = 0.0      # BTC gecikmesi
    
    # Toplam skor (ağırlıklı)
    total_score: float = 0.0
    
    # Tahmin
    predicted_direction: str = ""   # "up" veya "down"
    predicted_move_percent: float = 0.0
    confidence: float = 0.0         # 0-1
    
    # Detaylar
    reasons: List[str] = field(default_factory=list)
    
    # Market data detay
    funding_rate: float = 0.0
    long_short_ratio: float = 1.0
    oi_change: float = 0.0
    
    # Multi-TF uyumu
    tf_5m_trend: str = ""      # "up", "down", "neutral"
    tf_1m_trend: str = ""      # "up", "down", "neutral"
    tf_alignment: bool = False # 5m ve 1m aynı yönde mi?
    tf_score: float = 0.0      # TF uyum skoru
    
    # Meta
    timestamp: datetime = field(default_factory=datetime.now)


class Predictor:
    """
    Tahmin Motoru.
    
    Birden fazla sinyali birleştirerek gelecek hareketi tahmin eder.
    20 dakika içindeki fiyat hareketini öngörür.
    
    Özellikler:
    - Rule-based + ML hibrit sistem
    - Backtest'ten optimize edilmiş ağırlıklar
    - LightGBM entegrasyonu
    """
    
    def __init__(self, use_ml: bool = True):
        self.base_url = "https://fapi.binance.com"
        
        # Market Data fetcher
        self.market_data = MarketDataFetcher()
        
        # Backtest optimize ağırlıkları yükle (varsa)
        self.weights = self._load_optimized_weights()
        
        # ML predictor (opsiyonel)
        self.use_ml = use_ml
        self.ml_predictor = None
        self.ml_weight = 0.4  # ML'in toplam skora etkisi
        
        if use_ml:
            try:
                from core.ai_predictor import LightGBMPredictor
                self.ml_predictor = LightGBMPredictor()
                if self.ml_predictor.is_trained:
                    print(f"[Predictor] LightGBM aktif (v{self.ml_predictor.model_version})")
                else:
                    print("[Predictor] LightGBM model eğitilmemiş, rule-based kullanılacak")
            except Exception as e:
                print(f"[Predictor] ML yüklenemedi: {e}")
                self.ml_predictor = None
        
        # BTC takibi
        self.btc_price_history: deque = deque(maxlen=30)  # Son 30 fiyat
        self.btc_last_move: float = 0.0
        self.btc_last_move_time: Optional[datetime] = None
        
        # Coin verileri cache
        self._coin_volume_history: Dict[str, deque] = {}  # symbol -> son 12 hacim (5m)
        self._coin_price_history: Dict[str, deque] = {}   # symbol -> son 5 fiyat (1m)
        self._coin_obi_history: Dict[str, deque] = {}     # symbol -> son 5 OBI
        
        # Tahmin sonuçları
        self._predictions: Dict[str, CoinPrediction] = {}
    
    def _load_optimized_weights(self) -> Dict[str, float]:
        """
        Backtest'ten optimize edilmiş ağırlıkları yükle.
        Yoksa varsayılan ağırlıkları kullan.
        """
        import json
        import os
        
        weights_path = "data/backtest/optimized_weights.json"
        
        default_weights = {
            "volume": 15,
            "momentum": 15,
            "obi": 10,
            "market": 25,      # Funding + OI + L/S
            "multi_tf": 20,    # 5m + 1m uyumu
            "btc_lag": 15
        }
        
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    loaded_weights = json.load(f)
                print(f"[Predictor] Optimize ağırlıklar yüklendi: {weights_path}")
                return loaded_weights
            except Exception as e:
                print(f"[Predictor] Ağırlık yükleme hatası: {e}")
        
        print("[Predictor] Varsayılan ağırlıklar kullanılıyor")
        return default_weights
    
    async def analyze(self, symbol: str, current_price: float, 
                      bids: List[Tuple[float, float]], 
                      asks: List[Tuple[float, float]]) -> CoinPrediction:
        """
        Coin için tam analiz yap.
        
        Returns:
            CoinPrediction objesi
        """
        prediction = CoinPrediction(
            symbol=symbol,
            current_price=current_price
        )
        
        # 1. Volume analizi
        vol_score, vol_reason = await self._analyze_volume(symbol)
        prediction.volume_score = vol_score
        if vol_reason:
            prediction.reasons.append(vol_reason)
        
        # 2. Momentum analizi
        mom_score, mom_dir, mom_reason = self._analyze_momentum(symbol, current_price)
        prediction.momentum_score = mom_score
        if mom_dir:
            prediction.predicted_direction = mom_dir
        if mom_reason:
            prediction.reasons.append(mom_reason)
        
        # 3. OBI analizi
        obi_score, obi_reason = self._analyze_obi(symbol, bids, asks)
        prediction.obi_score = obi_score
        if obi_reason:
            prediction.reasons.append(obi_reason)
        
        # 4. Market Data analizi (Funding, OI, L/S Ratio)
        market_score, market_direction, market_reasons = await self._analyze_market_data(symbol)
        prediction.market_score = market_score
        if market_direction and not prediction.predicted_direction:
            prediction.predicted_direction = market_direction
        for reason in market_reasons:
            prediction.reasons.append(reason)
        
        # 5. Multi-TF analizi (5m + 1m) - YENİ
        tf_score, tf_5m, tf_1m, tf_aligned, tf_reason = await self._analyze_multi_tf(symbol)
        prediction.tf_score = tf_score
        prediction.tf_5m_trend = tf_5m
        prediction.tf_1m_trend = tf_1m
        prediction.tf_alignment = tf_aligned
        if tf_reason:
            prediction.reasons.append(tf_reason)
        # TF yönü ile tahmin yönünü eşitle (uyumluysa)
        if tf_aligned and tf_5m:
            prediction.predicted_direction = tf_5m
        
        # 6. BTC lag analizi
        btc_score, btc_reason = self._analyze_btc_lag(symbol)
        prediction.btc_lag_score = btc_score
        if btc_reason:
            prediction.reasons.append(btc_reason)
        
        # 7. Rule-based toplam skor hesapla
        rule_score = (
            prediction.volume_score * self.weights["volume"] / 100 +
            prediction.momentum_score * self.weights["momentum"] / 100 +
            prediction.obi_score * self.weights["obi"] / 100 +
            prediction.market_score * self.weights["market"] / 100 +
            prediction.tf_score * self.weights["multi_tf"] / 100 +
            prediction.btc_lag_score * self.weights["btc_lag"] / 100
        )
        
        # 8. ML Hibrit Tahmin (LightGBM varsa)
        ml_score = 0
        ml_direction = None
        
        if self.ml_predictor and self.ml_predictor.is_trained:
            try:
                # ML için feature'ları hazırla
                ml_features = {
                    'obi': prediction.obi_score / 100,
                    'volume_ratio': prediction.volume_score / 50 if prediction.volume_score > 0 else 1.0,
                    'momentum_score': prediction.momentum_score,
                    'funding_rate': prediction.funding_rate,
                    'long_percent': 50 + (prediction.long_short_ratio - 1) * 20,
                    'oi_change_5m': prediction.oi_change,
                    'tf_5m': 1 if prediction.tf_5m_trend == "up" else (-1 if prediction.tf_5m_trend == "down" else 0),
                    'tf_1m': 1 if prediction.tf_1m_trend == "up" else (-1 if prediction.tf_1m_trend == "down" else 0),
                    'btc_lag': prediction.btc_lag_score / 100,
                    'hour_of_day': datetime.now().hour
                }
                
                ml_prediction = self.ml_predictor.predict(ml_features, symbol)
                
                # ML skoru (predicted_change'i 0-100'e çevir)
                ml_score = 50 + ml_prediction.predicted_change * 10
                ml_score = max(0, min(100, ml_score))
                
                ml_direction = ml_prediction.direction
                
                if ml_prediction.confidence > 0.5:
                    prediction.reasons.append(
                        f"🤖 ML: {ml_prediction.predicted_change:+.2f}% "
                        f"({ml_direction}, güven: {ml_prediction.confidence:.0%})"
                    )
            except Exception as e:
                pass  # ML hatası durumunda rule-based devam et
        
        # 9. Final Skor (Hibrit)
        if self.ml_predictor and self.ml_predictor.is_trained and ml_score > 0:
            # 60% Rule-based + 40% ML
            prediction.total_score = (
                rule_score * (1 - self.ml_weight) + 
                ml_score * self.ml_weight
            )
            
            # Yön uyumu bonus/penaltı
            if ml_direction and prediction.predicted_direction:
                if ml_direction == prediction.predicted_direction:
                    prediction.total_score *= 1.1  # +10% bonus
                else:
                    prediction.total_score *= 0.9  # -10% penaltı
        else:
            prediction.total_score = rule_score
        
        # Confidence ve tahmini hareket (20 dakika için kalibre)
        prediction.confidence = min(prediction.total_score / 80, 1.0)
        
        if prediction.total_score >= 60:
            prediction.predicted_move_percent = 1.0 + (prediction.total_score - 60) / 40
        elif prediction.total_score >= 40:
            prediction.predicted_move_percent = 0.5 + (prediction.total_score - 40) / 40
        else:
            prediction.predicted_move_percent = prediction.total_score / 80
        
        # Default yön (eğer belirlenmemişse)
        if not prediction.predicted_direction:
            prediction.predicted_direction = "up"  # Default
        
        # Cache'e kaydet
        self._predictions[symbol] = prediction
        
        return prediction
    
    async def _analyze_multi_tf(self, symbol: str) -> Tuple[float, str, str, bool, str]:
        """
        Multi-Timeframe analizi: 5m trend + 1m giriş.
        
        Returns:
            (skor 0-100, 5m_trend, 1m_trend, uyumlu_mu, açıklama)
        """
        try:
            async with aiohttp.ClientSession() as session:
                # 5m mumları çek (son 3 = 15dk)
                url_5m = f"{self.base_url}/fapi/v1/klines"
                params_5m = {"symbol": symbol, "interval": "5m", "limit": 4}
                
                async with session.get(url_5m, params=params_5m) as resp:
                    klines_5m = await resp.json()
                
                # 1m mumları çek (son 5 = 5dk)
                params_1m = {"symbol": symbol, "interval": "1m", "limit": 6}
                
                async with session.get(url_5m, params=params_1m) as resp:
                    klines_1m = await resp.json()
                
                if len(klines_5m) < 4 or len(klines_1m) < 6:
                    return 0, "", "", False, ""
                
                # 5m trend analizi
                tf_5m_trend = self._calc_trend(klines_5m)
                
                # 1m trend analizi
                tf_1m_trend = self._calc_trend(klines_1m)
                
                # Uyum kontrolü
                aligned = (tf_5m_trend == tf_1m_trend and tf_5m_trend != "neutral")
                
                # Skor hesapla
                score = 0
                reason = ""
                
                if aligned:
                    # Her iki TF aynı yön
                    score = 100
                    direction = "YUKARI" if tf_5m_trend == "up" else "ASAGI"
                    reason = f"TF Uyumu: 5m+1m {direction}"
                elif tf_5m_trend != "neutral" and tf_1m_trend == "neutral":
                    # 5m net, 1m beklemede
                    score = 50
                    reason = f"5m {tf_5m_trend}, 1m beklemede"
                elif tf_5m_trend == "neutral" and tf_1m_trend != "neutral":
                    # 1m hareket var ama 5m teyit yok
                    score = 25
                    reason = f"1m {tf_1m_trend}, 5m teyit yok"
                else:
                    # Çakışma veya nötr
                    score = 0
                    if tf_5m_trend and tf_1m_trend and tf_5m_trend != tf_1m_trend:
                        reason = f"TF cakismasi: 5m={tf_5m_trend} vs 1m={tf_1m_trend}"
                
                return score, tf_5m_trend, tf_1m_trend, aligned, reason
                
        except Exception as e:
            return 0, "", "", False, ""
    
    def _calc_trend(self, klines: list) -> str:
        """Mum listesinden trend hesapla"""
        if len(klines) < 3:
            return "neutral"
        
        # Son 3 mum
        closes = [float(k[4]) for k in klines[-3:]]
        opens = [float(k[1]) for k in klines[-3:]]
        
        # Değişim
        total_change = (closes[-1] - opens[0]) / opens[0] * 100
        
        # Her mumun yönü
        up_candles = sum(1 for i in range(len(closes)) if closes[i] > opens[i])
        down_candles = sum(1 for i in range(len(closes)) if closes[i] < opens[i])
        
        # Trend belirleme
        if total_change > 0.15 and up_candles >= 2:
            return "up"
        elif total_change < -0.15 and down_candles >= 2:
            return "down"
        else:
            return "neutral"
    
    async def _analyze_market_data(self, symbol: str) -> Tuple[float, str, List[str]]:
        """
        Market data analizi: Funding Rate, OI, Long/Short Ratio.
        
        Returns:
            (skor 0-100, yön "up"/"down", gerekçeler listesi)
        """
        try:
            metrics = await self.market_data.get_all_metrics(symbol)
            
            score = 0
            direction = ""
            reasons = []
            
            # 1. Funding Rate (-30 ile +30 arası)
            if metrics.funding_rate > 0.0005:  # Çok yüksek funding
                score -= 30
                direction = "down"
                reasons.append(f"Funding yuksek ({metrics.funding_rate*100:.3f}%) - Short baskisi")
            elif metrics.funding_rate > 0.0003:  # Yüksek
                score -= 15
                reasons.append(f"Funding pozitif ({metrics.funding_rate*100:.3f}%)")
            elif metrics.funding_rate < -0.0003:  # Negatif funding
                score += 25
                direction = "up"
                reasons.append(f"Funding negatif ({metrics.funding_rate*100:.3f}%) - Long firsati")
            elif metrics.funding_rate < -0.0001:
                score += 10
            
            # 2. Long/Short Ratio (-35 ile +35 arası)
            if metrics.long_percent > 70:  # Aşırı long
                score -= 35
                if not direction:
                    direction = "down"
                reasons.append(f"Asiri Long ({metrics.long_percent:.0f}%) - Short squeeze riski")
            elif metrics.long_percent > 60:
                score -= 15
                reasons.append(f"Long baskisi ({metrics.long_percent:.0f}%)")
            elif metrics.long_percent < 30:  # Aşırı short
                score += 35
                if not direction:
                    direction = "up"
                reasons.append(f"Asiri Short ({metrics.short_percent:.0f}%) - Long squeeze firsati")
            elif metrics.long_percent < 40:
                score += 15
                reasons.append(f"Short baskisi ({metrics.short_percent:.0f}%)")
            
            # 3. OI Değişimi (-20 ile +20 arası)
            if metrics.oi_change_5m > 3:  # OI hızla artıyor
                score += 20
                reasons.append(f"OI artiyor (+{metrics.oi_change_5m:.1f}%) - Yeni pozisyonlar")
            elif metrics.oi_change_5m > 1:
                score += 10
            elif metrics.oi_change_5m < -3:  # OI hızla düşüyor
                score -= 15
                reasons.append(f"OI dusuyor ({metrics.oi_change_5m:.1f}%) - Pozisyonlar kapaniyor")
            
            # Skoru 0-100 aralığına normalize et
            # Raw score: -85 ile +85 arası
            normalized_score = max(0, min(100, (score + 85) * 100 / 170))
            
            return normalized_score, direction, reasons
            
        except Exception as e:
            return 0, "", []
    
    async def _analyze_volume(self, symbol: str) -> Tuple[float, str]:
        """
        Hacim artışı analizi.
        Son 5dk hacim vs 1 saat ortalama.
        
        Returns:
            (skor 0-100, açıklama)
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Son 12 tane 5dk mum (1 saat)
                url = f"{self.base_url}/fapi/v1/klines"
                params = {"symbol": symbol, "interval": "5m", "limit": 12}
                
                async with session.get(url, params=params) as response:
                    klines = await response.json()
                    
                    if not klines or len(klines) < 12:
                        return 0, ""
                    
                    # Hacimler
                    volumes = [float(k[5]) for k in klines]  # Volume
                    
                    # Son mum hacmi vs ortalama
                    avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
                    last_volume = volumes[-1]
                    
                    if avg_volume == 0:
                        return 0, ""
                    
                    volume_ratio = last_volume / avg_volume
                    
                    # Cache'e kaydet
                    if symbol not in self._coin_volume_history:
                        self._coin_volume_history[symbol] = deque(maxlen=12)
                    self._coin_volume_history[symbol].append(last_volume)
                    
                    # Skor hesapla
                    if volume_ratio >= 3.0:
                        return 100, f"🔥 Hacim {volume_ratio:.1f}x (çok yüksek)"
                    elif volume_ratio >= 2.0:
                        return 75, f"📈 Hacim {volume_ratio:.1f}x (yüksek)"
                    elif volume_ratio >= 1.5:
                        return 50, f"📊 Hacim {volume_ratio:.1f}x (artış)"
                    elif volume_ratio >= 1.2:
                        return 25, f"Hacim {volume_ratio:.1f}x"
                    else:
                        return 0, ""
                        
        except Exception as e:
            return 0, ""
    
    def _analyze_momentum(self, symbol: str, current_price: float) -> Tuple[float, str, str]:
        """
        Momentum analizi.
        Son 5 fiyat değişimine bakarak tutarlı yön var mı?
        
        Returns:
            (skor 0-100, yön "up"/"down", açıklama)
        """
        # Fiyat history'e ekle
        if symbol not in self._coin_price_history:
            self._coin_price_history[symbol] = deque(maxlen=10)
        
        self._coin_price_history[symbol].append(current_price)
        
        prices = list(self._coin_price_history[symbol])
        if len(prices) < 5:
            return 0, "", ""
        
        # Son 5 fiyat
        recent = prices[-5:]
        
        # Değişimler
        changes = []
        for i in range(1, len(recent)):
            change = ((recent[i] - recent[i-1]) / recent[i-1]) * 100
            changes.append(change)
        
        # Tutarlılık kontrolü
        up_count = sum(1 for c in changes if c > 0.05)
        down_count = sum(1 for c in changes if c < -0.05)
        total_change = sum(changes)
        
        # Tutarlı yükseliş
        if up_count >= 3 and total_change > 0.2:
            score = min(up_count * 20 + abs(total_change) * 20, 100)
            return score, "up", f"📈 Tutarlı yükseliş ({up_count}/4 mum)"
        
        # Tutarlı düşüş
        elif down_count >= 3 and total_change < -0.2:
            score = min(down_count * 20 + abs(total_change) * 20, 100)
            return score, "down", f"📉 Tutarlı düşüş ({down_count}/4 mum)"
        
        return 0, "", ""
    
    def _analyze_obi(self, symbol: str, bids: List[Tuple[float, float]], 
                     asks: List[Tuple[float, float]]) -> Tuple[float, str]:
        """
        OBI baskı analizi.
        OBI'nin değişim HIZI önemli - birikiyor mu?
        
        Returns:
            (skor 0-100, açıklama)
        """
        if not bids or not asks:
            return 0, ""
        
        # OBI hesapla
        bid_vol = sum(b[1] for b in bids[:5])
        ask_vol = sum(a[1] for a in asks[:5])
        total = bid_vol + ask_vol
        
        if total == 0:
            return 0, ""
        
        obi = (bid_vol - ask_vol) / total
        
        # History'e ekle
        if symbol not in self._coin_obi_history:
            self._coin_obi_history[symbol] = deque(maxlen=10)
        
        self._coin_obi_history[symbol].append(obi)
        
        obi_history = list(self._coin_obi_history[symbol])
        
        # OBI trendi (son 5 okuma)
        if len(obi_history) >= 5:
            recent_obi = obi_history[-5:]
            obi_trend = recent_obi[-1] - recent_obi[0]
            
            # Hızlı artış
            if obi_trend > 0.15 and obi > 0.2:
                return 80, f"💪 OBI hızla artıyor ({obi:.2f})"
            elif obi_trend > 0.10 and obi > 0.15:
                return 60, f"📈 OBI artıyor ({obi:.2f})"
            
            # Hızlı düşüş
            elif obi_trend < -0.15 and obi < -0.2:
                return 80, f"💪 OBI hızla düşüyor ({obi:.2f})"
            elif obi_trend < -0.10 and obi < -0.15:
                return 60, f"📉 OBI düşüyor ({obi:.2f})"
            
            # Güçlü OBI (değişim olmasa bile)
            elif abs(obi) > 0.4:
                return 50, f"OBI güçlü ({obi:.2f})"
        
        return 0, ""
    
    def _analyze_btc_lag(self, symbol: str) -> Tuple[float, str]:
        """
        BTC gecikme analizi.
        BTC hareket ettiyse, bu altcoin henüz tepki vermedi mi?
        
        Returns:
            (skor 0-100, açıklama)
        """
        if symbol == "BTCUSDT":
            return 0, ""
        
        if not self.btc_last_move_time:
            return 0, ""
        
        # BTC hareketi ne kadar önce oldu?
        elapsed = (datetime.now() - self.btc_last_move_time).total_seconds()
        
        # 5 dakikadan eski hareketleri ignore et
        if elapsed > 300:
            return 0, ""
        
        # BTC hareket etti ve henüz 2 dakika geçmedi
        if abs(self.btc_last_move) >= 0.3 and elapsed < 120:
            if self.btc_last_move > 0:
                return 70, f"🔗 BTC +{self.btc_last_move:.2f}% (catch-up fırsatı)"
            else:
                return 70, f"🔗 BTC {self.btc_last_move:.2f}% (catch-up fırsatı)"
        
        return 0, ""
    
    async def update_btc(self):
        """BTC fiyatını güncelle ve hareket tespit et"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/fapi/v1/ticker/price?symbol=BTCUSDT"
                async with session.get(url) as response:
                    data = await response.json()
                    price = float(data.get("price", 0))
                    
                    if price > 0:
                        self.btc_price_history.append({
                            "price": price,
                            "time": datetime.now()
                        })
                        
                        # Son 5 fiyatı kontrol et
                        if len(self.btc_price_history) >= 5:
                            old_price = self.btc_price_history[-5]["price"]
                            change = ((price - old_price) / old_price) * 100
                            
                            # %0.3+ hareket varsa kaydet
                            if abs(change) >= 0.3:
                                self.btc_last_move = change
                                self.btc_last_move_time = datetime.now()
                                
        except Exception as e:
            pass
    
    def get_prediction(self, symbol: str) -> Optional[CoinPrediction]:
        """Coin tahminini döndür"""
        return self._predictions.get(symbol)
    
    def get_high_potential_coins(self, min_score: float = 50) -> List[CoinPrediction]:
        """Yüksek potansiyelli coinleri döndür"""
        return [p for p in self._predictions.values() if p.total_score >= min_score]
    
    def clear_history(self, symbol: str):
        """Coin history'sini temizle"""
        if symbol in self._coin_price_history:
            del self._coin_price_history[symbol]
        if symbol in self._coin_obi_history:
            del self._coin_obi_history[symbol]
        if symbol in self._coin_volume_history:
            del self._coin_volume_history[symbol]
