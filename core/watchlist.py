# HFT Trading Tools - İzleme Listesi (Watchlist)
"""
Volatil coinleri izleme listesine alıp sürekli takip eden modül.
Doğru an geldiğinde sinyal üretir.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class WatchState(Enum):
    """İzleme durumu"""
    WATCHING = "watching"      # İzleniyor
    READY = "ready"            # Sinyal aşamasında
    SIGNALED = "signaled"      # Sinyal verildi
    EXPIRED = "expired"        # Süre doldu


@dataclass
class WatchedCoin:
    """İzleme listesindeki coin"""
    symbol: str
    added_price: float              # İzlemeye alındığındaki fiyat
    current_price: float = 0.0
    
    # İzlemeye alınma nedeni
    initial_change_1m: float = 0.0  # İlk 1m değişim %
    direction: str = ""             # "up" veya "down"
    
    # Anlık veriler (sürekli güncellenir)
    obi_current: float = 0.0
    obi_history: List[float] = field(default_factory=list)
    price_history: List[float] = field(default_factory=list)
    
    # Durum
    state: WatchState = WatchState.WATCHING
    added_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    # Sinyal kriterleri skor
    signal_score: float = 0.0       # 0-100 arası
    signal_reasons: List[str] = field(default_factory=list)


class Watchlist:
    """
    İzleme Listesi Yöneticisi.
    
    Akış:
    1. Scanner volatil coin bulur
    2. Kriterlere uyarsa izleme listesine eklenir
    3. Sürekli OBI, fiyat verisi toplanır
    4. Koşullar uygun olunca sinyal verilir
    """
    
    def __init__(self):
        self._coins: Dict[str, WatchedCoin] = {}
        
        # Parametreler
        self.add_threshold_percent = 0.9     # İzlemeye almak için min %0.9 1m değişim
        self.max_watch_time_minutes = 10    # Maks. 10 dk izle
        self.max_coins = 10                 # Maks. 10 coin izle
        
        # Sinyal için gerekli koşullar
        self.signal_obi_threshold = 0.20    # OBI > 0.20 (veya < -0.20)
        self.signal_obi_confirm_count = 3   # 3 kez üst üste OBI uyumlu olmalı
        self.signal_min_score = 60          # Min 60 puan
        
        # Callback'ler
        self.on_signal: Optional[Callable] = None
        self.on_add: Optional[Callable] = None
        self.on_remove: Optional[Callable] = None
    
    def should_add(self, symbol: str, change_1m: float, volume_24h: float) -> bool:
        """
        Coin izleme listesine eklenmeli mi?
        
        Args:
            symbol: Sembol
            change_1m: 1 dakikalık değişim %
            volume_24h: 24 saatlik hacim
        """
        # Zaten izleniyor mu?
        if symbol in self._coins:
            return False
        
        # Maks. coin sayısı?
        if len(self._coins) >= self.max_coins:
            return False
        
        # Yeterli volatilite?
        if abs(change_1m) < self.add_threshold_percent:
            return False
        
        return True
    
    def add(self, symbol: str, price: float, change_1m: float):
        """
        Coin'i izleme listesine ekle.
        """
        if symbol in self._coins:
            return
        
        direction = "up" if change_1m > 0 else "down"
        
        coin = WatchedCoin(
            symbol=symbol,
            added_price=price,
            current_price=price,
            initial_change_1m=change_1m,
            direction=direction,
            price_history=[price],
            obi_history=[]
        )
        
        self._coins[symbol] = coin
        print(f"[Watchlist] ➕ {symbol} izlemeye alındı ({change_1m:+.2f}%)")
        
        if self.on_add:
            self.on_add(coin)
    
    def update(self, symbol: str, price: float, obi: float) -> Optional[Dict]:
        """
        İzlenen coin'i güncelle ve sinyal kontrolü yap.
        
        Returns:
            Sinyal varsa sinyal bilgisi, yoksa None
        """
        if symbol not in self._coins:
            return None
        
        coin = self._coins[symbol]
        now = datetime.now()
        
        # Süre kontrolü
        elapsed = (now - coin.added_at).total_seconds() / 60
        if elapsed > self.max_watch_time_minutes:
            coin.state = WatchState.EXPIRED
            self._remove(symbol, reason="süre doldu")
            return None
        
        # Verileri güncelle
        coin.current_price = price
        coin.obi_current = obi
        coin.last_update = now
        
        # History'e ekle (son 20 veri)
        coin.price_history.append(price)
        coin.obi_history.append(obi)
        if len(coin.price_history) > 20:
            coin.price_history.pop(0)
        if len(coin.obi_history) > 20:
            coin.obi_history.pop(0)
        
        # Sinyal skoru hesapla
        score, reasons = self._calculate_signal_score(coin)
        coin.signal_score = score
        coin.signal_reasons = reasons
        
        # Sinyal kontrolü
        if score >= self.signal_min_score and coin.state != WatchState.SIGNALED:
            coin.state = WatchState.SIGNALED
            
            signal_data = {
                "symbol": symbol,
                "direction": coin.direction,
                "price": price,
                "score": score,
                "reasons": reasons,
                "watched_for": f"{elapsed:.1f} dk"
            }
            
            print(f"[Watchlist] 🎯 SİNYAL! {symbol} ({score:.0f} puan)")
            for r in reasons:
                print(f"            → {r}")
            
            if self.on_signal:
                self.on_signal(signal_data)
            
            # Sinyalden sonra listeden çıkar
            self._remove(symbol, reason="sinyal verildi")
            
            return signal_data
        
        return None
    
    def _calculate_signal_score(self, coin: WatchedCoin) -> tuple:
        """
        Sinyal skoru hesapla (0-100).
        
        Kriterler:
        - OBI yönle uyumlu mu? (+30)
        - OBI tutarlı mı? (+20)
        - Fiyat yönde devam etti mi? (+20)
        - İlk hareket güçlü müydü? (+15)
        - Momentum devam ediyor mu? (+15)
        """
        score = 0
        reasons = []
        
        if len(coin.obi_history) < 3:
            return score, reasons
        
        obi = coin.obi_current
        direction = coin.direction
        
        # 1. OBI yönle uyumlu mu? (+30)
        if direction == "up" and obi > self.signal_obi_threshold:
            score += 30
            reasons.append(f"OBI pozitif ({obi:.2f})")
        elif direction == "down" and obi < -self.signal_obi_threshold:
            score += 30
            reasons.append(f"OBI negatif ({obi:.2f})")
        
        # 2. OBI tutarlı mı? Son 3 okuma (+20)
        recent_obi = coin.obi_history[-3:]
        if direction == "up":
            if all(o > 0.1 for o in recent_obi):
                score += 20
                reasons.append("OBI tutarlı (3x pozitif)")
        else:
            if all(o < -0.1 for o in recent_obi):
                score += 20
                reasons.append("OBI tutarlı (3x negatif)")
        
        # 3. Fiyat devam etti mi? (+20)
        if len(coin.price_history) >= 3:
            first_price = coin.price_history[0]
            current_price = coin.current_price
            price_change = ((current_price - first_price) / first_price) * 100
            
            if direction == "up" and price_change > 0.3:
                score += 20
                reasons.append(f"Yükseliş devam ({price_change:+.2f}%)")
            elif direction == "down" and price_change < -0.3:
                score += 20
                reasons.append(f"Düşüş devam ({price_change:+.2f}%)")
        
        # 4. İlk hareket güçlü müydü? (+15)
        if abs(coin.initial_change_1m) >= 1.0:
            score += 15
            reasons.append(f"Güçlü başlangıç ({coin.initial_change_1m:+.1f}%)")
        elif abs(coin.initial_change_1m) >= 0.7:
            score += 10
        
        # 5. Momentum (+15) - OBI artıyor/azalıyor mu?
        if len(coin.obi_history) >= 3:
            obi_trend = coin.obi_history[-1] - coin.obi_history[-3]
            if direction == "up" and obi_trend > 0.05:
                score += 15
                reasons.append("OBI momentum artıyor")
            elif direction == "down" and obi_trend < -0.05:
                score += 15
                reasons.append("OBI momentum artıyor")
        
        return score, reasons
    
    def _remove(self, symbol: str, reason: str = ""):
        """Coin'i listeden çıkar"""
        if symbol in self._coins:
            coin = self._coins.pop(symbol)
            print(f"[Watchlist] ➖ {symbol} çıkarıldı ({reason})")
            if self.on_remove:
                self.on_remove(coin, reason)
    
    def get_all(self) -> List[WatchedCoin]:
        """Tüm izlenen coinler"""
        return list(self._coins.values())
    
    def get_count(self) -> int:
        """İzlenen coin sayısı"""
        return len(self._coins)
    
    def is_watching(self, symbol: str) -> bool:
        """Coin izleniyor mu?"""
        return symbol in self._coins
    
    def get(self, symbol: str) -> Optional[WatchedCoin]:
        """Coin bilgisi getir"""
        return self._coins.get(symbol)
    
    def cleanup_expired(self):
        """Süresi dolan coinleri temizle"""
        now = datetime.now()
        to_remove = []
        
        for symbol, coin in self._coins.items():
            elapsed = (now - coin.added_at).total_seconds() / 60
            if elapsed > self.max_watch_time_minutes:
                to_remove.append(symbol)
        
        for symbol in to_remove:
            self._remove(symbol, "süre doldu")
    
    def clear(self):
        """Tüm listeyi temizle"""
        self._coins.clear()
