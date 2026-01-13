# Smart Watchlist - Akıllı İzleme Listesi
"""
Coinleri izler ve optimal giriş noktasını bekler.

Durumlar:
- SCANNING: Taramada
- WATCHING: İzleniyor (henüz uygun değil)
- READY: Neredeyse hazır
- HOT: GİRİŞ ZAMANI!
- MISSED: Fırsat kaçtı
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class WatchStatus(Enum):
    SCANNING = "⚪"
    WATCHING = "👁️"
    READY = "🟡"
    HOT = "🔥"
    MISSED = "💨"
    ENTERED = "✅"


@dataclass
class WatchEntry:
    """İzleme listesi girişi"""
    symbol: str
    direction: str  # LONG / SHORT
    added_at: datetime
    current_price: float
    target_rsi: float  # Hedef RSI
    current_rsi: float
    target_volume: float  # Hedef volume ratio
    current_volume: float
    score: float
    market_score: float = 50.0
    status: WatchStatus = WatchStatus.WATCHING
    
    # Entry planning
    ideal_entry: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    
    # Tracking
    peak_readiness: float = 0  # En yüksek hazırlık skoru
    checks: int = 0  # Kaç kez kontrol edildi
    
    def get_readiness(self) -> float:
        """
        Giriş hazırlığı (0-100)
        RSI ve Volume hedefe ne kadar yakın?
        """
        rsi_ready = 0
        vol_ready = 0
        
        if self.direction == "LONG":
            # RSI düşük olmalı
            if self.current_rsi <= self.target_rsi:
                rsi_ready = 100
            else:
                rsi_ready = max(0, 100 - (self.current_rsi - self.target_rsi) * 2)
        else:
            # RSI yüksek olmalı
            if self.current_rsi >= self.target_rsi:
                rsi_ready = 100
            else:
                rsi_ready = max(0, 100 - (self.target_rsi - self.current_rsi) * 2)
        
        # Volume kontrolü
        if self.current_volume >= self.target_volume:
            vol_ready = 100
        else:
            vol_ready = min(100, (self.current_volume / self.target_volume) * 100)
        
        # Market Score kontrolü
        market_ready = self.market_score if self.direction == "LONG" else (100 - self.market_score)
        
        return (rsi_ready * 0.4 + vol_ready * 0.3 + market_ready * 0.3)
    
    def update_status(self):
        """Durumu güncelle"""
        readiness = self.get_readiness()
        self.peak_readiness = max(self.peak_readiness, readiness)
        
        if readiness >= 85:
            self.status = WatchStatus.HOT
        elif readiness >= 65:
            self.status = WatchStatus.READY
        elif self.checks > 20 and readiness < 40:
            self.status = WatchStatus.MISSED
        else:
            self.status = WatchStatus.WATCHING


class SmartWatchlist:
    """
    Akıllı İzleme Listesi
    
    - Coinleri takip eder
    - Optimal giriş noktasını bekler
    - HOT olunca sinyal verir
    """
    
    def __init__(self, max_items: int = 10):
        self.max_items = max_items
        self.items: Dict[str, WatchEntry] = {}
        self.history: List[Dict] = []  # Geçmiş girişler
    
    def add(self, symbol: str, direction: str, current_price: float,
            current_rsi: float, current_volume: float, score: float, 
            market_score: float = 50.0) -> WatchEntry:
        """İzleme listesine ekle"""
        
        # Zaten varsa ve yön aynıysa güncelle (Resetlemeyi önle)
        if symbol in self.items and self.items[symbol].direction == direction:
            return self.update(symbol, current_price, current_rsi, current_volume, score, market_score)

        # Hedefleri hesapla
        if direction == "LONG":
            target_rsi = 30  # RSI 30'un altına düşmesini bekle
            ideal_entry = current_price * 0.995  # %0.5 aşağıda
            stop_loss = ideal_entry * 0.99  # %1 aşağıda
            take_profit = ideal_entry * 1.02  # %2 yukarıda
        else:
            target_rsi = 70  # RSI 70'in üstüne çıkmasını bekle
            ideal_entry = current_price * 1.005
            stop_loss = ideal_entry * 1.01
            take_profit = ideal_entry * 0.98
        
        entry = WatchEntry(
            symbol=symbol,
            direction=direction,
            added_at=datetime.now(),
            current_price=current_price,
            target_rsi=target_rsi,
            current_rsi=current_rsi,
            target_volume=1.5,  # 1.5x volume hedefi
            current_volume=current_volume,
            score=score,
            market_score=market_score,
            ideal_entry=ideal_entry,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Eğer liste dolu ise en düşük skorluyu çıkar
        if len(self.items) >= self.max_items:
            min_score_symbol = min(self.items, key=lambda s: self.items[s].score)
            if score > self.items[min_score_symbol].score:
                del self.items[min_score_symbol]
            else:
                return entry  # Ekleme, skor yetersiz
        
        self.items[symbol] = entry
        print(f"[Watchlist] ➕ {symbol} {direction} eklendi (RSI:{current_rsi:.0f}→{target_rsi})")
        
        return entry
    
    def update(self, symbol: str, current_price: float, 
               current_rsi: float, current_volume: float,
               score: float = 0.0, market_score: float = 50.0) -> Optional[WatchEntry]:
        """Coin verilerini güncelle"""
        if symbol not in self.items:
            return None
        
        entry = self.items[symbol]
        entry.current_price = current_price
        entry.current_rsi = current_rsi
        entry.current_volume = current_volume
        if score > 0: entry.score = score
        entry.market_score = market_score
        entry.checks += 1
        
        old_status = entry.status
        entry.update_status()
        
        if entry.status != old_status:
            print(f"[Watchlist] {symbol}: {old_status.value} → {entry.status.value}")
        
        return entry
    
    def get_hot_items(self) -> List[WatchEntry]:
        """HOT durumundaki coinleri getir"""
        return [e for e in self.items.values() if e.status == WatchStatus.HOT]
    
    def get_ready_items(self) -> List[WatchEntry]:
        """READY durumundaki coinleri getir"""
        return [e for e in self.items.values() if e.status == WatchStatus.READY]
    
    def mark_entered(self, symbol: str):
        """Giriş yapıldı olarak işaretle"""
        if symbol in self.items:
            entry = self.items[symbol]
            entry.status = WatchStatus.ENTERED
            
            # Geçmişe ekle
            self.history.append({
                'symbol': symbol,
                'direction': entry.direction,
                'entry_time': datetime.now().isoformat(),
                'readiness': entry.get_readiness()
            })
            
            # Listeden çıkar
            del self.items[symbol]
    
    def remove_missed(self):
        """MISSED olanları temizle"""
        missed = [s for s, e in self.items.items() if e.status == WatchStatus.MISSED]
        for symbol in missed:
            print(f"[Watchlist] 💨 {symbol} kaçırıldı, siliniyor")
            del self.items[symbol]
    
    def get_display_data(self) -> List[Dict]:
        """UI için veri"""
        result = []
        for symbol, entry in self.items.items():
            readiness = entry.get_readiness()
            result.append({
                'symbol': symbol,
                'direction': entry.direction,
                'status': entry.status.value,
                'status_name': entry.status.name,
                'readiness': readiness,
                'current_rsi': entry.current_rsi,
                'target_rsi': entry.target_rsi,
                'current_volume': entry.current_volume,
                'ideal_entry': entry.ideal_entry,
                'stop_loss': entry.stop_loss,
                'take_profit': entry.take_profit,
                'age_minutes': (datetime.now() - entry.added_at).seconds // 60
            })
        
        # Readiness'a göre sırala
        result.sort(key=lambda x: x['readiness'], reverse=True)
        return result
    
    def __len__(self):
        return len(self.items)


# Singleton
_watchlist_instance = None

def get_watchlist() -> SmartWatchlist:
    """Smart Watchlist singleton"""
    global _watchlist_instance
    if _watchlist_instance is None:
        _watchlist_instance = SmartWatchlist()
    return _watchlist_instance


# Test
if __name__ == "__main__":
    wl = get_watchlist()
    
    # Test coin ekle
    wl.add("BTCUSDT", "LONG", 95000, 42, 1.2, 65)
    wl.add("ETHUSDT", "LONG", 2400, 35, 1.8, 70)
    wl.add("SOLUSDT", "SHORT", 180, 68, 1.3, 60)
    
    print("\n=== WATCHLIST ===")
    for item in wl.get_display_data():
        print(f"{item['status']} {item['symbol']} | {item['direction']} | Hazırlık: {item['readiness']:.0f}%")
        print(f"   RSI: {item['current_rsi']:.0f} → {item['target_rsi']}")
    
    # Güncelle
    print("\n=== GÜNCELLEME ===")
    wl.update("BTCUSDT", 94500, 32, 1.8)  # RSI düştü, volume arttı
    wl.update("ETHUSDT", 2380, 28, 2.1)   # Çok iyi!
    
    print("\n=== HOT ITEMS ===")
    for hot in wl.get_hot_items():
        print(f"🔥 {hot.symbol} - GİRİŞ ZAMANI!")
