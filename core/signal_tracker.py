# Signal Tracker - Closed Loop Learning
"""
Sinyal sonuçlarını takip eder ve AI'ya geri bildirir.
Bu modül, sinyallerin Kar/Zarar durumunu tespit ederek
AI modelinin kendini geliştirmesini sağlar.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.database import Database


@dataclass
class SignalOutcome:
    """Sinyal sonucu"""
    signal_id: int
    symbol: str
    direction: str
    entry_price: float
    current_price: float
    pnl_percent: float
    status: str  # "open", "tp_hit", "sl_hit", "timeout"
    duration_minutes: int


class SignalTracker:
    """
    Closed-Loop Learning için sinyal takip sistemi.
    
    - Aktif sinyallerin fiyatını takip eder
    - TP/SL vurulduğunda veritabanını günceller
    - Yeterli veri olduğunda AI'yı yeniden eğitir
    
    Kullanım:
        tracker = SignalTracker()
        await tracker.start_tracking()  # Arka planda çalışır
    """
    
    def __init__(self, 
                 tp_percent: float = 1.5,      # Take Profit %
                 sl_percent: float = 1.0,      # Stop Loss %
                 timeout_minutes: int = 20,    # Sinyal timeout
                 retrain_threshold: int = 50): # Yeniden eğitim eşiği
        
        self.db = Database()
        self.base_url = "https://fapi.binance.com"
        
        self.tp_percent = tp_percent
        self.sl_percent = sl_percent
        self.timeout_minutes = timeout_minutes
        self.retrain_threshold = retrain_threshold
        
        self.running = False
        self._outcomes: List[SignalOutcome] = []
    
    async def get_current_price(self, symbol: str) -> float:
        """Binance'den güncel fiyat çek"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fapi/v1/ticker/price?symbol={symbol}"
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return float(data.get("price", 0))
            except Exception as e:
                print(f"[Tracker] Fiyat hatası {symbol}: {e}")
        return 0
    
    async def check_signal(self, signal: dict) -> Optional[SignalOutcome]:
        """
        Tek bir sinyalin durumunu kontrol et.
        
        Returns:
            SignalOutcome if signal should be closed, None otherwise
        """
        symbol = signal["symbol"]
        entry_price = signal["entry_price"]
        direction = signal["direction"]
        stop_loss = signal.get("stop_loss", entry_price * 0.99)
        take_profit = signal.get("take_profit", entry_price * 1.015)
        timestamp = signal.get("timestamp")
        
        # Güncel fiyat
        current_price = await self.get_current_price(symbol)
        if current_price <= 0:
            return None
        
        # PnL hesapla
        if direction == "up":
            pnl_percent = (current_price - entry_price) / entry_price * 100
        else:
            pnl_percent = (entry_price - current_price) / entry_price * 100
        
        # Süre kontrolü
        duration = 0
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            duration = int((datetime.now() - timestamp).total_seconds() / 60)
        
        # Durum belirleme
        status = "open"
        
        # TP kontrolü
        if direction == "up" and current_price >= take_profit:
            status = "tp_hit"
        elif direction == "down" and current_price <= take_profit:
            status = "tp_hit"
        
        # SL kontrolü
        if direction == "up" and current_price <= stop_loss:
            status = "sl_hit"
        elif direction == "down" and current_price >= stop_loss:
            status = "sl_hit"
        
        # Timeout kontrolü
        if status == "open" and duration >= self.timeout_minutes:
            status = "timeout"
        
        return SignalOutcome(
            signal_id=signal["id"],
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            pnl_percent=pnl_percent,
            status=status,
            duration_minutes=duration
        )
    
    async def track_active_signals(self) -> List[SignalOutcome]:
        """
        Tüm aktif sinyalleri kontrol et ve sonuçlandır.
        
        Returns:
            Kapatılan sinyallerin listesi
        """
        closed_signals = []
        
        try:
            active_signals = self.db.get_active_signals()
            
            for signal in active_signals:
                outcome = await self.check_signal(signal)
                
                if outcome and outcome.status != "open":
                    # Sinyali kapat
                    self.db.close_signal(outcome.signal_id, outcome.pnl_percent)
                    closed_signals.append(outcome)
                    
                    # Log
                    result = "WIN" if outcome.pnl_percent > 0 else "LOSS"
                    print(f"[Tracker] {result}: {outcome.symbol} "
                          f"{outcome.pnl_percent:+.2f}% ({outcome.status})")
            
            self._outcomes.extend(closed_signals)
            
        except Exception as e:
            print(f"[Tracker] Takip hatası: {e}")
        
        return closed_signals
    
    def should_retrain(self) -> bool:
        """Yeniden eğitim gerekiyor mu?"""
        try:
            stats = self.db.get_stats()
            closed_signals = stats.get("closed_signals", 0)
            return closed_signals >= self.retrain_threshold
        except:
            return False
    
    async def trigger_retraining(self) -> Dict:
        """
        Yeterli veri varsa AI modelini yeniden eğit.
        
        Returns:
            Eğitim sonuçları
        """
        print("[Tracker] AI yeniden eğitimi başlıyor...")
        
        try:
            from core.ai_predictor import LightGBMPredictor
            import pandas as pd
            
            # Kapatılmış sinyalleri çek
            # Not: get_closed_signals metodu database.py'ye eklenecek
            conn = self.db.conn
            cursor = conn.execute("""
                SELECT symbol, direction, confidence, entry_price, 
                       stop_loss, take_profit, pnl, timestamp
                FROM signals 
                WHERE status = 'closed' AND pnl IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 500
            """)
            rows = cursor.fetchall()
            
            if len(rows) < self.retrain_threshold:
                return {"success": False, "error": f"Yetersiz veri: {len(rows)}"}
            
            # DataFrame oluştur
            training_data = pd.DataFrame([
                {
                    "obi": 0.5,  # Varsayılan
                    "volume_ratio": 1.5,
                    "momentum_score": 50 if r[1] == "up" else -50,
                    "funding_rate": 0,
                    "long_percent": 55 if r[1] == "up" else 45,
                    "oi_change_5m": 0,
                    "tf_5m": 1 if r[1] == "up" else -1,
                    "tf_1m": 1 if r[1] == "up" else -1,
                    "btc_lag": 0,
                    "hour_of_day": 12,
                    "target_change_percent": r[6]  # pnl
                }
                for r in rows
            ])
            
            # LightGBM eğit
            lgb = LightGBMPredictor()
            result = lgb.train(df=training_data)
            
            if result.get("success"):
                print(f"[Tracker] AI eğitimi tamamlandı! "
                      f"Yön doğruluğu: {result.get('direction_accuracy', 0):.1f}%")
            
            return result
            
        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "trace": traceback.format_exc()}
    
    async def start_tracking(self, interval_seconds: int = 30):
        """
        Arka planda sürekli sinyal takibi başlat.
        
        Args:
            interval_seconds: Kontrol aralığı (saniye)
        """
        self.running = True
        print(f"[Tracker] Sinyal takibi başladı (her {interval_seconds}s)")
        
        retrain_counter = 0
        
        while self.running:
            try:
                # Aktif sinyalleri kontrol et
                closed = await self.track_active_signals()
                
                if closed:
                    print(f"[Tracker] {len(closed)} sinyal kapatıldı")
                
                # Her 10 döngüde bir retrain kontrolü
                retrain_counter += 1
                if retrain_counter >= 10:
                    retrain_counter = 0
                    if self.should_retrain():
                        await self.trigger_retraining()
                
            except Exception as e:
                print(f"[Tracker] Döngü hatası: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def stop_tracking(self):
        """Takibi durdur"""
        self.running = False
        print("[Tracker] Sinyal takibi durduruldu")
    
    def get_performance_summary(self) -> Dict:
        """Performans özeti"""
        if not self._outcomes:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0}
        
        wins = [o for o in self._outcomes if o.pnl_percent > 0]
        losses = [o for o in self._outcomes if o.pnl_percent <= 0]
        
        return {
            "total": len(self._outcomes),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self._outcomes) * 100 if self._outcomes else 0,
            "total_pnl": sum(o.pnl_percent for o in self._outcomes),
            "avg_pnl": sum(o.pnl_percent for o in self._outcomes) / len(self._outcomes)
        }


async def run_tracker():
    """Tek seferlik tracker çalıştır"""
    tracker = SignalTracker()
    closed = await tracker.track_active_signals()
    print(f"Kapatılan: {len(closed)}")
    print(f"Performans: {tracker.get_performance_summary()}")


if __name__ == "__main__":
    asyncio.run(run_tracker())
