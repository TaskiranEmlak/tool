# HFT Trading Tools - Ana Uygulama v3 (Tahmin Sistemi)
"""
Tahmine dayalı hareket tespit sistemi.
15-20 dakika içinde %1-1.5 hareket edecek coinleri önceden bulur.

Akış:
1. Scanner: Tüm futures'ı tarar, potansiyel olanları bulur
2. Predictor: Hacim, momentum, OBI, BTC lag analizi yapar
3. Signal: Skor 60+ olunca sinyal verir (hareket olmadan ÖNCE)
"""

import asyncio
import threading
from datetime import datetime
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scanner import VolatilityScanner
from core.predictor import Predictor, CoinPrediction
from core.database import Database
from signals.signal_manager import SignalManager
from indicators.benford import WashTradingFilter
from gui.app import TradingApp


class TradingEngine:
    """
    Tahmin tabanlı trading motoru.
    """
    
    def __init__(self, app: TradingApp):
        self.app = app
        
        # Bilesenler
        self.scanner = VolatilityScanner()
        self.predictor = Predictor()
        self.db = Database()
        self.signal_manager = SignalManager(self.db)
        self.wash_filter = WashTradingFilter()  # Benford filtresi
        
        # Durum
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # İzlenen coinler
        self.watched_symbols: List[str] = []
        self.max_watch = 15
        
        # İstatistikler
        self.stats = {
            "scans": 0,
            "signals": 0,
            "analyzed": 0
        }
        
        # Parametreler
        self.scan_interval = 15      # Her 15 saniyede tarama
        self.analyze_interval = 3    # Her 3 saniyede analiz
        self.signal_threshold = 55   # Sinyal için min skor
    
    def start(self):
        """Motoru başlat"""
        self.running = True
        
        def run_async():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._main_loop())
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
    
    def stop(self):
        """Durdur"""
        self.running = False
        self.watched_symbols.clear()
    
    async def _main_loop(self):
        """Ana döngü"""
        print("\n[Engine] Tahmin sistemi baslatiliyor...")
        self.app.update_status("▶ Başlatılıyor...")
        
        last_scan = 0
        
        while self.running:
            try:
                now = asyncio.get_event_loop().time()
                
                # BTC güncelle (her döngüde)
                await self.predictor.update_btc()
                
                # 1. Tarama zamanı mı?
                if now - last_scan >= self.scan_interval:
                    await self._scan_for_candidates()
                    last_scan = now
                
                # 2. İzlenenleri analiz et
                await self._analyze_watched()
                
                # Kısa bekleme
                await asyncio.sleep(self.analyze_interval)
                
            except Exception as e:
                print(f"[Engine] Hata: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)
        
        print("[Engine] Durduruldu")
    
    async def _scan_for_candidates(self):
        """Potansiyel coinleri bul"""
        self.app.update_status("🔍 Tarama yapılıyor...")
        print(f"\n[Scan #{self.stats['scans']+1}] Tarama basliyor...")
        
        # 1m volatilite taraması
        results = await self.scanner.scan_1m_volatility()
        self.stats["scans"] += 1
        
        if not results:
            print("[Scan] Sonuc bulunamadi")
            return
        
        print(f"[Scan] {len(results)} volatil coin bulundu:")
        for coin in results[:5]:
            print(f"  - {coin.symbol}: {coin.price_change_percent:+.2f}%")
        
        # İzleme listesine ekle (en volatil olanlar + mevcut takiptekiler)
        new_symbols = [c.symbol for c in results[:10]]
        
        # Mevcut takipteki ama yeni listede olmayanları da tut (max 15)
        for sym in self.watched_symbols:
            if sym not in new_symbols and len(new_symbols) < self.max_watch:
                new_symbols.append(sym)
        
        self.watched_symbols = new_symbols[:self.max_watch]
        
        # GUI - Sol panel (volatil coinler)
        table_data = []
        for coin in results[:15]:
            table_data.append((
                coin.symbol,
                coin.price,
                coin.price_change_percent,
                coin.volume_24h
            ))
        
        self.app.root.after(0, lambda: self.app.update_coins_table(table_data))
        self.app.root.after(0, lambda: self.app.update_status(
            f"✓ Tarama #{self.stats['scans']} | {len(self.watched_symbols)} coin takipte"
        ))
    
    async def _analyze_watched(self):
        """İzlenen coinleri analiz et"""
        if not self.watched_symbols:
            return
        
        import aiohttp
        
        predictions = []
        
        async with aiohttp.ClientSession() as session:
            for symbol in self.watched_symbols[:10]:  # İlk 10
                try:
                    # Order book çek
                    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=10"
                    async with session.get(url) as response:
                        data = await response.json()
                        
                        bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
                        asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
                        
                        if not bids or not asks:
                            continue
                        
                        price = (bids[0][0] + asks[0][0]) / 2
                        
                        # Tahmin analizi
                        pred = await self.predictor.analyze(symbol, price, bids, asks)
                        predictions.append(pred)
                        self.stats["analyzed"] += 1
                        
                        # Sinyal kontrolü
                        if pred.total_score >= self.signal_threshold:
                            self._emit_signal(pred)
                
                except Exception as e:
                    print(f"[Analyze] {symbol} hatasi: {e}")
                    continue
        
        # GUI güncelle
        self._update_gui(predictions)
    
    def _emit_signal(self, pred: CoinPrediction):
        """Sinyal yayinla ve veritabanina kaydet"""
        self.stats["signals"] += 1
        
        icon = "YUKARI" if pred.predicted_direction == "up" else "ASAGI"
        dir_text = "LONG" if pred.predicted_direction == "up" else "SHORT"
        
        print(f"\n{'='*50}")
        print(f"TAHMIN SINYALI #{self.stats['signals']}")
        print(f"   {icon} {dir_text} {pred.symbol}")
        print(f"   Fiyat: ${pred.current_price:,.4f}")
        print(f"   Skor: {pred.total_score:.0f}/100")
        print(f"   TF Uyumu: {pred.tf_5m_trend}/{pred.tf_1m_trend}")
        print(f"   Tahmini hareket: {pred.predicted_move_percent:.1f}%")
        print(f"   Guven: {pred.confidence:.0%}")
        print(f"   Gerekceler:")
        for r in pred.reasons:
            print(f"      - {r}")
        print(f"{'='*50}\n")
        
        # Benford wash trading kontrolu
        is_suspicious = self.wash_filter.is_suspicious(pred.symbol)
        if is_suspicious:
            print(f"[UYARI] {pred.symbol} supheli hacim tespit edildi (Wash Trading riski)")
        
        # Veritabanina kaydet
        try:
            entry_price = pred.current_price
            stop_loss = entry_price * (0.99 if pred.predicted_direction == "up" else 1.01)
            take_profit = entry_price * (1.015 if pred.predicted_direction == "up" else 0.985)
            
            self.db.insert_signal(
                symbol=pred.symbol,
                direction=pred.predicted_direction,
                confidence=pred.confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason="; ".join(pred.reasons[:3]) if pred.reasons else "Teknik sinyal",
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"[DB] Kayit hatasi: {e}")
        
        # GUI'ye ekle
        self.app.root.after(0, lambda: self.app.add_signal(
            pred.symbol,
            pred.predicted_direction,
            pred.current_price,
            pred.total_score,
            pred.reasons
        ))
    
    def _update_gui(self, predictions: List[CoinPrediction]):
        """GUI güncelle"""
        if not predictions:
            return
        
        # İzleme listesi tablosu
        watch_data = []
        for pred in predictions:
            # OBI history'den son değeri al
            obi_hist = self.predictor._coin_obi_history.get(pred.symbol, [])
            obi = obi_hist[-1] if obi_hist else 0
            
            watch_data.append((
                pred.symbol,
                obi,
                pred.total_score,
                0  # elapsed (şimdilik)
            ))
        
        # Sırala (skora göre)
        watch_data.sort(key=lambda x: x[2], reverse=True)
        
        # İstatistikler
        avg_score = sum(p.total_score for p in predictions) / len(predictions) if predictions else 0
        top_score = max(p.total_score for p in predictions) if predictions else 0
        
        # En yüksek skora sahip coin bilgisi
        if predictions:
            top = max(predictions, key=lambda p: p.total_score)
            print(f"[Top] {top.symbol}: Skor={top.total_score:.0f}, 5m={top.tf_5m_trend}, 1m={top.tf_1m_trend}")
        
        # GUI güncelle
        self.app.root.after(0, lambda: self.app.update_watchlist(watch_data))
        self.app.root.after(0, lambda: self.app.update_indicators(
            len(predictions), avg_score, top_score
        ))


def main():
    """Ana fonksiyon"""
    print("""
    ============================================================
    
       HFT TRADING TOOLS v4.0 - TAHMIN SISTEMI
       ----------------------------------------
    
       Hedef: 15-20dk icinde %1-1.5 hareket edecek coinleri
              ONCEDEN tespit et
    
       Sinyaller:
       - Hacim Spike (volume 2-3x artis)
       - Momentum (tutarli kucuk hareket)
       - OBI Trend (baski birikiyor)
       - Multi-TF (5m + 1m uyumu)
       - Market Data (Funding, L/S, OI)
       - BTC Lag (gecikmis tepki firsati)
    
       Skor 55+ = Sinyal verilir
    
    ============================================================
    
    COIN DETAY PANELI icin ayrica calistir:
    python gui/coin_detail.py
    
    ============================================================
    """)
    
    # GUI oluştur
    app = TradingApp()
    
    # Engine oluştur
    engine = TradingEngine(app)
    
    # Callback'leri bağla
    app.on_start = engine.start
    app.on_stop = engine.stop
    
    # GUI başlat
    app.run()


if __name__ == "__main__":
    main()
