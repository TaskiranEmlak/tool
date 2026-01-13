# Brain Dashboard - Öğrenen Zeka Arayüzü
"""
Heyecan verici, modern trading dashboard.
- Beyin durumu ve öğrenme görselleştirme
- Smart Watchlist (HOT/READY/WATCHING)
- Canlı P&L
- Isı haritası

CustomTkinter ile modern koyu tema.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trading_brain import get_brain, TradingBrain, BrainDecision, TradeResult
from core.smart_watchlist import get_watchlist, SmartWatchlist, WatchStatus
from core.scanner import VolatilityScanner
from core.predictor import Predictor
from tools.helpers import VoiceAlert

# Tema
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class BrainDashboard(ctk.CTk):
    """
    Öğrenen Zeka Dashboard
    Heyecan verici, canlı, akıllı.
    """
    
    def __init__(self):
        super().__init__()
        
        # Pencere
        self.title("🧠 Trading Brain - Öğrenen Zeka")
        self.geometry("1500x900")
        self.minsize(1300, 800)
        
        # Sistemler
        self.brain = get_brain()
        self.watchlist = get_watchlist()
        self.scanner = VolatilityScanner()
        self.predictor = Predictor()
        self.voice = VoiceAlert(enabled=True)
        
        # Durum
        self.running = False
        self.active_signals: Dict[str, Dict] = {}  # Aktif sinyaller
        self.pnl_today = 0.0
        
        # UI oluştur
        self._create_header()
        self._create_main_panels()
        self._create_footer()
        
        # İlk veri
        self._update_brain_status()
    
    def _create_header(self):
        """Üst bar - Beyin durumu"""
        header = ctk.CTkFrame(self, height=80, corner_radius=0, fg_color="#1a1a2e")
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)
        
        # Sol - Logo ve durum
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=20, pady=10)
        
        ctk.CTkLabel(left, text="🧠", font=("Segoe UI", 36)).pack(side="left")
        
        title_frame = ctk.CTkFrame(left, fg_color="transparent")
        title_frame.pack(side="left", padx=10)
        
        ctk.CTkLabel(title_frame, text="Trading Brain", 
                    font=("Segoe UI", 20, "bold")).pack(anchor="w")
        self.brain_status = ctk.CTkLabel(title_frame, text="Başlatılıyor...", 
                                         font=("Segoe UI", 11), text_color="#888")
        self.brain_status.pack(anchor="w")
        
        # Orta - Öğrenme pulse animasyonu
        center = ctk.CTkFrame(header, fg_color="transparent")
        center.pack(side="left", padx=50)
        
        self.learning_indicator = ctk.CTkLabel(center, text="⚡ ÖĞRENME AKTİF", 
                                               font=("Segoe UI", 12, "bold"),
                                               text_color="#00d4ff")
        self.learning_indicator.pack()
        
        self.trades_label = ctk.CTkLabel(center, text="0 işlem analiz edildi", 
                                         font=("Segoe UI", 10), text_color="#666")
        self.trades_label.pack()
        
        # Sağ - P&L ve kontroller
        right = ctk.CTkFrame(header, fg_color="transparent")
        right.pack(side="right", padx=20)
        
        pnl_frame = ctk.CTkFrame(right, fg_color="#0f3460", corner_radius=10)
        pnl_frame.pack(side="left", padx=10)
        
        ctk.CTkLabel(pnl_frame, text="Bugün", font=("Segoe UI", 10), 
                    text_color="#888").pack(padx=15, pady=(5,0))
        self.pnl_label = ctk.CTkLabel(pnl_frame, text="$0.00", 
                                      font=("Consolas", 20, "bold"),
                                      text_color="#00ff88")
        self.pnl_label.pack(padx=15, pady=(0,5))
        
        # Ses toggle
        self.voice_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(right, text="🔊", variable=self.voice_var,
                     command=self._toggle_voice, width=50).pack(side="left", padx=10)
    
    def _create_main_panels(self):
        """Ana paneller"""
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=10, pady=10)
        
        # === SOL: HOT LIST ===
        left = ctk.CTkFrame(main, width=350, fg_color="#16213e", corner_radius=15)
        left.pack(side="left", fill="y", padx=(0,10))
        left.pack_propagate(False)
        
        ctk.CTkLabel(left, text="🔥 WATCHLIST", 
                    font=("Segoe UI", 16, "bold")).pack(pady=15)
        
        # Watchlist scroll
        self.watchlist_frame = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.watchlist_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder
        self.watchlist_placeholder = ctk.CTkLabel(self.watchlist_frame, 
                                                  text="Tarama başlatılıyor...",
                                                  text_color="#666")
        self.watchlist_placeholder.pack(pady=50)
        
        # === ORTA: AKTİF SİNYALLER ===
        center = ctk.CTkFrame(main, fg_color="#16213e", corner_radius=15)
        center.pack(side="left", fill="both", expand=True, padx=10)
        
        # Sinyal başlık
        signal_header = ctk.CTkFrame(center, fg_color="transparent")
        signal_header.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(signal_header, text="🎯 AKTİF SİNYALLER", 
                    font=("Segoe UI", 16, "bold")).pack(side="left")
        
        self.signal_count_label = ctk.CTkLabel(signal_header, text="0 sinyal", 
                                               text_color="#888")
        self.signal_count_label.pack(side="right")
        
        # Sinyal kartları
        self.signals_frame = ctk.CTkScrollableFrame(center, fg_color="transparent")
        self.signals_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # "Beyin neden" açıklama alanı
        explain_frame = ctk.CTkFrame(center, fg_color="#0f3460", corner_radius=10)
        explain_frame.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(explain_frame, text="🧠 BEYİN NEDEN BU SİNYALİ VERDİ:", 
                    font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(10,5))
        self.explain_text = ctk.CTkLabel(explain_frame, 
                                         text="Sinyal bekleniyor...",
                                         font=("Segoe UI", 10),
                                         text_color="#aaa",
                                         wraplength=500)
        self.explain_text.pack(anchor="w", padx=10, pady=(0,10))
        
        # === SAĞ: HEATMAP & STATS ===
        right = ctk.CTkFrame(main, width=350, fg_color="#16213e", corner_radius=15)
        right.pack(side="right", fill="y", padx=(10,0))
        right.pack_propagate(False)
        
        ctk.CTkLabel(right, text="📊 BEYİN ANALİZ", 
                    font=("Segoe UI", 16, "bold")).pack(pady=15)
        
        # Heatmap
        heatmap_frame = ctk.CTkFrame(right, fg_color="#0f3460", corner_radius=10)
        heatmap_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(heatmap_frame, text="ISI HARİTASI (RSI x Volume)", 
                    font=("Segoe UI", 10, "bold")).pack(pady=(10,5))
        self.heatmap_label = ctk.CTkLabel(heatmap_frame, 
                                          text="Veri toplanıyor...",
                                          font=("Consolas", 9),
                                          justify="left")
        self.heatmap_label.pack(padx=10, pady=(0,10))
        
        # Win rate
        stats_frame = ctk.CTkFrame(right, fg_color="#0f3460", corner_radius=10)
        stats_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(stats_frame, text="PERFORMANS", 
                    font=("Segoe UI", 10, "bold")).pack(pady=(10,5))
        
        self.winrate_label = ctk.CTkLabel(stats_frame, text="Win Rate: -%", 
                                          font=("Consolas", 14, "bold"),
                                          text_color="#00ff88")
        self.winrate_label.pack()
        
        self.patterns_label = ctk.CTkLabel(stats_frame, text="0 pattern öğrenildi", 
                                           text_color="#888")
        self.patterns_label.pack(pady=(0,10))
        
        # Son öğrenmeler
        learn_frame = ctk.CTkFrame(right, fg_color="#0f3460", corner_radius=10)
        learn_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        ctk.CTkLabel(learn_frame, text="SON ÖĞRENMELER", 
                    font=("Segoe UI", 10, "bold")).pack(pady=(10,5))
        
        self.learn_text = ctk.CTkTextbox(learn_frame, font=("Consolas", 9), 
                                         height=200, fg_color="#16213e")
        self.learn_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_footer(self):
        """Alt bar"""
        footer = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="#1a1a2e")
        footer.pack(fill="x")
        footer.pack_propagate(False)
        
        # Kontroller
        self.start_btn = ctk.CTkButton(footer, text="▶ BAŞLAT", width=120,
                                       fg_color="#00d26a", hover_color="#00b359",
                                       command=self._start)
        self.start_btn.pack(side="left", padx=20, pady=15)
        
        self.stop_btn = ctk.CTkButton(footer, text="⏹ DURDUR", width=120,
                                      fg_color="#ff4757", hover_color="#ff3344",
                                      command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        # Durum
        self.status_label = ctk.CTkLabel(footer, text="⏸ Hazır", 
                                         font=("Segoe UI", 11))
        self.status_label.pack(side="left", padx=30)
        
        # Sağ - versiyon
        ctk.CTkLabel(footer, text="Trading Brain v3.0 | Öğrenen Zeka Sistemi",
                    font=("Segoe UI", 10), text_color="#666").pack(side="right", padx=20)
    
    # ========== CALLBACKS ==========
    
    def _toggle_voice(self):
        self.voice.enabled = self.voice_var.get()
    
    def _update_brain_status(self):
        """Beyin durumunu güncelle"""
        status = self.brain.get_status()
        
        self.brain_status.configure(text=f"Win Rate: {status['win_rate']:.1f}% | {status['total_pnl']:+.2f}%")
        self.trades_label.configure(text=f"{status['total_trades']} işlem analiz edildi")
        self.winrate_label.configure(text=f"Win Rate: {status['win_rate']:.1f}%")
        self.patterns_label.configure(text=f"{status['patterns_learned']} pattern öğrenildi")
        
        # Heatmap
        self.heatmap_label.configure(text=self.brain.heatmap.get_heatmap_display())
    
    def _update_watchlist_ui(self):
        """Watchlist UI güncelle"""
        # Eski itemları temizle
        for widget in self.watchlist_frame.winfo_children():
            widget.destroy()
        
        items = self.watchlist.get_display_data()
        
        if not items:
            ctk.CTkLabel(self.watchlist_frame, text="Tarama devam ediyor...",
                        text_color="#666").pack(pady=50)
            return
        
        for item in items:
            card = ctk.CTkFrame(self.watchlist_frame, fg_color="#0f3460", 
                               corner_radius=10, height=80)
            card.pack(fill="x", pady=5)
            card.pack_propagate(False)
            
            # Sol - status ve symbol
            left = ctk.CTkFrame(card, fg_color="transparent")
            left.pack(side="left", padx=10, pady=10)
            
            status_icon = item['status']
            ctk.CTkLabel(left, text=status_icon, font=("Segoe UI", 20)).pack(side="left")
            
            info = ctk.CTkFrame(left, fg_color="transparent")
            info.pack(side="left", padx=10)
            
            symbol_text = item['symbol'].replace("USDT", "")
            dir_color = "#00ff88" if item['direction'] == "LONG" else "#ff4757"
            ctk.CTkLabel(info, text=symbol_text, font=("Segoe UI", 14, "bold")).pack(anchor="w")
            ctk.CTkLabel(info, text=item['direction'], font=("Segoe UI", 10),
                        text_color=dir_color).pack(anchor="w")
            
            # Sağ - readiness
            right = ctk.CTkFrame(card, fg_color="transparent")
            right.pack(side="right", padx=10)
            
            readiness = item['readiness']
            color = "#00ff88" if readiness >= 80 else "#ffd93d" if readiness >= 60 else "#888"
            ctk.CTkLabel(right, text=f"{readiness:.0f}%", 
                        font=("Consolas", 16, "bold"),
                        text_color=color).pack()
            ctk.CTkLabel(right, text=f"RSI:{item['current_rsi']:.0f}→{item['target_rsi']:.0f}",
                        font=("Consolas", 9), text_color="#888").pack()
    
    def _add_signal_card(self, symbol: str, direction: str, entry: float, 
                         sl: float, tp: float, confidence: float, reasons: List[str]):
        """Sinyal kartı ekle - CANLI PnL ile"""
        card = ctk.CTkFrame(self.signals_frame, fg_color="#0f3460", 
                           corner_radius=10, height=140)
        card.pack(fill="x", pady=5)
        card.pack_propagate(False)
        
        # Sol - bilgiler
        left = ctk.CTkFrame(card, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=15, pady=10)
        
        # Üst satır - symbol ve yön
        top = ctk.CTkFrame(left, fg_color="transparent")
        top.pack(fill="x")
        
        dir_color = "#00ff88" if direction == "LONG" else "#ff4757"
        dir_icon = "🟢" if direction == "LONG" else "🔴"
        
        ctk.CTkLabel(top, text=f"{dir_icon} {direction}", 
                    font=("Segoe UI", 12, "bold"),
                    text_color=dir_color).pack(side="left")
        ctk.CTkLabel(top, text=symbol.replace("USDT", ""), 
                    font=("Segoe UI", 16, "bold")).pack(side="left", padx=10)
        
        # Fiyatlar satırı
        prices = ctk.CTkFrame(left, fg_color="transparent")
        prices.pack(fill="x", pady=3)
        
        ctk.CTkLabel(prices, text=f"Entry: ${entry:.4f}", 
                    font=("Consolas", 10)).pack(side="left", padx=(0,10))
        ctk.CTkLabel(prices, text=f"SL: ${sl:.4f}", 
                    font=("Consolas", 10), text_color="#ff4757").pack(side="left", padx=(0,10))
        ctk.CTkLabel(prices, text=f"TP: ${tp:.4f}", 
                    font=("Consolas", 10), text_color="#00ff88").pack(side="left")
        
        # Canlı fiyat ve PnL satırı
        live_frame = ctk.CTkFrame(left, fg_color="#16213e", corner_radius=5)
        live_frame.pack(fill="x", pady=5)
        
        current_label = ctk.CTkLabel(live_frame, text=f"Şu an: ${entry:.4f}", 
                                     font=("Consolas", 11, "bold"))
        current_label.pack(side="left", padx=10, pady=5)
        
        pnl_label = ctk.CTkLabel(live_frame, text="P&L: +0.00%", 
                                 font=("Consolas", 12, "bold"),
                                 text_color="#00ff88")
        pnl_label.pack(side="left", padx=20)
        
        # Sağ - güven ve durum
        right = ctk.CTkFrame(card, fg_color="transparent", width=100)
        right.pack(side="right", padx=15)
        right.pack_propagate(False)
        
        conf_color = "#00ff88" if confidence >= 75 else "#ffd93d" if confidence >= 60 else "#888"
        ctk.CTkLabel(right, text=f"{confidence:.0f}%", 
                    font=("Consolas", 24, "bold"),
                    text_color=conf_color).pack()
        ctk.CTkLabel(right, text="güven", font=("Segoe UI", 9),
                    text_color="#666").pack()
        
        # Açıklama güncelle
        self.explain_text.configure(text=" | ".join(reasons[:4]))
        
        # Sinyal bilgisini kaydet (canlı güncelleme için)
        self.active_signals[symbol] = {
            'entry': entry, 
            'direction': direction,
            'sl': sl,
            'tp': tp,
            'card': card,
            'current_label': current_label,
            'pnl_label': pnl_label,
            'time': datetime.now()
        }
        self.signal_count_label.configure(text=f"{len(self.active_signals)} sinyal")
    
    async def _update_pnl(self):
        """Aktif sinyallerin PnL'ini güncelle"""
        import aiohttp
        
        if not self.active_signals:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                # Tüm fiyatları tek seferde çek
                url = "https://fapi.binance.com/fapi/v1/ticker/price"
                async with session.get(url) as resp:
                    prices = await resp.json()
                    price_dict = {p['symbol']: float(p['price']) for p in prices}
                
                total_pnl = 0
                
                for symbol, info in list(self.active_signals.items()):
                    if symbol not in price_dict:
                        continue
                    
                    current_price = price_dict[symbol]
                    entry = info['entry']
                    direction = info['direction']
                    sl = info['sl']
                    tp = info['tp']
                    
                    # PnL hesapla
                    if direction == "LONG":
                        pnl_pct = ((current_price - entry) / entry) * 100
                    else:
                        pnl_pct = ((entry - current_price) / entry) * 100
                    
                    total_pnl += pnl_pct
                    
                    # SL veya TP vuruldu mu?
                    hit_sl = (direction == "LONG" and current_price <= sl) or \
                             (direction == "SHORT" and current_price >= sl)
                    hit_tp = (direction == "LONG" and current_price >= tp) or \
                             (direction == "SHORT" and current_price <= tp)
                    
                    # UI güncelle
                    pnl_color = "#00ff88" if pnl_pct >= 0 else "#ff4757"
                    pnl_text = f"P&L: {pnl_pct:+.2f}%"
                    
                    if hit_tp:
                        pnl_text = f"✅ TP HIT! {pnl_pct:+.2f}%"
                        pnl_color = "#00ff88"
                    elif hit_sl:
                        pnl_text = f"❌ SL HIT! {pnl_pct:+.2f}%"
                        pnl_color = "#ff4757"
                    
                    # UI thread'de güncelle
                    def update_ui(s=symbol, cp=current_price, pt=pnl_text, pc=pnl_color):
                        if s in self.active_signals:
                            self.active_signals[s]['current_label'].configure(
                                text=f"Şu an: ${cp:.4f}")
                            self.active_signals[s]['pnl_label'].configure(
                                text=pt, text_color=pc)
                    
                    self.after(0, update_ui)
                    
                    # TP veya SL vurulduysa öğren ve kaldır
                    if hit_tp or hit_sl:
                        # Beyine öğret
                        result = TradeResult(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry,
                            exit_price=current_price,
                            entry_time=info['time'],
                            exit_time=datetime.now(),
                            pnl_percent=pnl_pct,
                            is_win=hit_tp,
                            rsi=50,  # TODO: gerçek RSI
                            volume_ratio=1.0,
                            trend="up" if direction == "LONG" else "down",
                            score=70
                        )
                        self.brain.learn(result)
                        
                        # Listeden çıkar (biraz bekle)
                        await asyncio.sleep(3)
                        if symbol in self.active_signals:
                            self.after(0, lambda s=symbol: self._remove_signal(s))
                
                # Toplam PnL güncelle
                pnl_color = "#00ff88" if total_pnl >= 0 else "#ff4757"
                self.after(0, lambda: self.pnl_label.configure(
                    text=f"${total_pnl:.2f}%", text_color=pnl_color))
                
        except Exception as e:
            print(f"[PnL] Hata: {e}")
    
    def _remove_signal(self, symbol: str):
        """Sinyali kaldır"""
        if symbol in self.active_signals:
            card = self.active_signals[symbol].get('card')
            if card:
                card.destroy()
            del self.active_signals[symbol]
            self.signal_count_label.configure(text=f"{len(self.active_signals)} sinyal")
    
    def _start(self):
        """Başlat"""
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="▶ Çalışıyor...")
        
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._main_loop())
        
        threading.Thread(target=run, daemon=True).start()
    
    def _stop(self):
        """Durdur"""
        self.running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="⏸ Durduruldu")
    
    async def _main_loop(self):
        """Ana döngü - GERÇEK RSI ile"""
        import aiohttp
        import numpy as np
        
        def calc_rsi(closes, period=14):
            """RSI hesapla"""
            if len(closes) < period + 1:
                return 50
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        while self.running:
            try:
                self.after(0, lambda: self.status_label.configure(text="🔍 Taranıyor..."))
                
                # Scanner ile coin bul
                results = await self.scanner.scan_1m_volatility()
                
                if results:
                    async with aiohttp.ClientSession() as session:
                        for coin in results[:15]:
                            try:
                                # Klines verisi al (RSI için)
                                kline_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={coin.symbol}&interval=5m&limit=30"
                                async with session.get(kline_url) as resp:
                                    klines = await resp.json()
                                    closes = [float(k[4]) for k in klines]
                                    volumes = [float(k[5]) for k in klines]
                                    
                                    # RSI hesapla
                                    rsi = calc_rsi(closes)
                                    
                                    # Volume ratio hesapla
                                    if len(volumes) >= 20:
                                        avg_vol = np.mean(volumes[-20:])
                                        current_vol = volumes[-1]
                                        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                                    else:
                                        vol_ratio = 1.0
                                
                                # Order book verisi al
                                url = f"https://fapi.binance.com/fapi/v1/depth?symbol={coin.symbol}&limit=10"
                                async with session.get(url) as resp:
                                    data = await resp.json()
                                    bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
                                    asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
                                    
                                    if bids and asks:
                                        # Predictor analizi
                                        pred = await self.predictor.analyze(coin.symbol, coin.price, bids, asks)
                                        
                                        # Yön belirleme (RSI'ya göre)
                                        if rsi < 35:
                                            direction = "LONG"
                                            trend = "up"
                                        elif rsi > 65:
                                            direction = "SHORT"
                                            trend = "down"
                                        else:
                                            direction = "LONG" if pred.predicted_direction == "up" else "SHORT"
                                            trend = pred.predicted_direction
                                        
                                        # Beyin kararı
                                        decision = self.brain.decide(
                                            symbol=coin.symbol,
                                            rsi=rsi,
                                            volume_ratio=vol_ratio,
                                            trend=trend,
                                            score=pred.total_score,
                                            direction=direction
                                        )
                                        
                                        print(f"[Brain] {coin.symbol}: RSI={rsi:.0f}, Vol={vol_ratio:.1f}x, {decision.action} ({decision.confidence:.0f}%)")
                                        
                                        if decision.action == "SIGNAL":
                                            # Sinyal ver!
                                            entry = coin.price
                                            sl = entry * (0.99 if direction == "LONG" else 1.01)
                                            tp = entry * (1.02 if direction == "LONG" else 0.98)
                                            
                                            self.after(0, lambda s=coin.symbol, d=direction, e=entry, 
                                                      slv=sl, tpv=tp, c=decision.confidence, r=decision.reasons:
                                                      self._add_signal_card(s, d, e, slv, tpv, c, r))
                                            
                                            # Sesli uyarı
                                            if self.voice.enabled:
                                                self.voice.signal_alert(coin.symbol, direction, 
                                                                       int(decision.confidence))
                                        
                                        elif decision.action == "WATCH":
                                            # Watchlist'e ekle
                                            self.watchlist.add(
                                                symbol=coin.symbol,
                                                direction=direction,
                                                current_price=coin.price,
                                                current_rsi=rsi,
                                                current_volume=vol_ratio,
                                                score=pred.total_score
                                            )
                            except Exception as e:
                                print(f"[Brain] {coin.symbol} hata: {e}")
                                continue
                
                # Watchlist güncelle
                self.after(0, self._update_watchlist_ui)
                
                # HOT items kontrol
                for hot in self.watchlist.get_hot_items():
                    entry = hot.ideal_entry
                    sl = hot.stop_loss
                    tp = hot.take_profit
                    
                    self.after(0, lambda s=hot.symbol, d=hot.direction, e=entry, 
                              slv=sl, tpv=tp:
                              self._add_signal_card(s, d, e, slv, tpv, 85, 
                                                   ["Watchlist'ten HOT!", f"RSI:{hot.current_rsi:.0f}"]))
                    
                    self.watchlist.mark_entered(hot.symbol)
                    
                    if self.voice.enabled:
                        self.voice.signal_alert(hot.symbol, hot.direction, 85)
                
                # Brain status güncelle
                self.after(0, self._update_brain_status)
                
                # PnL güncelle
                await self._update_pnl()
                
                self.after(0, lambda: self.status_label.configure(
                    text=f"✓ {datetime.now().strftime('%H:%M:%S')}"
                ))
                
                await asyncio.sleep(5)  # Daha sık güncelle
                
            except Exception as e:
                print(f"[Brain] Hata: {e}")
                self.after(0, lambda: self.status_label.configure(text=f"⚠ Hata"))
                await asyncio.sleep(5)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🧠 TRADING BRAIN - ÖĞRENEN ZEKA SİSTEMİ                ║
    ║   ─────────────────────────────────────────               ║
    ║                                                           ║
    ║   ⚡ Her işlemden öğrenen beyin                           ║
    ║   🔥 Akıllı watchlist (HOT/READY/WATCHING)               ║
    ║   📊 Isı haritası & pattern recognition                  ║
    ║   🎯 Çoklu onay sistemi                                   ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    app = BrainDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
