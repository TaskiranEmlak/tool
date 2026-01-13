# Ultimate Scalping Dashboard
"""
3 Panelli Modern Dashboard - CustomTkinter
Sol: Radar (Fırsat veren coinler)
Orta: Hedef (Seçili coin detay)
Sağ: Bağlam (BTC, Korelasyon, Risk)
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import asyncio
import threading
from datetime import datetime
from typing import List, Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scanner import VolatilityScanner
from core.predictor import Predictor, CoinPrediction
from core.coin_analyzer import CoinAnalyzer, CoinAnalysis
from core.database import Database
from core.proven_strategy import get_strategy  # Smart entry system
from indicators.benford import WashTradingFilter
from tools.helpers import VoiceAlert, CorrelationTracker, MarketRegimeDetector, RiskCalculator

# Tema ayarları
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class UltimateDashboard(ctk.CTk):
    """
    Ultimate Scalping Dashboard.
    Tek ekran, tam hakimiyet.
    """
    
    def __init__(self):
        super().__init__()
        
        # Pencere ayarları
        self.title("🎯 Ultimate Scalping Dashboard")
        self.geometry("1400x900")
        self.minsize(1200, 700)
        
        # Bilesenler
        self.scanner = VolatilityScanner()
        self.predictor = Predictor()
        self.analyzer = CoinAnalyzer()
        self.db = Database()  # Kalici hafiza
        self.wash_filter = WashTradingFilter()  # Sahte hacim filtresi
        self.voice = VoiceAlert(enabled=True)
        self.correlation = CorrelationTracker()
        self.regime_detector = MarketRegimeDetector()
        self.risk_calc = RiskCalculator(account_size=1000)
        
        # Sinyal sayaci
        self.signal_count = 0
        self.win_count = 0
        
        # Smart Entry Strategy
        self.smart_strategy = get_strategy()
        
        # İzleme Listesi (Watchlist)
        self.watchlist: Dict[str, dict] = {}  # {symbol: {entry, sl, tp, reason}}
        
        # Durum
        self.running = False
        self.selected_coin = ""
        self.predictions: Dict[str, CoinPrediction] = {}
        
        # UI oluştur
        self._create_top_bar()
        self._create_main_panels()
        self._create_bottom_bar()
        
        # Başlangıç verileri
        self._initial_load()
    
    def _create_top_bar(self):
        """Üst bar - Trafik ışığı ve genel bilgiler"""
        self.top_bar = ctk.CTkFrame(self, height=60, corner_radius=0)
        self.top_bar.pack(fill="x", padx=5, pady=5)
        self.top_bar.pack_propagate(False)
        
        # Trafik ışığı
        self.traffic_frame = ctk.CTkFrame(self.top_bar, width=200)
        self.traffic_frame.pack(side="left", padx=10, pady=5)
        
        self.traffic_label = ctk.CTkLabel(self.traffic_frame, text="🟡", font=("Segoe UI", 30))
        self.traffic_label.pack(side="left", padx=5)
        
        self.regime_label = ctk.CTkLabel(self.traffic_frame, text="BEKLE", 
                                         font=("Segoe UI", 14, "bold"))
        self.regime_label.pack(side="left", padx=5)
        
        # BTC bilgisi
        self.btc_frame = ctk.CTkFrame(self.top_bar, width=200)
        self.btc_frame.pack(side="left", padx=20, pady=5)
        
        ctk.CTkLabel(self.btc_frame, text="BTC:", font=("Segoe UI", 12)).pack(side="left")
        self.btc_price_label = ctk.CTkLabel(self.btc_frame, text="$0", 
                                            font=("Consolas", 14, "bold"))
        self.btc_price_label.pack(side="left", padx=5)
        self.btc_change_label = ctk.CTkLabel(self.btc_frame, text="+0%", 
                                             font=("Consolas", 12))
        self.btc_change_label.pack(side="left")
        
        # Status
        self.status_label = ctk.CTkLabel(self.top_bar, text="⏸ Hazır", 
                                         font=("Segoe UI", 11))
        self.status_label.pack(side="right", padx=20)
        
        # Sesli uyarı toggle
        self.voice_var = ctk.BooleanVar(value=True)
        self.voice_toggle = ctk.CTkSwitch(self.top_bar, text="🔊 Sesli", 
                                          variable=self.voice_var,
                                          command=self._toggle_voice)
        self.voice_toggle.pack(side="right", padx=10)
    
    def _create_main_panels(self):
        """3 ana panel"""
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # === SOL PANEL: RADAR ===
        self.radar_frame = ctk.CTkFrame(self.main_frame, width=280)
        self.radar_frame.pack(side="left", fill="y", padx=(0, 5))
        self.radar_frame.pack_propagate(False)
        
        ctk.CTkLabel(self.radar_frame, text="📡 RADAR", 
                    font=("Segoe UI", 14, "bold")).pack(pady=10)
        
        # Arama
        self.search_var = ctk.StringVar()
        self.search_entry = ctk.CTkEntry(self.radar_frame, placeholder_text="Ara...",
                                        textvariable=self.search_var, width=200)
        self.search_entry.pack(pady=5)
        self.search_var.trace("w", self._filter_radar)
        
        # Coin listesi (Treeview kullanacağız - scrollable)
        self.radar_tree_frame = ctk.CTkFrame(self.radar_frame)
        self.radar_tree_frame.pack(fill="both", expand=True, pady=5, padx=5)
        
        # Treeview with dark style
        style = ttk.Style()
        style.configure("Radar.Treeview", 
                       background="#2b2b2b", foreground="white",
                       fieldbackground="#2b2b2b", rowheight=28)
        style.configure("Radar.Treeview.Heading",
                       background="#1f1f1f", foreground="white")
        style.map("Radar.Treeview", background=[("selected", "#1f6aa5")])
        
        columns = ("symbol", "change", "score")
        self.radar_tree = ttk.Treeview(self.radar_tree_frame, columns=columns, 
                                       show="headings", style="Radar.Treeview", height=25)
        self.radar_tree.heading("symbol", text="Coin")
        self.radar_tree.heading("change", text="1m%")
        self.radar_tree.heading("score", text="Skor")
        self.radar_tree.column("symbol", width=100)
        self.radar_tree.column("change", width=70, anchor="e")
        self.radar_tree.column("score", width=50, anchor="e")
        self.radar_tree.pack(fill="both", expand=True)
        self.radar_tree.bind("<<TreeviewSelect>>", self._on_radar_select)
        
        self.radar_count = ctk.CTkLabel(self.radar_frame, text="0 coin")
        self.radar_count.pack(pady=5)
        
        # === ORTA PANEL: HEDEF ===
        self.target_frame = ctk.CTkFrame(self.main_frame)
        self.target_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Hedef başlık
        self.target_header = ctk.CTkFrame(self.target_frame)
        self.target_header.pack(fill="x", pady=10, padx=10)
        
        self.target_symbol = ctk.CTkLabel(self.target_header, text="← Coin Seç", 
                                          font=("Segoe UI", 20, "bold"))
        self.target_symbol.pack(side="left")
        
        self.target_price = ctk.CTkLabel(self.target_header, text="", 
                                         font=("Consolas", 18))
        self.target_price.pack(side="left", padx=20)
        
        self.target_change = ctk.CTkLabel(self.target_header, text="", 
                                          font=("Consolas", 14))
        self.target_change.pack(side="left")
        
        # Yenile butonu
        self.refresh_btn = ctk.CTkButton(self.target_header, text="🔄", width=40,
                                        command=self._refresh_target)
        self.refresh_btn.pack(side="right")
        
        # Scrollable içerik
        self.target_scroll = ctk.CTkScrollableFrame(self.target_frame)
        self.target_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder
        self.target_placeholder = ctk.CTkLabel(self.target_scroll, 
                                               text="Radar'dan bir coin seçin",
                                               font=("Segoe UI", 16))
        self.target_placeholder.pack(pady=50)
        
        # === SAĞ PANEL: BAĞLAM ===
        self.context_frame = ctk.CTkFrame(self.main_frame, width=280)
        self.context_frame.pack(side="right", fill="y", padx=(5, 0))
        self.context_frame.pack_propagate(False)
        
        ctk.CTkLabel(self.context_frame, text="🌍 BAĞLAM", 
                    font=("Segoe UI", 14, "bold")).pack(pady=10)
        
        # Korelasyon
        self.corr_frame = ctk.CTkFrame(self.context_frame)
        self.corr_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(self.corr_frame, text="📊 BTC Korelasyon", 
                    font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.corr_value = ctk.CTkLabel(self.corr_frame, text="0.00", 
                                       font=("Consolas", 16, "bold"))
        self.corr_value.pack(anchor="w")
        self.corr_bar = ctk.CTkProgressBar(self.corr_frame, width=200)
        self.corr_bar.pack(fill="x", pady=5)
        self.corr_bar.set(0)
        self.corr_comment = ctk.CTkLabel(self.corr_frame, text="Veri yok", 
                                         font=("Segoe UI", 10))
        self.corr_comment.pack(anchor="w")
        
        # Market verileri
        self.market_frame = ctk.CTkFrame(self.context_frame)
        self.market_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(self.market_frame, text="📈 Market Verileri", 
                    font=("Segoe UI", 11, "bold")).pack(anchor="w")
        
        self.funding_label = ctk.CTkLabel(self.market_frame, text="Funding: -")
        self.funding_label.pack(anchor="w")
        self.ls_label = ctk.CTkLabel(self.market_frame, text="Long/Short: -")
        self.ls_label.pack(anchor="w")
        self.oi_label = ctk.CTkLabel(self.market_frame, text="OI: -")
        self.oi_label.pack(anchor="w")
        
        # Risk hesaplayıcı
        self.risk_frame = ctk.CTkFrame(self.context_frame)
        self.risk_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(self.risk_frame, text="💰 Risk Hesap", 
                    font=("Segoe UI", 11, "bold")).pack(anchor="w")
        
        # Kasa girişi
        kasa_row = ctk.CTkFrame(self.risk_frame)
        kasa_row.pack(fill="x", pady=5)
        ctk.CTkLabel(kasa_row, text="Kasa $:").pack(side="left")
        self.kasa_entry = ctk.CTkEntry(kasa_row, width=80, placeholder_text="1000")
        self.kasa_entry.pack(side="left", padx=5)
        self.kasa_entry.insert(0, "1000")
        
        self.sl_label = ctk.CTkLabel(self.risk_frame, text="SL: -%", 
                                     font=("Consolas", 12))
        self.sl_label.pack(anchor="w")
        self.maxpos_label = ctk.CTkLabel(self.risk_frame, text="Max: $0", 
                                         font=("Consolas", 12, "bold"))
        self.maxpos_label.pack(anchor="w")
        self.risk_comment = ctk.CTkLabel(self.risk_frame, text="", 
                                         font=("Segoe UI", 10), wraplength=220)
        self.risk_comment.pack(anchor="w")
        
        # Sinyal log
        self.signal_frame = ctk.CTkFrame(self.context_frame)
        self.signal_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(self.signal_frame, text="🎯 Son Sinyaller", 
                    font=("Segoe UI", 11, "bold")).pack(anchor="w")
        
        self.signal_text = ctk.CTkTextbox(self.signal_frame, height=150, 
                                          font=("Consolas", 10))
        self.signal_text.pack(fill="both", expand=True)
    
    def _create_bottom_bar(self):
        """Alt bar - Kontroller"""
        self.bottom_bar = ctk.CTkFrame(self, height=50, corner_radius=0)
        self.bottom_bar.pack(fill="x", padx=5, pady=5)
        
        self.start_btn = ctk.CTkButton(self.bottom_bar, text="▶ BAŞLAT", 
                                       fg_color="#238636", hover_color="#2ea043",
                                       command=self._start, width=120)
        self.start_btn.pack(side="left", padx=10, pady=10)
        
        self.stop_btn = ctk.CTkButton(self.bottom_bar, text="⏹ DURDUR", 
                                      fg_color="#da3633", hover_color="#f85149",
                                      command=self._stop, width=120, state="disabled")
        self.stop_btn.pack(side="left", padx=5, pady=10)
        
        # Bilgi
        self.info_label = ctk.CTkLabel(self.bottom_bar, 
                                       text="Scalping Dashboard v1.0 | Tek Ekran, Tam Hakimiyet",
                                       font=("Segoe UI", 10))
        self.info_label.pack(side="right", padx=20)
    
    # ===========================
    # CALLBACKS & LOGIC
    # ===========================
    
    def _initial_load(self):
        """Başlangıç verilerini yükle ve geçmiş performansı göster"""
        self._update_regime()
        
        # Veritabanından geçmiş performans
        try:
            stats = self.db.get_stats()
            if stats.get('signals', 0) > 0:
                win_rate = stats.get('signals', 0)
                self.info_label.configure(
                    text=f"DB: {stats.get('signals', 0)} sinyal | Scalping Dashboard v2.0"
                )
        except Exception as e:
            print(f"[Dashboard] DB yükleme hatası: {e}")
    
    def _toggle_voice(self):
        """Sesli uyarı toggle"""
        self.voice.enabled = self.voice_var.get()
    
    def _filter_radar(self, *args):
        """Radar filtrele"""
        search = self.search_var.get().upper()
        for item in self.radar_tree.get_children():
            values = self.radar_tree.item(item)["values"]
            if search not in str(values[0]):
                self.radar_tree.detach(item)
    
    def _on_radar_select(self, event):
        """Radar'dan coin seçildi"""
        selection = self.radar_tree.selection()
        if selection:
            item = self.radar_tree.item(selection[0])
            symbol = item["values"][0]
            self.selected_coin = symbol
            self._load_target(symbol)
    
    def _refresh_target(self):
        """Hedef yenile"""
        if self.selected_coin:
            self._load_target(self.selected_coin)
    
    def _load_target(self, symbol: str):
        """Hedef coin yükle"""
        self.target_symbol.configure(text=f"🎯 {symbol}")
        
        def load():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(self.analyzer.analyze(symbol))
            risk = loop.run_until_complete(self.risk_calc.calculate(symbol))
            self.after(0, lambda: self._display_target(analysis, risk))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _display_target(self, analysis: CoinAnalysis, risk):
        """Hedef göster"""
        # Başlık
        price_str = f"${analysis.price:,.4f}" if analysis.price < 1 else f"${analysis.price:,.2f}"
        self.target_price.configure(text=price_str)
        
        change_color = "#3fb950" if analysis.change_24h > 0 else "#f85149"
        self.target_change.configure(text=f"{analysis.change_24h:+.2f}%", 
                                     text_color=change_color)
        
        # İçeriği temizle
        for widget in self.target_scroll.winfo_children():
            widget.destroy()
        
        # TF Analizi
        tf_frame = ctk.CTkFrame(self.target_scroll)
        tf_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(tf_frame, text="📊 TIMEFRAME", 
                    font=("Segoe UI", 12, "bold")).pack(anchor="w")
        
        tf_row = ctk.CTkFrame(tf_frame)
        tf_row.pack(fill="x", pady=5)
        
        for tf in [analysis.tf_15m, analysis.tf_5m, analysis.tf_3m, analysis.tf_1m]:
            if tf:
                icon = "↗" if tf.trend == "up" else "↘" if tf.trend == "down" else "→"
                color = "#3fb950" if tf.trend == "up" else "#f85149" if tf.trend == "down" else "#8b949e"
                
                box = ctk.CTkFrame(tf_row, width=70, height=60)
                box.pack(side="left", padx=5)
                box.pack_propagate(False)
                
                ctk.CTkLabel(box, text=tf.timeframe, font=("Segoe UI", 10)).pack()
                ctk.CTkLabel(box, text=icon, font=("Segoe UI", 20), text_color=color).pack()
                ctk.CTkLabel(box, text=f"{tf.change_percent:+.2f}%", 
                            font=("Consolas", 9), text_color=color).pack()
        
        if analysis.tf_summary:
            ctk.CTkLabel(tf_frame, text=f"💬 {analysis.tf_summary}", 
                        font=("Segoe UI", 10), wraplength=500).pack(anchor="w", pady=5)
        
        # OBI
        obi_frame = ctk.CTkFrame(self.target_scroll)
        obi_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(obi_frame, text="📈 ALIŞ/SATIŞ BASKISI", 
                    font=("Segoe UI", 12, "bold")).pack(anchor="w")
        
        obi_bar = ctk.CTkProgressBar(obi_frame, width=400)
        obi_bar.pack(fill="x", pady=5)
        obi_bar.set(analysis.bid_pressure / 100)
        
        obi_row = ctk.CTkFrame(obi_frame)
        obi_row.pack(fill="x")
        ctk.CTkLabel(obi_row, text=f"Alış: {analysis.bid_pressure:.0f}%", 
                    text_color="#3fb950").pack(side="left")
        ctk.CTkLabel(obi_row, text=f"Satış: {analysis.ask_pressure:.0f}%", 
                    text_color="#f85149").pack(side="right")
        
        if analysis.pressure_comment:
            ctk.CTkLabel(obi_frame, text=f"💬 {analysis.pressure_comment}", 
                        font=("Segoe UI", 10)).pack(anchor="w", pady=5)
        
        # Market
        self.funding_label.configure(text=f"Funding: {analysis.funding_rate*100:.4f}%")
        self.ls_label.configure(text=f"Long/Short: {analysis.long_percent:.0f}%/{analysis.short_percent:.0f}%")
        self.oi_label.configure(text=f"OI: ${analysis.open_interest/1e9:.2f}B")
        
        # Risk
        self.sl_label.configure(text=f"SL: {risk.suggested_sl:.2f}%")
        self.maxpos_label.configure(text=f"Max: ${risk.max_position:,.0f}")
        self.risk_comment.configure(text=risk.comment)
        
        # Sonuç
        result_frame = ctk.CTkFrame(self.target_scroll)
        result_frame.pack(fill="x", pady=10)
        
        risk_colors = {"DUSUK": "#3fb950", "ORTA": "#d29922", "YUKSEK": "#f85149"}
        risk_color = risk_colors.get(analysis.risk_level, "#8b949e")
        
        ctk.CTkLabel(result_frame, text=f"🎯 SONUÇ: {analysis.risk_level}", 
                    font=("Segoe UI", 14, "bold"), text_color=risk_color).pack(anchor="w")
        ctk.CTkLabel(result_frame, text=analysis.final_verdict, 
                    font=("Segoe UI", 11), wraplength=500).pack(anchor="w", pady=5)
    
    def _update_regime(self):
        """Market rejimini güncelle"""
        def update():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            regime = loop.run_until_complete(self.regime_detector.detect())
            self.after(0, lambda: self._display_regime(regime))
        
        threading.Thread(target=update, daemon=True).start()
    
    def _display_regime(self, regime):
        """Rejimi göster"""
        icons = {"green": "🟢", "red": "🔴", "yellow": "🟡", "gray": "⚪"}
        texts = {"bullish": "YUKARI", "bearish": "ASAGI", "neutral": "BEKLE", "unknown": "?"}
        
        self.traffic_label.configure(text=icons.get(regime.color, "⚪"))
        self.regime_label.configure(text=texts.get(regime.regime, "?"))
        
        if regime.btc_price > 0:
            self.btc_price_label.configure(text=f"${regime.btc_price:,.0f}")
            change_color = "#3fb950" if regime.btc_change > 0 else "#f85149"
            self.btc_change_label.configure(text=f"{regime.btc_change:+.1f}%", 
                                           text_color=change_color)
    
    def _start(self):
        """Motoru başlat"""
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
        """Ana döngü"""
        import aiohttp
        
        regime_counter = 0
        
        while self.running:
            try:
                self.after(0, lambda: self.status_label.configure(text="🔍 Tarama..."))
                
                # Market rejimi (her 30 saniyede)
                regime_counter += 1
                if regime_counter >= 3:
                    self._update_regime()
                    regime_counter = 0
                
                # Volatil coinleri tara
                results = await self.scanner.scan_1m_volatility()
                
                if results:
                    predictions = []
                    
                    async with aiohttp.ClientSession() as session:
                        # BTC fiyatını al ve korelasyon tracker'a ekle
                        try:
                            btc_url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
                            async with session.get(btc_url) as resp:
                                btc_data = await resp.json()
                                btc_price = float(btc_data.get("price", 0))
                                if btc_price > 0:
                                    self.correlation.update_btc(btc_price)
                        except:
                            pass
                        
                        for coin in results[:15]:
                            try:
                                url = f"https://fapi.binance.com/fapi/v1/depth?symbol={coin.symbol}&limit=10"
                                async with session.get(url) as resp:
                                    data = await resp.json()
                                    bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
                                    asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
                                    
                                    if bids and asks:
                                        pred = await self.predictor.analyze(coin.symbol, coin.price, bids, asks)
                                        predictions.append((coin, pred))
                                        
                                        # Korelasyon tracker'a ekle
                                        self.correlation.update_coin(coin.symbol, coin.price)
                                        
                                        # Sinyal kontrolü
                                        if pred.total_score >= 55:
                                            self._add_signal(coin.symbol, pred)
                            except:
                                continue
                    
                    # Radar güncelle
                    self.after(0, lambda p=predictions: self._update_radar(p))
                    
                    # Seçili coin korelasyonunu güncelle
                    if self.selected_coin:
                        corr, comment = self.correlation.get_correlation(self.selected_coin)
                        self.after(0, lambda c=corr, t=comment: self._update_correlation(c, t))
                
                self.after(0, lambda: self.status_label.configure(
                    text=f"✓ {datetime.now().strftime('%H:%M:%S')}"
                ))
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(text=f"⚠ Hata"))
                await asyncio.sleep(5)
    
    def _update_radar(self, predictions):
        """Radar guncelle - Benford filtresi ile"""
        for item in self.radar_tree.get_children():
            self.radar_tree.delete(item)
        
        for coin, pred in predictions:
            # Benford wash trading kontrolu
            self.wash_filter.add_volume(coin.symbol, coin.volume if hasattr(coin, 'volume') else 1000000)
            is_suspicious = self.wash_filter.is_suspicious(coin.symbol)
            
            # Icon ve tag belirleme
            warning = "⚠️" if is_suspicious else ""
            
            if pred.total_score >= 55:
                tag = "hot"
            elif is_suspicious:
                tag = "suspicious"
            else:
                tag = "normal"
            
            self.radar_tree.insert("", "end", values=(
                f"{warning}{coin.symbol}",
                f"{coin.price_change_percent:+.2f}%",
                f"{pred.total_score:.0f}"
            ), tags=(tag,))
        
        self.radar_tree.tag_configure("hot", foreground="#f0883e")  # Turuncu - guclue sinyal
        self.radar_tree.tag_configure("suspicious", foreground="#f85149")  # Kirmizi - wash trading
        self.radar_count.configure(text=f"{len(predictions)} coin")
    
    def _update_correlation(self, corr: float, comment: str):
        """Korelasyon değerini güncelle"""
        self.corr_value.configure(text=f"{corr:.2f}")
        self.corr_bar.set(abs(corr))
        self.corr_comment.configure(text=comment)
        
        # Renk
        if abs(corr) < 0.5:
            self.corr_value.configure(text_color="#3fb950")  # Yeşil - bağımsız
        elif abs(corr) < 0.8:
            self.corr_value.configure(text_color="#d29922")  # Sarı
        else:
            self.corr_value.configure(text_color="#f85149")  # Kırmızı - BTC bağımlı
    
    def _add_signal(self, symbol: str, pred: CoinPrediction):
        """Sinyal ekle - SMART ENTRY ile güçlendirilmiş"""
        direction = "LONG" if pred.predicted_direction == "up" else "SHORT"
        icon = "🟢" if direction == "LONG" else "🔴"
        
        # RSI kontrolü (smart entry)
        rsi_value = getattr(pred, 'rsi', 50)  # Varsayılan 50
        is_good_entry = False
        entry_reason = ""
        
        if direction == "LONG" and rsi_value < 40:
            is_good_entry = True
            entry_reason = f"RSI={rsi_value:.0f} (oversold)"
        elif direction == "SHORT" and rsi_value > 60:
            is_good_entry = True
            entry_reason = f"RSI={rsi_value:.0f} (overbought)"
        elif pred.total_score >= 65:  # Çok güçlü sinyal
            is_good_entry = True
            entry_reason = f"Güçlü skor={pred.total_score:.0f}"
        
        if not is_good_entry:
            # İzleme listesine ekle, sinyal verme
            self.watchlist[symbol] = {
                'direction': direction,
                'score': pred.total_score,
                'rsi': rsi_value,
                'price': pred.current_price,
                'time': datetime.now()
            }
            watch_text = f"👁 WATCH {symbol} | {direction} | RSI:{rsi_value:.0f}\n"
            self.signal_text.insert("0.0", watch_text)
            return
        
        # Güzel giriş noktası - SİNYAL VER
        entry_price = pred.current_price
        stop_loss = entry_price * (0.99 if direction == "LONG" else 1.01)
        take_profit = entry_price * (1.02 if direction == "LONG" else 0.98)
        
        # Veritabanına kaydet
        try:
            signal_id = self.db.insert_signal(
                symbol=symbol,
                direction=pred.predicted_direction,
                confidence=pred.confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"{entry_reason}; " + ("; ".join(pred.reasons[:2]) if pred.reasons else ""),
                timestamp=datetime.now()
            )
            self.signal_count += 1
        except Exception as e:
            print(f"[Dashboard] Sinyal kayıt hatası: {e}")
        
        # İzleme listesinden çıkar
        if symbol in self.watchlist:
            del self.watchlist[symbol]
        
        # Sinyal göster
        text = f"{icon} {direction} {symbol} | {entry_reason} | SL:{stop_loss:.4f} TP:{take_profit:.4f}\n"
        self.signal_text.insert("0.0", text)
        
        # Sesli uyarı
        if self.voice.enabled:
            self.voice.signal_alert(symbol, pred.predicted_direction, int(pred.total_score))


def main():
    print("""
    ============================================================
    
       🎯 ULTIMATE SCALPING DASHBOARD
       ─────────────────────────────
       Tek Ekran, Tam Hakimiyet
    
       Sol: Radar (Fırsat veren coinler)
       Orta: Hedef (Seçili coin detay)
       Sağ: Bağlam (BTC, Korelasyon, Risk)
    
       🚦 Trafik ışığına dikkat et!
       🔊 Sesli uyarılar aktif
    
    ============================================================
    """)
    
    app = UltimateDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
