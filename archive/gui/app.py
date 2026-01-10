# HFT Trading Tools - Tkinter GUI v2
"""
Volatilite tarayıcı + İzleme Listesi için masaüstü arayüzü.
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import List, Optional, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TradingApp:
    """
    Ana Tkinter uygulaması.
    3 bölüm: Volatil Coinler | İzleme Listesi | Sinyaller
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 HFT Trading Tools - Volatilite Tarayıcı v2")
        self.root.geometry("1400x750")
        self.root.configure(bg="#0d1117")
        
        # Durum
        self.running = False
        
        # Callback'ler
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None
        
        # UI oluştur
        self._create_styles()
        self._create_header()
        self._create_main_content()
        self._create_status_bar()
    
    def _create_styles(self):
        """Tema ve stiller"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Koyu tema
        style.configure("Dark.TFrame", background="#0d1117")
        style.configure("Dark.TLabel", background="#0d1117", foreground="#c9d1d9", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#0d1117", foreground="#58a6ff", font=("Segoe UI", 13, "bold"))
        style.configure("Title.TLabel", background="#0d1117", foreground="#f0f6fc", font=("Segoe UI", 11, "bold"))
        
        # Treeview
        style.configure("Dark.Treeview",
                       background="#161b22",
                       foreground="#c9d1d9",
                       fieldbackground="#161b22",
                       font=("Consolas", 10),
                       rowheight=25)
        style.configure("Dark.Treeview.Heading",
                       background="#21262d",
                       foreground="#c9d1d9",
                       font=("Segoe UI", 9, "bold"))
        style.map("Dark.Treeview",
                 background=[("selected", "#388bfd")],
                 foreground=[("selected", "white")])
    
    def _create_header(self):
        """Üst başlık"""
        header = ttk.Frame(self.root, style="Dark.TFrame")
        header.pack(fill=tk.X, padx=15, pady=10)
        
        # Logo
        title = ttk.Label(header, text="🚀 HFT Trading Tools", style="Header.TLabel")
        title.pack(side=tk.LEFT)
        
        # Butonlar
        btn_frame = ttk.Frame(header, style="Dark.TFrame")
        btn_frame.pack(side=tk.RIGHT)
        
        self.start_btn = tk.Button(btn_frame, text="▶ BAŞLAT", bg="#238636", fg="white",
                                   font=("Segoe UI", 10, "bold"), padx=15, pady=3,
                                   command=self._on_start_click, cursor="hand2",
                                   activebackground="#2ea043")
        self.start_btn.pack(side=tk.LEFT, padx=3)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ DURDUR", bg="#da3633", fg="white",
                                  font=("Segoe UI", 10, "bold"), padx=15, pady=3,
                                  command=self._on_stop_click, state=tk.DISABLED, cursor="hand2",
                                  activebackground="#f85149")
        self.stop_btn.pack(side=tk.LEFT, padx=3)
    
    def _create_main_content(self):
        """Ana içerik - 3 panel"""
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        # ===== SOL PANEL: Volatil Coinler =====
        left_frame = ttk.Frame(main, style="Dark.TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(left_frame, text="📊 Volatil Coinler (1m)", style="Title.TLabel").pack(anchor=tk.W)
        
        columns = ("symbol", "price", "change", "volume")
        self.coins_table = ttk.Treeview(left_frame, columns=columns, show="headings",
                                        style="Dark.Treeview", height=18)
        
        self.coins_table.heading("symbol", text="Sembol")
        self.coins_table.heading("price", text="Fiyat")
        self.coins_table.heading("change", text="1m Δ%")
        self.coins_table.heading("volume", text="24h Vol")
        
        self.coins_table.column("symbol", width=90, anchor=tk.CENTER)
        self.coins_table.column("price", width=90, anchor=tk.E)
        self.coins_table.column("change", width=70, anchor=tk.E)
        self.coins_table.column("volume", width=80, anchor=tk.E)
        
        self.coins_table.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ===== ORTA PANEL: İzleme Listesi =====
        mid_frame = ttk.Frame(main, style="Dark.TFrame", width=350)
        mid_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        mid_frame.pack_propagate(False)
        
        ttk.Label(mid_frame, text="🎯 Tahmin Skorları", style="Title.TLabel").pack(anchor=tk.W)
        
        # İzleme sayacı
        self.watch_count_label = ttk.Label(mid_frame, text="0 coin izleniyor", style="Dark.TLabel")
        self.watch_count_label.pack(anchor=tk.W)
        
        # İzleme tablosu
        watch_cols = ("symbol", "obi", "score", "time")
        self.watch_table = ttk.Treeview(mid_frame, columns=watch_cols, show="headings",
                                        style="Dark.Treeview", height=8)
        
        self.watch_table.heading("symbol", text="Sembol")
        self.watch_table.heading("obi", text="OBI")
        self.watch_table.heading("score", text="Tahmin")
        self.watch_table.heading("time", text="Pot.%")
        
        self.watch_table.column("symbol", width=80, anchor=tk.CENTER)
        self.watch_table.column("obi", width=60, anchor=tk.E)
        self.watch_table.column("score", width=50, anchor=tk.E)
        self.watch_table.column("time", width=50, anchor=tk.E)
        
        self.watch_table.pack(fill=tk.X, pady=5)
        
        # Göstergeler
        ttk.Label(mid_frame, text="📈 Anlık Göstergeler", style="Title.TLabel").pack(anchor=tk.W, pady=(15, 5))
        
        self.indicators_frame = ttk.Frame(mid_frame, style="Dark.TFrame")
        self.indicators_frame.pack(fill=tk.X)
        
        self.indicator_labels = {}
        self._create_indicator_row("Analiz:", "0")
        self._create_indicator_row("Ort. Skor:", "0")
        self._create_indicator_row("En Yüksek:", "0")
        
        # ===== SAĞ PANEL: Sinyaller =====
        right_frame = ttk.Frame(main, style="Dark.TFrame", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        ttk.Label(right_frame, text="🎯 Sinyaller", style="Title.TLabel").pack(anchor=tk.W)
        
        # Sinyal sayacı
        self.signal_count_label = ttk.Label(right_frame, text="0 sinyal", style="Dark.TLabel")
        self.signal_count_label.pack(anchor=tk.W)
        
        # Sinyal listesi
        self.signals_text = tk.Text(right_frame, bg="#161b22", fg="#c9d1d9",
                                    font=("Consolas", 10), height=18, wrap=tk.WORD,
                                    relief=tk.FLAT, padx=5, pady=5)
        self.signals_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.signals_text.insert(tk.END, "Sinyal bekleniyor...\n")
        self.signals_text.config(state=tk.DISABLED)
    
    def _create_indicator_row(self, label: str, value: str):
        """Gösterge satırı"""
        row = ttk.Frame(self.indicators_frame, style="Dark.TFrame")
        row.pack(fill=tk.X, pady=1)
        
        ttk.Label(row, text=label, style="Dark.TLabel", width=12).pack(side=tk.LEFT)
        value_label = ttk.Label(row, text=value, style="Dark.TLabel", foreground="#58a6ff")
        value_label.pack(side=tk.RIGHT)
        self.indicator_labels[label] = value_label
    
    def _create_status_bar(self):
        """Alt durum çubuğu"""
        status_bar = ttk.Frame(self.root, style="Dark.TFrame")
        status_bar.pack(fill=tk.X, padx=15, pady=5)
        
        self.status_label = ttk.Label(status_bar, text="⏸ Beklemede", style="Dark.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(status_bar, text="", style="Dark.TLabel")
        self.time_label.pack(side=tk.RIGHT)
        
        self._update_time()
    
    def _update_time(self):
        """Saati güncelle"""
        now = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=now)
        self.root.after(1000, self._update_time)
    
    def _on_start_click(self):
        """Başlat butonu"""
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="▶ Çalışıyor...")
        if self.on_start:
            self.on_start()
    
    def _on_stop_click(self):
        """Durdur butonu"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏸ Durduruldu")
        if self.on_stop:
            self.on_stop()
    
    # ============ Veri Güncelleme Metodları ============
    
    def update_coins_table(self, coins: list):
        """Volatil coinler tablosu"""
        for item in self.coins_table.get_children():
            self.coins_table.delete(item)
        
        for coin in coins:
            symbol, price, change, volume = coin[:4]
            tag = "up" if change > 0 else "down"
            
            self.coins_table.insert("", tk.END, values=(
                symbol,
                f"${price:,.4f}" if price < 1 else f"${price:,.2f}",
                f"{change:+.2f}%",
                f"{volume/1e9:.1f}B" if volume > 1e9 else f"{volume/1e6:.0f}M"
            ), tags=(tag,))
        
        self.coins_table.tag_configure("up", foreground="#3fb950")
        self.coins_table.tag_configure("down", foreground="#f85149")
    
    def update_watchlist(self, watchlist: list):
        """İzleme listesi tablosu"""
        for item in self.watch_table.get_children():
            self.watch_table.delete(item)
        
        for item in watchlist:
            symbol, obi, score, elapsed_sec = item
            
            # Skor rengi
            if score >= 60:
                tag = "hot"
            elif score >= 40:
                tag = "warm"
            else:
                tag = "cool"
            
            self.watch_table.insert("", tk.END, values=(
                symbol,
                f"{obi:+.2f}",
                f"{score:.0f}",
                f"{elapsed_sec:.0f}s"
            ), tags=(tag,))
        
        self.watch_table.tag_configure("hot", foreground="#f0883e")
        self.watch_table.tag_configure("warm", foreground="#d29922")
        self.watch_table.tag_configure("cool", foreground="#8b949e")
        
        self.watch_count_label.config(text=f"{len(watchlist)} coin izleniyor")
    
    def update_indicators(self, watched: int, avg_score: float, top_score: float):
        """Göstergeleri güncelle"""
        self.indicator_labels["Analiz:"].config(text=str(watched))
        
        score_color = "#f0883e" if avg_score >= 50 else "#d29922" if avg_score >= 30 else "#8b949e"
        self.indicator_labels["Ort. Skor:"].config(text=f"{avg_score:.0f}", foreground=score_color)
        
        top_color = "#3fb950" if top_score >= 55 else "#d29922" if top_score >= 40 else "#8b949e"
        self.indicator_labels["En Yüksek:"].config(text=f"{top_score:.0f}", foreground=top_color)
    
    def add_signal(self, symbol: str, direction: str, price: float, score: float, reasons: list = None):
        """Sinyal ekle"""
        self.signals_text.config(state=tk.NORMAL)
        
        now = datetime.now().strftime("%H:%M:%S")
        icon = "🟢" if direction == "up" else "🔴"
        dir_text = "LONG" if direction == "up" else "SHORT"
        
        msg = f"[{now}] {icon} {dir_text} {symbol}\n"
        msg += f"  Fiyat: ${price:,.4f}\n"
        msg += f"  Skor: {score:.0f}/100\n"
        
        if reasons:
            for r in reasons[:3]:
                msg += f"  → {r}\n"
        
        msg += "─" * 30 + "\n"
        
        self.signals_text.insert(tk.END, msg)
        self.signals_text.see(tk.END)
        self.signals_text.config(state=tk.DISABLED)
        
        # Sinyal sayısını güncelle
        current = self.signal_count_label.cget("text")
        count = int(current.split()[0]) + 1
        self.signal_count_label.config(text=f"{count} sinyal")
    
    def update_status(self, message: str):
        """Durum güncelle"""
        self.status_label.config(text=message)
    
    def run(self):
        """Uygulamayı başlat"""
        self.root.mainloop()


if __name__ == "__main__":
    app = TradingApp()
    
    # Test verileri
    test_coins = [
        ("BTCUSDT", 97500.00, 0.85, 5_200_000_000),
        ("ETHUSDT", 3450.00, -0.62, 2_100_000_000),
        ("SOLUSDT", 185.50, 1.23, 890_000_000),
    ]
    
    test_watchlist = [
        ("BTCUSDT", 0.35, 72, 45),
        ("SOLUSDT", 0.28, 58, 120),
        ("DOGEUSDT", -0.42, 35, 200),
    ]
    
    app.update_coins_table(test_coins)
    app.update_watchlist(test_watchlist)
    app.update_indicators(3, 0.07, 72)
    app.add_signal("BTCUSDT", "up", 97500, 72, ["OBI pozitif (0.35)", "Momentum artıyor"])
    
    app.run()
