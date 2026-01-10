# HFT Trading Tools - Coin Detay GUI
"""
Coin seçim ve detaylı analiz paneli.
"""

import tkinter as tk
from tkinter import ttk
import asyncio
import threading
from datetime import datetime
from typing import List, Optional, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.coin_analyzer import CoinAnalyzer, CoinAnalysis


class CoinDetailApp:
    """
    Coin Detay Uygulaması.
    Sol: Coin listesi | Sağ: Seçilen coin detayları
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔍 HFT Trading Tools - Coin Detay Paneli")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0d1117")
        
        # Analyzer
        self.analyzer = CoinAnalyzer()
        self.all_coins: List = []
        self.selected_symbol: str = ""
        self.current_analysis: Optional[CoinAnalysis] = None
        
        # Auto-refresh
        self.auto_refresh = True
        self.refresh_interval = 5000  # 5 saniye
        
        # UI
        self._create_styles()
        self._create_layout()
        
        # Başlangıç
        self._load_coins()
    
    def _create_styles(self):
        """Stiller"""
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("Dark.TFrame", background="#0d1117")
        style.configure("Dark.TLabel", background="#0d1117", foreground="#c9d1d9", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#0d1117", foreground="#58a6ff", font=("Segoe UI", 12, "bold"))
        style.configure("Header.TLabel", background="#0d1117", foreground="#f0f6fc", font=("Segoe UI", 14, "bold"))
        style.configure("Value.TLabel", background="#0d1117", foreground="#7ee787", font=("Consolas", 11))
        
        style.configure("Dark.Treeview",
                       background="#161b22",
                       foreground="#c9d1d9",
                       fieldbackground="#161b22",
                       font=("Consolas", 10),
                       rowheight=22)
        style.configure("Dark.Treeview.Heading",
                       background="#21262d",
                       foreground="#c9d1d9",
                       font=("Segoe UI", 9, "bold"))
        style.map("Dark.Treeview",
                 background=[("selected", "#388bfd")],
                 foreground=[("selected", "white")])
    
    def _create_layout(self):
        """Ana layout"""
        # Header
        header = ttk.Frame(self.root, style="Dark.TFrame")
        header.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(header, text="🔍 Coin Detay Paneli", style="Header.TLabel").pack(side=tk.LEFT)
        
        # Refresh butonu
        refresh_btn = tk.Button(header, text="🔄 Yenile", bg="#238636", fg="white",
                               font=("Segoe UI", 9), command=self._refresh_selected)
        refresh_btn.pack(side=tk.RIGHT, padx=5)
        
        # Auto-refresh toggle
        self.auto_var = tk.BooleanVar(value=True)
        auto_check = tk.Checkbutton(header, text="Otomatik (5sn)", bg="#0d1117", fg="#c9d1d9",
                                   selectcolor="#21262d", variable=self.auto_var,
                                   command=self._toggle_auto)
        auto_check.pack(side=tk.RIGHT, padx=5)
        
        # Main content
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Sol panel - Coin listesi
        left = ttk.Frame(main, style="Dark.TFrame", width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left.pack_propagate(False)
        
        # Arama
        search_frame = ttk.Frame(left, style="Dark.TFrame")
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Ara:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_coins)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, bg="#21262d", fg="white",
                               insertbackground="white", font=("Consolas", 10), width=15)
        search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Coin listesi
        ttk.Label(left, text="Tum Futures Coinleri", style="Title.TLabel").pack(anchor=tk.W)
        
        columns = ("symbol", "change")
        self.coin_list = ttk.Treeview(left, columns=columns, show="headings",
                                      style="Dark.Treeview", height=30)
        self.coin_list.heading("symbol", text="Sembol")
        self.coin_list.heading("change", text="24h%")
        self.coin_list.column("symbol", width=120)
        self.coin_list.column("change", width=80, anchor=tk.E)
        self.coin_list.pack(fill=tk.BOTH, expand=True, pady=5)
        self.coin_list.bind("<<TreeviewSelect>>", self._on_coin_select)
        
        self.coin_count_label = ttk.Label(left, text="0 coin", style="Dark.TLabel")
        self.coin_count_label.pack(anchor=tk.W)
        
        # Sağ panel - Detay
        right = ttk.Frame(main, style="Dark.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas + Scrollbar
        canvas = tk.Canvas(right, bg="#0d1117", highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
        self.detail_frame = ttk.Frame(canvas, style="Dark.TFrame")
        
        self.detail_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.detail_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Placeholder
        self.placeholder = ttk.Label(self.detail_frame, text="← Bir coin secin", 
                                    style="Header.TLabel")
        self.placeholder.pack(pady=50)
    
    def _load_coins(self):
        """Coin listesini yükle"""
        def load():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            coins = loop.run_until_complete(self.analyzer.get_all_symbols())
            self.root.after(0, lambda: self._populate_coins(coins))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _populate_coins(self, coins: list):
        """Coin listesini doldur"""
        self.all_coins = coins
        self._filter_coins()
    
    def _filter_coins(self, *args):
        """Arama filtresi"""
        search = self.search_var.get().upper()
        
        for item in self.coin_list.get_children():
            self.coin_list.delete(item)
        
        count = 0
        for symbol, price, change in self.all_coins:
            if search in symbol:
                tag = "up" if change > 0 else "down"
                self.coin_list.insert("", tk.END, values=(symbol, f"{change:+.2f}%"), tags=(tag,))
                count += 1
        
        self.coin_list.tag_configure("up", foreground="#3fb950")
        self.coin_list.tag_configure("down", foreground="#f85149")
        self.coin_count_label.config(text=f"{count} coin")
    
    def _on_coin_select(self, event):
        """Coin seçildi"""
        selection = self.coin_list.selection()
        if selection:
            item = self.coin_list.item(selection[0])
            symbol = item["values"][0]
            self.selected_symbol = symbol
            self._analyze_coin(symbol)
    
    def _analyze_coin(self, symbol: str):
        """Coin analizi başlat"""
        def analyze():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(self.analyzer.analyze(symbol))
            self.root.after(0, lambda: self._display_analysis(analysis))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def _display_analysis(self, analysis: CoinAnalysis):
        """Analizi göster"""
        self.current_analysis = analysis
        
        # Mevcut içeriği temizle
        for widget in self.detail_frame.winfo_children():
            widget.destroy()
        
        # === HEADER ===
        header = ttk.Frame(self.detail_frame, style="Dark.TFrame")
        header.pack(fill=tk.X, pady=10)
        
        ttk.Label(header, text=f"💰 {analysis.symbol}", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text=f"${analysis.price:,.4f}" if analysis.price < 1 else f"${analysis.price:,.2f}",
                 style="Value.TLabel").pack(side=tk.LEFT, padx=20)
        
        change_color = "#3fb950" if analysis.change_24h > 0 else "#f85149"
        change_label = tk.Label(header, text=f"24h: {analysis.change_24h:+.2f}%",
                               bg="#0d1117", fg=change_color, font=("Consolas", 11, "bold"))
        change_label.pack(side=tk.LEFT)
        
        # === TIMEFRAME ANALİZİ ===
        self._add_section("📊 TIMEFRAME ANALİZİ")
        
        tf_frame = ttk.Frame(self.detail_frame, style="Dark.TFrame")
        tf_frame.pack(fill=tk.X, padx=20, pady=5)
        
        for tf in [analysis.tf_15m, analysis.tf_5m, analysis.tf_3m, analysis.tf_1m]:
            if tf:
                icon = "↗" if tf.trend == "up" else "↘" if tf.trend == "down" else "→"
                color = "#3fb950" if tf.trend == "up" else "#f85149" if tf.trend == "down" else "#8b949e"
                
                tf_box = tk.Frame(tf_frame, bg="#21262d", padx=10, pady=5)
                tf_box.pack(side=tk.LEFT, padx=5)
                
                tk.Label(tf_box, text=tf.timeframe, bg="#21262d", fg="#8b949e", font=("Segoe UI", 9)).pack()
                tk.Label(tf_box, text=icon, bg="#21262d", fg=color, font=("Segoe UI", 16, "bold")).pack()
                tk.Label(tf_box, text=f"{tf.change_percent:+.2f}%", bg="#21262d", fg=color, font=("Consolas", 9)).pack()
        
        if analysis.tf_summary:
            self._add_comment(analysis.tf_summary)
        
        # === ALIŞ/SATIŞ BASKISI ===
        self._add_section("📈 ALIŞ/SATIŞ BASKISI")
        
        pressure_frame = ttk.Frame(self.detail_frame, style="Dark.TFrame")
        pressure_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Progress bar
        bar_frame = tk.Frame(pressure_frame, bg="#161b22", height=30)
        bar_frame.pack(fill=tk.X, pady=5)
        
        bid_width = int(analysis.bid_pressure * 3)
        ask_width = int(analysis.ask_pressure * 3)
        
        tk.Frame(bar_frame, bg="#3fb950", width=bid_width, height=25).place(x=0, y=2)
        tk.Frame(bar_frame, bg="#f85149", width=ask_width, height=25).place(x=300-ask_width, y=2)
        
        tk.Label(pressure_frame, text=f"Alış: {analysis.bid_pressure:.0f}%", bg="#0d1117", fg="#3fb950",
                font=("Consolas", 10)).pack(side=tk.LEFT)
        tk.Label(pressure_frame, text=f"Satış: {analysis.ask_pressure:.0f}%", bg="#0d1117", fg="#f85149",
                font=("Consolas", 10)).pack(side=tk.RIGHT)
        
        if analysis.pressure_comment:
            self._add_comment(analysis.pressure_comment)
        
        if analysis.bid_wall:
            self._add_comment(f"🧱 {analysis.bid_wall}", "#d29922")
        if analysis.ask_wall:
            self._add_comment(f"🧱 {analysis.ask_wall}", "#d29922")
        
        # === MARKET VERİLERİ ===
        self._add_section("📊 MARKET VERİLERİ")
        
        # Funding
        self._add_data_row("Funding Rate:", f"{analysis.funding_rate*100:.4f}%")
        if analysis.funding_comment:
            self._add_comment(analysis.funding_comment)
        
        # Long/Short
        self._add_data_row("Long/Short:", f"{analysis.long_percent:.0f}% / {analysis.short_percent:.0f}%")
        if analysis.ls_comment:
            self._add_comment(analysis.ls_comment)
        
        # OI
        self._add_data_row("Open Interest:", f"${analysis.open_interest/1e9:.2f}B")
        if analysis.oi_comment:
            self._add_comment(analysis.oi_comment)
        
        # === SONUÇ ===
        self._add_section("🎯 SONUÇ")
        
        risk_colors = {"DUSUK": "#3fb950", "ORTA": "#d29922", "YUKSEK": "#f85149"}
        risk_color = risk_colors.get(analysis.risk_level, "#8b949e")
        
        result_frame = tk.Frame(self.detail_frame, bg="#21262d", padx=15, pady=10)
        result_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(result_frame, text=f"Risk: {analysis.risk_level}", bg="#21262d", fg=risk_color,
                font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        tk.Label(result_frame, text=analysis.final_verdict, bg="#21262d", fg="#c9d1d9",
                font=("Segoe UI", 10), wraplength=500, justify=tk.LEFT).pack(anchor=tk.W, pady=5)
        
        # Timestamp
        ttk.Label(self.detail_frame, text=f"Son guncelleme: {datetime.now().strftime('%H:%M:%S')}",
                 style="Dark.TLabel").pack(pady=10)
        
        # Auto-refresh
        if self.auto_refresh and self.auto_var.get():
            self.root.after(self.refresh_interval, self._auto_refresh)
    
    def _add_section(self, title: str):
        """Bölüm başlığı ekle"""
        ttk.Label(self.detail_frame, text=title, style="Title.TLabel").pack(anchor=tk.W, pady=(15, 5), padx=10)
    
    def _add_data_row(self, label: str, value: str):
        """Veri satırı ekle"""
        row = ttk.Frame(self.detail_frame, style="Dark.TFrame")
        row.pack(fill=tk.X, padx=20, pady=2)
        ttk.Label(row, text=label, style="Dark.TLabel", width=15).pack(side=tk.LEFT)
        ttk.Label(row, text=value, style="Value.TLabel").pack(side=tk.LEFT)
    
    def _add_comment(self, text: str, color: str = "#58a6ff"):
        """Yorum ekle"""
        tk.Label(self.detail_frame, text=f"💬 {text}", bg="#0d1117", fg=color,
                font=("Segoe UI", 10), wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, padx=25, pady=2)
    
    def _refresh_selected(self):
        """Seçili coini yenile"""
        if self.selected_symbol:
            self._analyze_coin(self.selected_symbol)
    
    def _toggle_auto(self):
        """Auto-refresh toggle"""
        if self.auto_var.get() and self.selected_symbol:
            self._auto_refresh()
    
    def _auto_refresh(self):
        """Otomatik yenileme"""
        if self.auto_var.get() and self.selected_symbol:
            self._analyze_coin(self.selected_symbol)
    
    def run(self):
        """Uygulamayı başlat"""
        self.root.mainloop()


def main():
    app = CoinDetailApp()
    app.run()


if __name__ == "__main__":
    main()
