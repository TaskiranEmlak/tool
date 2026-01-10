# HFT Trading Tools - Modern Birlesik Arayuz v5
"""
Sekmeli modern arayuz - Tum ozellikler tek yerde.

Sekmeler:
1. RADAR - Volatil coinler ve tarama
2. ANALIZ - AI tahminleri ve grafikler
3. BACKTEST - Gecmis performans testi
4. ISTATISTIK - Sinyal gecmisi ve performans
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import List, Optional, Callable, Dict
import threading
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TradingApp:
    """
    Modern sekmeli trading arayuzu.
    Tum ozellikler tek uygulamada.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HFT Trading Tools v5.0 - God Mode")
        self.root.geometry("1400x850")
        self.root.configure(bg="#1a1a2e")
        
        # Callbacks
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None
        
        # Durum
        self.running = False
        self.signal_count = 0
        
        # UI olustur
        self._create_styles()
        self._create_header()
        self._create_notebook()
        self._create_status_bar()
        
        # Saat guncelle
        self._update_time()
    
    def _create_styles(self):
        """Tema ve stiller"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Renkler
        self.colors = {
            "bg": "#1a1a2e",
            "card": "#16213e",
            "accent": "#0f3460",
            "green": "#00d26a",
            "red": "#ff6b6b",
            "orange": "#f0883e",
            "text": "#e8e8e8",
            "muted": "#8b949e"
        }
        
        # Treeview stili
        style.configure("Dark.Treeview",
                       background=self.colors["card"],
                       foreground=self.colors["text"],
                       fieldbackground=self.colors["card"],
                       rowheight=28)
        style.configure("Dark.Treeview.Heading",
                       background=self.colors["accent"],
                       foreground=self.colors["text"],
                       font=("Segoe UI", 10, "bold"))
        style.map("Dark.Treeview", background=[("selected", self.colors["accent"])])
        
        # Notebook stili
        style.configure("TNotebook", background=self.colors["bg"])
        style.configure("TNotebook.Tab", 
                       background=self.colors["card"],
                       foreground=self.colors["text"],
                       padding=[20, 10],
                       font=("Segoe UI", 11, "bold"))
        style.map("TNotebook.Tab",
                 background=[("selected", self.colors["accent"])],
                 foreground=[("selected", self.colors["green"])])
        
        # Button stili
        style.configure("Start.TButton",
                       background=self.colors["green"],
                       foreground="white",
                       font=("Segoe UI", 11, "bold"),
                       padding=10)
        style.configure("Stop.TButton",
                       background=self.colors["red"],
                       foreground="white",
                       font=("Segoe UI", 11, "bold"),
                       padding=10)
    
    def _create_header(self):
        """Ust baslik"""
        header = tk.Frame(self.root, bg=self.colors["card"], height=70)
        header.pack(fill="x", padx=5, pady=5)
        header.pack_propagate(False)
        
        # Logo
        logo = tk.Label(header, text="HFT TRADING TOOLS", 
                       font=("Segoe UI", 18, "bold"),
                       bg=self.colors["card"], fg=self.colors["green"])
        logo.pack(side="left", padx=20, pady=15)
        
        # Versiyon
        ver = tk.Label(header, text="v5.0 God Mode", 
                      font=("Segoe UI", 10),
                      bg=self.colors["card"], fg=self.colors["muted"])
        ver.pack(side="left", pady=15)
        
        # Butonlar
        btn_frame = tk.Frame(header, bg=self.colors["card"])
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="BASLAT", 
                                   bg=self.colors["green"], fg="white",
                                   font=("Segoe UI", 11, "bold"),
                                   width=10, command=self._on_start_click)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="DURDUR", 
                                  bg=self.colors["red"], fg="white",
                                  font=("Segoe UI", 11, "bold"),
                                  width=10, command=self._on_stop_click, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        # Saat
        self.time_label = tk.Label(header, text="00:00:00",
                                  font=("Consolas", 14, "bold"),
                                  bg=self.colors["card"], fg=self.colors["text"])
        self.time_label.pack(side="right", padx=20)
    
    def _create_notebook(self):
        """Sekmeli ana icerik"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Sekmeler
        self._create_radar_tab()
        self._create_analysis_tab()
        self._create_backtest_tab()
        self._create_stats_tab()
    
    def _create_radar_tab(self):
        """RADAR sekmesi - Volatil coinler ve tarama"""
        tab = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.notebook.add(tab, text="  RADAR  ")
        
        # Iki panel
        left = tk.Frame(tab, bg=self.colors["card"], width=500)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5), pady=5)
        
        right = tk.Frame(tab, bg=self.colors["card"], width=500)
        right.pack(side="right", fill="both", expand=True, pady=5)
        
        # Sol: Volatil Coinler
        tk.Label(left, text="VOLATIL COINLER", font=("Segoe UI", 12, "bold"),
                bg=self.colors["card"], fg=self.colors["orange"]).pack(pady=10)
        
        cols = ("symbol", "price", "change", "volume")
        self.coins_tree = ttk.Treeview(left, columns=cols, show="headings", 
                                       style="Dark.Treeview", height=20)
        self.coins_tree.heading("symbol", text="Coin")
        self.coins_tree.heading("price", text="Fiyat")
        self.coins_tree.heading("change", text="1m %")
        self.coins_tree.heading("volume", text="Hacim")
        self.coins_tree.column("symbol", width=100)
        self.coins_tree.column("price", width=100, anchor="e")
        self.coins_tree.column("change", width=80, anchor="e")
        self.coins_tree.column("volume", width=120, anchor="e")
        self.coins_tree.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Sag: Izleme Listesi
        tk.Label(right, text="IZLEME LISTESI", font=("Segoe UI", 12, "bold"),
                bg=self.colors["card"], fg=self.colors["green"]).pack(pady=10)
        
        cols2 = ("symbol", "obi", "score", "status")
        self.watch_tree = ttk.Treeview(right, columns=cols2, show="headings",
                                       style="Dark.Treeview", height=20)
        self.watch_tree.heading("symbol", text="Coin")
        self.watch_tree.heading("obi", text="OBI")
        self.watch_tree.heading("score", text="Skor")
        self.watch_tree.heading("status", text="Durum")
        self.watch_tree.column("symbol", width=100)
        self.watch_tree.column("obi", width=80, anchor="e")
        self.watch_tree.column("score", width=80, anchor="e")
        self.watch_tree.column("status", width=100)
        self.watch_tree.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Gostergeler
        ind_frame = tk.Frame(right, bg=self.colors["accent"])
        ind_frame.pack(fill="x", padx=10, pady=10)
        
        self.ind_watched = tk.Label(ind_frame, text="Takip: 0", 
                                    font=("Consolas", 11, "bold"),
                                    bg=self.colors["accent"], fg=self.colors["text"])
        self.ind_watched.pack(side="left", padx=20, pady=10)
        
        self.ind_avg = tk.Label(ind_frame, text="Ort: 0", 
                               font=("Consolas", 11, "bold"),
                               bg=self.colors["accent"], fg=self.colors["text"])
        self.ind_avg.pack(side="left", padx=20, pady=10)
        
        self.ind_top = tk.Label(ind_frame, text="Top: 0", 
                               font=("Consolas", 11, "bold"),
                               bg=self.colors["accent"], fg=self.colors["green"])
        self.ind_top.pack(side="left", padx=20, pady=10)
    
    def _create_analysis_tab(self):
        """ANALIZ sekmesi - AI tahminleri"""
        tab = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.notebook.add(tab, text="  ANALIZ  ")
        
        # Sol: Sinyaller
        left = tk.Frame(tab, bg=self.colors["card"], width=400)
        left.pack(side="left", fill="both", padx=(0, 5), pady=5)
        left.pack_propagate(False)
        
        tk.Label(left, text="CANLI SINYALLER", font=("Segoe UI", 12, "bold"),
                bg=self.colors["card"], fg=self.colors["green"]).pack(pady=10)
        
        self.signal_list = tk.Text(left, bg=self.colors["bg"], fg=self.colors["text"],
                                   font=("Consolas", 10), height=30, width=45)
        self.signal_list.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Sag: AI Bilgi
        right = tk.Frame(tab, bg=self.colors["card"])
        right.pack(side="right", fill="both", expand=True, pady=5)
        
        tk.Label(right, text="AI DURUMU", font=("Segoe UI", 12, "bold"),
                bg=self.colors["card"], fg=self.colors["orange"]).pack(pady=10)
        
        # AI bilgileri
        info_frame = tk.Frame(right, bg=self.colors["accent"])
        info_frame.pack(fill="x", padx=20, pady=10)
        
        self.ai_status = tk.Label(info_frame, text="LightGBM: Yukleniyor...",
                                  font=("Segoe UI", 11),
                                  bg=self.colors["accent"], fg=self.colors["text"])
        self.ai_status.pack(anchor="w", padx=20, pady=5)
        
        self.ai_accuracy = tk.Label(info_frame, text="Yon Dogrulugu: --%",
                                    font=("Segoe UI", 11),
                                    bg=self.colors["accent"], fg=self.colors["text"])
        self.ai_accuracy.pack(anchor="w", padx=20, pady=5)
        
        self.ai_signals = tk.Label(info_frame, text="Toplam Sinyal: 0",
                                   font=("Segoe UI", 11),
                                   bg=self.colors["accent"], fg=self.colors["text"])
        self.ai_signals.pack(anchor="w", padx=20, pady=5)
        
        # Sinyal istatistikleri
        stats_frame = tk.Frame(right, bg=self.colors["card"])
        stats_frame.pack(fill="x", padx=20, pady=20)
        
        tk.Label(stats_frame, text="BUGUNUN OZETI", font=("Segoe UI", 11, "bold"),
                bg=self.colors["card"], fg=self.colors["text"]).pack(anchor="w")
        
        self.today_signals = tk.Label(stats_frame, text="Sinyal: 0 | Win: --%",
                                      font=("Consolas", 12),
                                      bg=self.colors["card"], fg=self.colors["green"])
        self.today_signals.pack(anchor="w", pady=5)
    
    def _create_backtest_tab(self):
        """BACKTEST sekmesi"""
        tab = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.notebook.add(tab, text="  BACKTEST  ")
        
        # Kontroller
        ctrl_frame = tk.Frame(tab, bg=self.colors["card"])
        ctrl_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(ctrl_frame, text="Gun:", font=("Segoe UI", 11),
                bg=self.colors["card"], fg=self.colors["text"]).pack(side="left", padx=10, pady=15)
        
        self.days_var = tk.StringVar(value="3")
        days_entry = tk.Entry(ctrl_frame, textvariable=self.days_var, width=5,
                             font=("Consolas", 12))
        days_entry.pack(side="left", padx=5)
        
        tk.Label(ctrl_frame, text="Coin:", font=("Segoe UI", 11),
                bg=self.colors["card"], fg=self.colors["text"]).pack(side="left", padx=10)
        
        self.coins_var = tk.StringVar(value="5")
        coins_entry = tk.Entry(ctrl_frame, textvariable=self.coins_var, width=5,
                              font=("Consolas", 12))
        coins_entry.pack(side="left", padx=5)
        
        self.backtest_btn = tk.Button(ctrl_frame, text="BACKTEST BASLAT",
                                      bg=self.colors["orange"], fg="white",
                                      font=("Segoe UI", 11, "bold"),
                                      command=self._run_backtest)
        self.backtest_btn.pack(side="left", padx=20, pady=10)
        
        # Sonuclar
        result_frame = tk.Frame(tab, bg=self.colors["card"])
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        tk.Label(result_frame, text="BACKTEST SONUCLARI", font=("Segoe UI", 12, "bold"),
                bg=self.colors["card"], fg=self.colors["orange"]).pack(pady=10)
        
        self.backtest_result = tk.Text(result_frame, bg=self.colors["bg"], 
                                       fg=self.colors["text"],
                                       font=("Consolas", 11), height=25)
        self.backtest_result.pack(fill="both", expand=True, padx=10, pady=5)
        self.backtest_result.insert("1.0", "Backtest calistirmak icin BACKTEST BASLAT'a basin...")
    
    def _create_stats_tab(self):
        """ISTATISTIK sekmesi"""
        tab = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.notebook.add(tab, text="  ISTATISTIK  ")
        
        # Sinyal gecmisi
        left = tk.Frame(tab, bg=self.colors["card"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 5), pady=5)
        
        tk.Label(left, text="SINYAL GECMISI", font=("Segoe UI", 12, "bold"),
                bg=self.colors["card"], fg=self.colors["green"]).pack(pady=10)
        
        cols = ("time", "symbol", "dir", "entry", "pnl", "status")
        self.history_tree = ttk.Treeview(left, columns=cols, show="headings",
                                         style="Dark.Treeview", height=20)
        self.history_tree.heading("time", text="Zaman")
        self.history_tree.heading("symbol", text="Coin")
        self.history_tree.heading("dir", text="Yon")
        self.history_tree.heading("entry", text="Giris")
        self.history_tree.heading("pnl", text="PnL")
        self.history_tree.heading("status", text="Durum")
        self.history_tree.column("time", width=80)
        self.history_tree.column("symbol", width=100)
        self.history_tree.column("dir", width=60)
        self.history_tree.column("entry", width=100, anchor="e")
        self.history_tree.column("pnl", width=80, anchor="e")
        self.history_tree.column("status", width=80)
        self.history_tree.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Performans ozeti
        right = tk.Frame(tab, bg=self.colors["card"], width=350)
        right.pack(side="right", fill="y", pady=5)
        right.pack_propagate(False)
        
        tk.Label(right, text="PERFORMANS", font=("Segoe UI", 12, "bold"),
                bg=self.colors["card"], fg=self.colors["orange"]).pack(pady=10)
        
        perf_frame = tk.Frame(right, bg=self.colors["accent"])
        perf_frame.pack(fill="x", padx=10, pady=5)
        
        stats = [
            ("Toplam Sinyal:", "0"),
            ("Win Rate:", "--%"),
            ("Toplam PnL:", "+0.00%"),
            ("Profit Factor:", "--"),
            ("Max Drawdown:", "--%")
        ]
        
        self.perf_labels = {}
        for label, value in stats:
            row = tk.Frame(perf_frame, bg=self.colors["accent"])
            row.pack(fill="x", padx=10, pady=5)
            
            tk.Label(row, text=label, font=("Segoe UI", 11),
                    bg=self.colors["accent"], fg=self.colors["muted"]).pack(side="left")
            
            val_label = tk.Label(row, text=value, font=("Consolas", 12, "bold"),
                                bg=self.colors["accent"], fg=self.colors["text"])
            val_label.pack(side="right")
            self.perf_labels[label] = val_label
        
        # Yenile butonu
        tk.Button(right, text="ISTATISTIKLERI GUNCELLE",
                 bg=self.colors["accent"], fg=self.colors["text"],
                 font=("Segoe UI", 10),
                 command=self._refresh_stats).pack(pady=20)
    
    def _create_status_bar(self):
        """Alt durum cubugu"""
        status = tk.Frame(self.root, bg=self.colors["card"], height=30)
        status.pack(fill="x", padx=5, pady=5)
        status.pack_propagate(False)
        
        self.status_label = tk.Label(status, text="Hazir",
                                    font=("Segoe UI", 10),
                                    bg=self.colors["card"], fg=self.colors["muted"])
        self.status_label.pack(side="left", padx=10, pady=5)
        
        self.db_label = tk.Label(status, text="DB: Baglaniyor...",
                                font=("Segoe UI", 10),
                                bg=self.colors["card"], fg=self.colors["muted"])
        self.db_label.pack(side="right", padx=10, pady=5)
    
    def _update_time(self):
        """Saati guncelle"""
        self.time_label.config(text=datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._update_time)
    
    def _on_start_click(self):
        """Baslat butonu"""
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Calisiyor...", fg=self.colors["green"])
        
        if self.on_start:
            self.on_start()
    
    def _on_stop_click(self):
        """Durdur butonu"""
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Durduruldu", fg=self.colors["orange"])
        
        if self.on_stop:
            self.on_stop()
    
    def _run_backtest(self):
        """Backtest calistir"""
        self.backtest_result.delete("1.0", "end")
        self.backtest_result.insert("1.0", "Backtest baslatiliyor...\n")
        self.backtest_btn.config(state="disabled")
        
        days = int(self.days_var.get() or 3)
        coins = int(self.coins_var.get() or 5)
        
        def run():
            try:
                import asyncio
                from core.backtester import run_smart_backtest
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                metrics = loop.run_until_complete(
                    run_smart_backtest(days=days, coin_count=coins, optimize=False, train_ml=True)
                )
                
                result = f"""
BACKTEST TAMAMLANDI
{'='*50}

Toplam Sinyal: {metrics.total_signals}
Dogru Tahmin:  {metrics.correct_predictions}
Win Rate:      {metrics.win_rate:.1f}%

Toplam PnL:    {metrics.total_pnl:+.2f}%
Ortalama PnL:  {metrics.avg_pnl_per_trade:+.3f}%
Max Kar:       {metrics.max_profit:+.2f}%
Max Zarar:     {metrics.max_loss:+.2f}%

Sharpe Ratio:  {metrics.sharpe_ratio:.2f}
Max Drawdown:  {metrics.max_drawdown:.2f}%
Profit Factor: {metrics.profit_factor:.2f}

UP Sinyalleri:   {metrics.up_signals} (Win: {metrics.up_win_rate:.1f}%)
DOWN Sinyalleri: {metrics.down_signals} (Win: {metrics.down_win_rate:.1f}%)

{'='*50}
ML modeli egitildi ve kaydedildi!
"""
                self.root.after(0, lambda: self._update_backtest_result(result))
                
            except Exception as e:
                self.root.after(0, lambda: self._update_backtest_result(f"Hata: {e}"))
            finally:
                self.root.after(0, lambda: self.backtest_btn.config(state="normal"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _update_backtest_result(self, text: str):
        """Backtest sonucunu guncelle"""
        self.backtest_result.delete("1.0", "end")
        self.backtest_result.insert("1.0", text)
    
    def _refresh_stats(self):
        """Istatistikleri guncelle"""
        try:
            from core.database import Database
            db = Database()
            stats = db.get_stats()
            
            self.perf_labels["Toplam Sinyal:"].config(text=str(stats.get("signals", 0)))
            self.db_label.config(text=f"DB: {stats.get('signals', 0)} sinyal", 
                                fg=self.colors["green"])
        except Exception as e:
            print(f"Stats hatasi: {e}")
    
    # === PUBLIC API ===
    
    def update_coins_table(self, coins: list):
        """Volatil coinler tablosu"""
        for item in self.coins_tree.get_children():
            self.coins_tree.delete(item)
        
        for coin in coins:
            symbol, price, change, volume = coin
            
            # Renk belirleme
            tag = "green" if change > 0 else "red"
            
            self.coins_tree.insert("", "end", values=(
                symbol,
                f"${price:,.4f}" if price < 1 else f"${price:,.2f}",
                f"{change:+.2f}%",
                f"${volume/1e6:.0f}M"
            ), tags=(tag,))
        
        self.coins_tree.tag_configure("green", foreground=self.colors["green"])
        self.coins_tree.tag_configure("red", foreground=self.colors["red"])
    
    def update_watchlist(self, watchlist: list):
        """Izleme listesi tablosu"""
        for item in self.watch_tree.get_children():
            self.watch_tree.delete(item)
        
        for item in watchlist:
            symbol, obi, score, elapsed = item
            
            # Durum belirleme
            if score >= 55:
                status = "SINYAL"
                tag = "hot"
            elif score >= 40:
                status = "Hazir"
                tag = "warm"
            else:
                status = "Izleniyor"
                tag = "normal"
            
            self.watch_tree.insert("", "end", values=(
                symbol,
                f"{obi:.2f}" if isinstance(obi, float) else str(obi),
                f"{score:.0f}",
                status
            ), tags=(tag,))
        
        self.watch_tree.tag_configure("hot", foreground=self.colors["green"])
        self.watch_tree.tag_configure("warm", foreground=self.colors["orange"])
    
    def update_indicators(self, watched: int, avg_score: float, top_score: float):
        """Gostergeleri guncelle"""
        self.ind_watched.config(text=f"Takip: {watched}")
        self.ind_avg.config(text=f"Ort: {avg_score:.0f}")
        self.ind_top.config(text=f"Top: {top_score:.0f}")
    
    def add_signal(self, symbol: str, direction: str, price: float, 
                   score: float, reasons: list = None):
        """Sinyal ekle"""
        self.signal_count += 1
        
        icon = "LONG" if direction == "up" else "SHORT"
        color_tag = "green" if direction == "up" else "red"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        text = f"""
[{timestamp}] #{self.signal_count}
{icon} {symbol} @ ${price:,.4f}
Skor: {score:.0f}/100
"""
        if reasons:
            for r in reasons[:3]:
                text += f"  - {r}\n"
        text += "-" * 40 + "\n"
        
        self.signal_list.insert("1.0", text)
        self.ai_signals.config(text=f"Toplam Sinyal: {self.signal_count}")
    
    def update_status(self, message: str):
        """Durum guncelle"""
        self.status_label.config(text=message)
    
    def update_ai_status(self, is_trained: bool, accuracy: float = 0):
        """AI durumunu guncelle"""
        if is_trained:
            self.ai_status.config(text="LightGBM: Aktif", fg=self.colors["green"])
            self.ai_accuracy.config(text=f"Yon Dogrulugu: {accuracy:.1f}%")
        else:
            self.ai_status.config(text="LightGBM: Egitilmedi", fg=self.colors["orange"])
    
    def run(self):
        """Uygulamayi baslat"""
        self._refresh_stats()
        self.root.mainloop()


if __name__ == "__main__":
    app = TradingApp()
    app.run()
