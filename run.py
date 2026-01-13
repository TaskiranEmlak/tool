# Copy Trading System - Ana Baslatici
"""
Web dashboard ve scraper'i birlikte baslatir.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import threading
import webbrowser
from pathlib import Path


def print_banner():
    """Baslangic banner'i"""
    print("""
============================================================
                                                           
     COPY TRADING SYSTEM                                   
     Binance Leaderboard Trader Takip Botu                 
                                                           
============================================================
    """)


def ensure_directories():
    """Gerekli klasorleri olustur"""
    dirs = [
        "data",
        "config",
        "web/static/css",
        "web/static/sounds"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def start_dashboard():
    """Web dashboard'u baslat"""
    from web.app import run_dashboard
    run_dashboard(host="127.0.0.1", port=8000)


def main():
    print_banner()
    ensure_directories()
    
    print("Dashboard baslatiliyor...")
    print("http://127.0.0.1:8000")
    print()
    print("Cikmak icin Ctrl+C")
    print("-" * 40)
    
    # Tarayiciyi ac
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except:
        pass
    
    # Dashboard'u baslat
    start_dashboard()


if __name__ == "__main__":
    main()
