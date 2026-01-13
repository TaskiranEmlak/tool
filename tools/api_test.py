# Scraper Direkt Test
"""
Bu scripti calistirarak Binance API'nin calisip calismadigini test et.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scraper import BinanceScraper

async def main():
    print("=" * 50)
    print("BINANCE LEADERBOARD API TESTI")
    print("=" * 50)
    
    scraper = BinanceScraper()
    
    # 1. Leaderboard test
    print("\n[1] Leaderboard cekiliyor...")
    traders = await scraper.fetch_leaderboard(limit=5)
    
    if traders:
        print(f"    BASARILI! {len(traders)} trader bulundu:")
        for t in traders:
            print(f"    - {t['nickname']}: ROI={t['roi']:.1f}%, Takipci={t['follower_count']}")
        
        # 2. Pozisyon test
        print(f"\n[2] Ilk trader'in pozisyonlari cekiliyor...")
        uid = traders[0]['uid']
        positions = await scraper.fetch_trader_positions(uid)
        
        if positions:
            print(f"    BASARILI! {len(positions)} pozisyon bulundu:")
            for p in positions:
                print(f"    - {p['direction']} {p['symbol']}: PnL={p['pnl_percent']:.2f}%")
        else:
            print("    Acik pozisyon yok (normal olabilir)")
    else:
        print("    BASARISIZ! Trader bulunamadi.")
        print("    Binance API erisimi engellenmi olabilir.")
        print("    VPN kullanmayi dene.")
    
    await scraper._close()
    print("\n" + "=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
