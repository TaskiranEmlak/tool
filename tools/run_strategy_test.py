# Proven Strategy Test - Backtest + Live Sinyal
"""
Stratejiyi backtest et ve canlı sinyal üret
"""

import asyncio
import sys
import os

# Path fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.proven_strategy import get_strategy
from core.signal_generator import get_signal_generator


async def run_backtest():
    """Backtest çalıştır"""
    print("=" * 60)
    print("PROVEN STRATEGY BACKTEST")
    print("=" * 60)
    
    gen = get_signal_generator()
    strategy = get_strategy()
    
    # En volatil coinleri bul
    print("\n[1] En volatil coinler taranıyor...")
    await gen.data_fetcher.update_coin_list(limit=20)
    
    symbols = gen.data_fetcher.SYMBOLS[:10]  # En volatil 10 coin
    
    total_results = {
        "total_trades": 0,
        "total_wins": 0,
        "total_pnl": 0
    }
    
    for symbol in symbols:
        print(f"\n[{symbol}] Veri çekiliyor...")
        df = await gen.data_fetcher.fetch_klines(symbol, "1h", 500)  # ~20 gün
        
        if df.empty:
            print(f"  Veri çekilemedi!")
            continue
        
        print(f"  {len(df)} mum alındı")
        
        # Backtest
        results = strategy.backtest(df)
        
        if "error" in results:
            print(f"  Hata: {results.get('error')}")
            continue
        
        print(f"\n  === {symbol} SONUÇLARI ===")
        print(f"  Toplam Trade: {results['total_trades']}")
        print(f"  Kazanan: {results['winning_trades']} | Kaybeden: {results['losing_trades']}")
        print(f"  Win Rate: {results['win_rate']}%")
        print(f"  Toplam Return: {results['total_return_pct']}%")
        print(f"  Max Drawdown: {results['max_drawdown']}%")
        print(f"  Profit Factor: {results['profit_factor']}")
        
        # Toplam
        total_results["total_trades"] += results["total_trades"]
        total_results["total_wins"] += results["winning_trades"]
        total_results["total_pnl"] += results["total_return_pct"]
    
    await gen.data_fetcher.close()
    
    # Genel özet
    print("\n" + "=" * 60)
    print("GENEL ÖZET")
    print("=" * 60)
    
    if total_results["total_trades"] > 0:
        overall_winrate = total_results["total_wins"] / total_results["total_trades"] * 100
        print(f"Toplam Trade: {total_results['total_trades']}")
        print(f"Genel Win Rate: {overall_winrate:.1f}%")
        print(f"Ortalama Return/Coin: {total_results['total_pnl'] / len(symbols):.2f}%")
    else:
        print("Hiç trade bulunamadı!")


async def run_live_signals():
    """Canlı sinyal üret"""
    print("\n" + "=" * 60)
    print("CANLI SİNYAL TARAMASI")
    print("=" * 60)
    
    gen = get_signal_generator()
    signals = await gen.generate_signals("1h")
    
    if signals:
        print(f"\n{len(signals)} SİNYAL BULUNDU:\n")
        for s in signals:
            print(f"{'🟢' if s.signal_type.value == 'LONG' else '🔴'} {s.signal_type.value} {s.symbol}")
            print(f"   Entry: ${s.entry_price:.2f}")
            print(f"   Stop Loss: ${s.stop_loss:.2f}")
            print(f"   Take Profit: ${s.take_profit:.2f}")
            print(f"   Güven: {s.confidence}%")
            print(f"   Sebep: {s.reason}")
            print()
    else:
        print("\nŞu an sinyal yok. Piyasa koşulları uygun değil.")
        print("Bu normal - strateji sadece güçlü fırsatlarda sinyal veriyor.")
    
    await gen.data_fetcher.close()


async def main():
    print("\n" + "🚀" * 20)
    print("\nPROVEN TRADING STRATEGY TEST")
    print("\n" + "🚀" * 20)
    
    # 1. Backtest
    await run_backtest()
    
    # 2. Canlı sinyal
    await run_live_signals()
    
    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
