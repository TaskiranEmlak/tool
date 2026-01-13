# Keşfedilen Strateji Backtest Scripti
"""
config/discovered_strategy.json'daki stratejiyi test eder
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json

async def fetch_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Veri çek"""
    base_url = "https://fapi.binance.com"
    all_klines = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    async with aiohttp.ClientSession() as session:
        current = start_time
        while current < end_time:
            url = f"{base_url}/fapi/v1/klines"
            params = {"symbol": symbol, "interval": "5m", "startTime": current, "limit": 1500}
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not data: break
                        all_klines.extend(data)
                        current = data[-1][0] + 1
                    else: break
            except: break
    
    if not all_klines: return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """İndikatörler ekle"""
    df = df.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))
    
    # EMA
    df['ema_fast'] = df['close'].ewm(span=9).mean()
    df['ema_slow'] = df['close'].ewm(span=50).mean()
    
    # Bollinger
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_pct'] = (df['close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 0.0001)
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Momentum
    df['momentum'] = df['close'].pct_change(5) * 100
    
    return df.dropna()

def apply_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Stratejiyi uygula"""
    df = df.copy()
    
    # LONG koşulları
    long_mask = pd.Series([True] * len(df), index=df.index)
    
    if params.get('use_rsi', False):
        if params.get('rsi_dir') == 'low':
            long_mask &= (df['rsi'] < params['rsi_val'])
        else:
            long_mask &= (df['rsi'] > params['rsi_val'])
    
    if params.get('use_ema', False):
        if params.get('ema_dir') == 'above':
            long_mask &= (df['ema_fast'] > df['ema_slow'])
        else:
            long_mask &= (df['ema_fast'] < df['ema_slow'])
    
    if params.get('use_bb', False):
        if params.get('bb_dir') == 'low':
            long_mask &= (df['bb_pct'] < params['bb_val'])
        else:
            long_mask &= (df['bb_pct'] > params['bb_val'])
    
    if params.get('use_mom', False):
        if params.get('mom_dir') == 'up':
            long_mask &= (df['momentum'] > params['mom_val'])
        else:
            long_mask &= (df['momentum'] < -params['mom_val'])
    
    df['signal'] = 0
    df.loc[long_mask, 'signal'] = 1
    
    return df

def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    """Backtest çalıştır"""
    COMMISSION = 0.001
    
    df = apply_strategy(df, params)
    signals = df[df['signal'] == 1]
    
    if len(signals) < 1:
        return {'trades': 0, 'pnl': 0, 'win_rate': 0, 'wins': 0, 'losses': 0}
    
    trades = []
    total_pnl = 0
    wins = 0
    losses = 0
    
    sl_mult = params.get('sl_mult', 1.5)
    tp_mult = params.get('tp_mult', 4.5)
    
    for idx, row in signals.iterrows():
        entry = row['close']
        atr = row['atr']
        entry_time = row['timestamp']
        
        sl_price = entry - (atr * sl_mult)
        tp_price = entry + (atr * tp_mult)
        
        # Gelecek 60 mumu kontrol et
        future_idx = df.index.get_loc(idx)
        future = df.iloc[future_idx+1:future_idx+61]
        
        if len(future) < 10:
            continue
        
        exit_price = None
        exit_type = None
        exit_time = None
        
        for _, frow in future.iterrows():
            if frow['low'] <= sl_price:
                exit_price = sl_price
                exit_type = 'SL'
                exit_time = frow['timestamp']
                break
            elif frow['high'] >= tp_price:
                exit_price = tp_price
                exit_type = 'TP'
                exit_time = frow['timestamp']
                break
        
        if exit_price is None:
            exit_price = future.iloc[-1]['close']
            exit_type = 'TIME'
            exit_time = future.iloc[-1]['timestamp']
        
        pnl = ((exit_price - entry) / entry) - COMMISSION
        total_pnl += pnl
        
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry': entry,
            'exit': exit_price,
            'type': exit_type,
            'pnl': pnl * 100
        })
    
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    return {
        'trades': len(trades),
        'pnl': total_pnl * 100,
        'win_rate': win_rate * 100,
        'wins': wins,
        'losses': losses,
        'details': trades
    }

async def main():
    print("=" * 60)
    print("     📊 KEŞFEDİLEN STRATEJİ BACKTEST")
    print("=" * 60)
    
    # Stratejiyi yükle
    try:
        with open("config/discovered_strategy.json", "r") as f:
            params = json.load(f)
        print("\n✅ Strateji yüklendi: config/discovered_strategy.json")
    except:
        print("\n❌ Strateji dosyası bulunamadı!")
        print("   Önce strategy_optimizer.py çalıştırın.")
        return
    
    # Aktif koşulları göster
    print("\n🎯 TEST EDİLECEK STRATEJİ:")
    print("-" * 40)
    conditions = []
    if params.get('use_rsi'): conditions.append(f"RSI {params['rsi_dir']} {params['rsi_val']}")
    if params.get('use_ema'): conditions.append(f"EMA {params['ema_dir']}")
    if params.get('use_bb'): conditions.append(f"BB {params['bb_dir']} {params['bb_val']:.2f}")
    if params.get('use_mom'): conditions.append(f"Momentum {params['mom_dir']} {params['mom_val']:.2f}")
    print(f"   Koşullar: {' & '.join(conditions)}")
    print(f"   Stop Loss: {params.get('sl_mult', 1.5):.2f}x ATR")
    print(f"   Take Profit: {params.get('tp_mult', 4.5):.2f}x ATR")
    print("-" * 40)
    
    # Coinler
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT']
    days = 30
    
    print(f"\n📥 {len(symbols)} coin için {days} günlük veri çekiliyor...\n")
    
    all_results = []
    total_trades = 0
    total_pnl = 0
    total_wins = 0
    total_losses = 0
    
    for symbol in symbols:
        df = await fetch_data(symbol, days)
        if df.empty:
            print(f"   ❌ {symbol}: Veri yok")
            continue
        
        df = add_indicators(df)
        result = run_backtest(df, params)
        
        total_trades += result['trades']
        total_pnl += result['pnl']
        total_wins += result['wins']
        total_losses += result['losses']
        
        status = "✅" if result['pnl'] > 0 else "❌"
        print(f"   {status} {symbol}: {result['trades']} işlem, PnL: {result['pnl']:+.2f}%, WR: {result['win_rate']:.1f}%")
        
        all_results.append({'symbol': symbol, **result})
    
    # Özet
    overall_wr = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
    
    print("\n" + "=" * 60)
    print("     📈 BACKTEST SONUÇLARI")
    print("=" * 60)
    print(f"\n   Toplam İşlem: {total_trades}")
    print(f"   Kazanan: {total_wins} | Kaybeden: {total_losses}")
    print(f"   Win Rate: {overall_wr:.1f}%")
    print(f"   Toplam PnL: {total_pnl:+.2f}%")
    
    if total_pnl > 0:
        print(f"\n   🎉 STRATEJİ KARLI!")
    else:
        print(f"\n   ⚠️ Strateji bu dönemde zarar etti.")
    
    print("=" * 60)
    
    # En iyi/kötü işlemler
    all_trades = []
    for r in all_results:
        for t in r.get('details', []):
            t['symbol'] = r['symbol']
            all_trades.append(t)
    
    if all_trades:
        sorted_trades = sorted(all_trades, key=lambda x: x['pnl'], reverse=True)
        
        print("\n🏆 EN İYİ 3 İŞLEM:")
        for t in sorted_trades[:3]:
            print(f"   {t['symbol']}: {t['pnl']:+.2f}% ({t['type']})")
        
        print("\n💀 EN KÖTÜ 3 İŞLEM:")
        for t in sorted_trades[-3:]:
            print(f"   {t['symbol']}: {t['pnl']:+.2f}% ({t['type']})")


if __name__ == "__main__":
    asyncio.run(main())
