# Walk-Forward Strategy Optimizer
"""
OVERFITTING'I ÖNLEYEN OPTİMİZASYON
Veriyi TRAIN ve TEST olarak ayırır.
Strateji TRAIN'de bulunur, TEST'te doğrulanır.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json

class WalkForwardOptimizer:
    def __init__(self, train_days: int = 30, test_days: int = 14):
        self.train_days = train_days  # Eğitim için
        self.test_days = test_days    # Test için
        self.base_url = "https://fapi.binance.com"
        self.train_data = {}
        self.test_data = {}
        self.trial_count = 0
    
    async def fetch_data(self, symbol: str) -> pd.DataFrame:
        total_days = self.train_days + self.test_days
        all_klines = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=total_days)).timestamp() * 1000)
        
        async with aiohttp.ClientSession() as session:
            current = start_time
            while current < end_time:
                url = f"{self.base_url}/fapi/v1/klines"
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
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))
        
        # EMA
        df['ema_fast'] = df['close'].ewm(span=9).mean()
        df['ema_slow'] = df['close'].ewm(span=50).mean()
        
        # BB
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
        
        # Stoch
        low14 = df['low'].rolling(14).min()
        high14 = df['high'].rolling(14).max()
        df['stoch'] = 100 * (df['close'] - low14) / (high14 - low14 + 0.0001)
        
        # Future
        df['future_high'] = df['high'].shift(-12).rolling(12).max()
        df['future_low'] = df['low'].shift(-12).rolling(12).min()
        df['future_ret'] = df['close'].shift(-12) / df['close'] - 1
        
        return df.dropna()
    
    async def prepare(self):
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT']
        print(f"\n📥 {len(symbols)} coin için veri çekiliyor...")
        print(f"   Train: {self.train_days} gün | Test: {self.test_days} gün")
        
        for s in symbols:
            df = await self.fetch_data(s)
            if df.empty: continue
            
            df = self.add_indicators(df)
            if len(df) < 500: continue
            
            # Veriyi böl
            test_start = datetime.now() - timedelta(days=self.test_days)
            
            train_df = df[df['timestamp'] < test_start]
            test_df = df[df['timestamp'] >= test_start]
            
            if len(train_df) > 100 and len(test_df) > 100:
                self.train_data[s] = train_df
                self.test_data[s] = test_df
                print(f"   ✓ {s}: Train={len(train_df)}, Test={len(test_df)}")
    
    def backtest(self, df: pd.DataFrame, params: dict) -> tuple:
        """Basit, sağlam backtest"""
        COMM = 0.001
        
        # Çok basit koşullar (overfitting'i azaltmak için)
        # LONG: RSI düşük VE trend yukarı
        long_mask = (
            (df['rsi'] < params['rsi_buy']) &
            (df['ema_fast'] > df['ema_slow'] if params['trend_filter'] else True) &
            (df['stoch'] < params['stoch_buy'])
        )
        
        # SHORT: RSI yüksek VE trend aşağı
        short_mask = (
            (df['rsi'] > params['rsi_sell']) &
            (df['ema_fast'] < df['ema_slow'] if params['trend_filter'] else True) &
            (df['stoch'] > params['stoch_sell'])
        )
        
        df = df.copy()
        df['signal'] = 0
        df.loc[long_mask, 'signal'] = 1
        df.loc[short_mask, 'signal'] = -1
        
        signals = df[df['signal'] != 0]
        if len(signals) < 3:
            return -100, 0, 0
        
        pnl = 0
        wins = 0
        trades = 0
        
        for _, row in signals.iterrows():
            d = row['signal']
            entry = row['close']
            atr = row['atr']
            
            sl = atr * params['sl_mult']
            tp = atr * params['tp_mult']
            
            fh = row['future_high']
            fl = row['future_low']
            fr = row['future_ret']
            
            if pd.isna(fh): continue
            
            if d == 1:  # LONG
                if fl <= entry - sl:
                    p = -sl/entry - COMM
                elif fh >= entry + tp:
                    p = tp/entry - COMM
                    wins += 1
                else:
                    p = fr - COMM
                    if p > 0: wins += 1
            else:  # SHORT
                if fh >= entry + sl:
                    p = -sl/entry - COMM
                elif fl <= entry - tp:
                    p = tp/entry - COMM
                    wins += 1
                else:
                    p = -fr - COMM
                    if p > 0: wins += 1
            
            pnl += p
            trades += 1
        
        wr = wins/trades if trades > 0 else 0
        return pnl, wr, trades
    
    def objective(self, trial):
        self.trial_count += 1
        
        # Daha basit, daha az parametre (overfitting önleme)
        params = {
            'rsi_buy': trial.suggest_int('rsi_buy', 25, 40),
            'rsi_sell': trial.suggest_int('rsi_sell', 60, 75),
            'stoch_buy': trial.suggest_int('stoch_buy', 15, 35),
            'stoch_sell': trial.suggest_int('stoch_sell', 65, 85),
            'trend_filter': trial.suggest_categorical('trend_filter', [True, False]),
            'sl_mult': trial.suggest_float('sl_mult', 1.0, 2.0),
            'tp_mult': trial.suggest_float('tp_mult', 1.5, 3.5),
        }
        
        if params['tp_mult'] < params['sl_mult']:
            return -999
        
        # SADECE TRAIN VERİSİNDE TEST ET
        total_pnl = 0
        total_wr = 0
        count = 0
        
        for symbol, df in self.train_data.items():
            try:
                pnl, wr, trades = self.backtest(df, params)
                if trades >= 5:
                    total_pnl += pnl
                    total_wr += wr
                    count += 1
            except:
                continue
        
        if count == 0:
            return -100
        
        avg_wr = total_wr / count
        score = total_pnl
        if avg_wr > 0.55: score *= 1.3
        if avg_wr < 0.45: score *= 0.5
        
        if self.trial_count % 50 == 0:
            print(f"  [{self.trial_count}] Train PnL={total_pnl:.3f}, WR={avg_wr:.1%}")
        
        return score
    
    def validate(self, params: dict):
        """TEST verisinde doğrula (out-of-sample)"""
        print("\n" + "=" * 60)
        print("     📊 OUT-OF-SAMPLE TEST (GERÇEK DOĞRULAMA)")
        print("=" * 60)
        
        total_pnl = 0
        total_wr = 0
        total_trades = 0
        count = 0
        
        for symbol, df in self.test_data.items():
            try:
                pnl, wr, trades = self.backtest(df, params)
                if trades >= 1:
                    status = "✅" if pnl > 0 else "❌"
                    print(f"   {status} {symbol}: {trades} işlem, PnL={pnl*100:+.2f}%, WR={wr:.1%}")
                    total_pnl += pnl
                    total_wr += wr
                    total_trades += trades
                    count += 1
            except:
                continue
        
        if count > 0:
            avg_wr = total_wr / count
            print("\n" + "-" * 40)
            print(f"   TOPLAM: {total_trades} işlem")
            print(f"   ORTALAMA WIN RATE: {avg_wr:.1%}")
            print(f"   TOPLAM PnL: {total_pnl*100:+.2f}%")
            
            if total_pnl > 0:
                print("\n   🎉 STRATEJİ OUT-OF-SAMPLE TESTİNİ GEÇTİ!")
            else:
                print("\n   ⚠️ Strateji test verisinde zarar etti (Overfitting olabilir)")
    
    async def run(self, n_trials: int = 200):
        print("=" * 60)
        print("     🔬 WALK-FORWARD OPTİMİZASYON")
        print("     (Overfitting'i Önleyen Doğru Yöntem)")
        print("=" * 60)
        
        await self.prepare()
        
        if not self.train_data:
            print("HATA: Veri yok!")
            return
        
        print(f"\n🚀 {n_trials} deneme başlıyor (SADECE TRAIN VERİSİNDE)...\n")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print("\n" + "=" * 60)
        print("     ✅ TRAIN OPTİMİZASYONU TAMAMLANDI!")
        print("=" * 60)
        
        best = study.best_params
        print(f"\n📊 Train'de En İyi Skor: {study.best_value:.4f}")
        print("\n🎯 BULUNAN STRATEJİ:")
        print("-" * 40)
        print(f"   RSI Alış: < {best['rsi_buy']}")
        print(f"   RSI Satış: > {best['rsi_sell']}")
        print(f"   Stoch Alış: < {best['stoch_buy']}")
        print(f"   Stoch Satış: > {best['stoch_sell']}")
        print(f"   Trend Filtresi: {'Açık' if best['trend_filter'] else 'Kapalı'}")
        print(f"   Stop Loss: {best['sl_mult']:.2f}x ATR")
        print(f"   Take Profit: {best['tp_mult']:.2f}x ATR")
        print("-" * 40)
        
        # OUT-OF-SAMPLE TEST
        self.validate(best)
        
        # Kaydet
        os.makedirs("config", exist_ok=True)
        with open("config/validated_strategy.json", "w") as f:
            json.dump(best, f, indent=4)
        print(f"\n✅ Strateji: config/validated_strategy.json")
        
        return best


async def main():
    print("\n🔬 Walk-Forward Optimizer Başlatılıyor...\n")
    opt = WalkForwardOptimizer(train_days=30, test_days=14)
    await opt.run(n_trials=200)


if __name__ == "__main__":
    asyncio.run(main())
