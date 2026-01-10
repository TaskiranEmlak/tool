# HFT Trading Tools - Backtest Engine
"""
Geçmiş veriler üzerinde strateji testi.
Binance'ten veri indirir ve mevcut sistemi simüle eder.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class Trade:
    """Bir işlem kaydı"""
    entry_time: str
    exit_time: str
    symbol: str
    direction: str      # "long" or "short"
    entry_price: float
    exit_price: float
    score: float        # Giriş skoru
    profit_percent: float
    profit_usd: float


@dataclass
class BacktestResult:
    """Backtest sonucu"""
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit_percent: float
    max_drawdown: float
    best_trade: float
    worst_trade: float
    avg_profit: float
    trades: List[Trade]


class Backtester:
    """
    Strateji backtest motoru.
    """
    
    def __init__(self, initial_balance: float = 1000):
        self.initial_balance = initial_balance
        self.base_url = "https://fapi.binance.com"
    
    async def download_historical_data(self, symbol: str, interval: str = "1m", 
                                       days: int = 7) -> pd.DataFrame:
        """
        Binance'ten geçmiş veri indir.
        
        Args:
            symbol: Trading çifti (örn: BTCUSDT)
            interval: Zaman dilimi (1m, 5m, 15m, 1h)
            days: Kaç günlük veri
        """
        print(f"[Backtest] {symbol} icin {days} gunluk veri indiriliyor...")
        
        all_klines = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        async with aiohttp.ClientSession() as session:
            current_start = start_time
            
            while current_start < end_time:
                url = f"{self.base_url}/fapi/v1/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "limit": 1000
                }
                
                async with session.get(url, params=params) as resp:
                    klines = await resp.json()
                    
                    if not klines:
                        break
                    
                    all_klines.extend(klines)
                    current_start = klines[-1][0] + 1  # Sonraki batch
                    
                    await asyncio.sleep(0.1)  # Rate limit
        
        # DataFrame'e çevir
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['quote_volume'] = df['quote_volume'].astype(float)
        
        print(f"[Backtest] {len(df)} mum indirildi")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """İndikatörleri hesapla"""
        # Fiyat değişimleri
        df['change_1m'] = df['close'].pct_change() * 100
        df['change_5m'] = df['close'].pct_change(5) * 100
        df['change_15m'] = df['close'].pct_change(15) * 100
        
        # Hacim ortalaması ve oranı
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum (RSI benzeri)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Trend (SMA)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['trend'] = (df['close'] > df['sma_20']).astype(int)  # 1=uptrend, 0=downtrend
        
        # Volatilite (ATR)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df['volatility'] = (df['atr'] / df['close']) * 100
        
        # Basitleştirilmiş OBI (Price Position)
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001) * 100
        
        return df
    
    def calculate_score(self, row) -> Tuple[float, str]:
        """
        Mevcut sisteme benzer skor hesapla.
        
        Returns:
            (skor, yon)
        """
        score = 0
        
        # 1. Hacim skoru (0-25)
        if row['volume_ratio'] > 2:
            score += 25
        elif row['volume_ratio'] > 1.5:
            score += 15
        elif row['volume_ratio'] > 1:
            score += 5
        
        # 2. Momentum skoru (0-25)
        if row['rsi'] > 70:
            score += 20  # Overbought - kısa vadede düşüş olabilir ama momentum var
        elif row['rsi'] > 60:
            score += 25
        elif row['rsi'] < 30:
            score += 20  # Oversold - short için
        elif row['rsi'] < 40:
            score += 15
        
        # 3. Trend skoru (0-20)
        if row['trend'] == 1 and row['change_5m'] > 0:
            score += 20
        elif row['trend'] == 0 and row['change_5m'] < 0:
            score += 20
        elif row['trend'] == 1:
            score += 10
        
        # 4. Volatilite skoru (0-15)
        if row['volatility'] > 0.5:
            score += 15
        elif row['volatility'] > 0.3:
            score += 10
        elif row['volatility'] > 0.2:
            score += 5
        
        # 5. Price position (0-15)
        if row['price_position'] > 80:
            score += 15  # Long momentum
        elif row['price_position'] < 20:
            score += 15  # Short momentum
        
        # Yön belirleme
        if row['trend'] == 1 and row['change_5m'] > 0:
            direction = "long"
        elif row['trend'] == 0 and row['change_5m'] < 0:
            direction = "short"
        elif row['rsi'] > 60:
            direction = "long"
        elif row['rsi'] < 40:
            direction = "short"
        else:
            direction = "long" if row['change_1m'] > 0 else "short"
        
        return score, direction
    
    def run_backtest(self, df: pd.DataFrame, symbol: str,
                     entry_threshold: int = 60,
                     exit_after_candles: int = 15,
                     stop_loss: float = 1.0,
                     take_profit: float = 1.5) -> BacktestResult:
        """
        Backtest çalıştır.
        
        Args:
            df: Fiyat verisi
            symbol: Trading çifti
            entry_threshold: Giriş için min skor
            exit_after_candles: Kaç mum sonra çık
            stop_loss: Stop loss %
            take_profit: Take profit %
        """
        print(f"\n[Backtest] {symbol} - Esik: {entry_threshold}, SL: {stop_loss}%, TP: {take_profit}%")
        
        balance = self.initial_balance
        trades: List[Trade] = []
        
        in_position = False
        entry_price = 0
        entry_time = ""
        entry_direction = ""
        entry_score = 0
        entry_index = 0
        
        max_balance = balance
        max_drawdown = 0
        
        for i in range(50, len(df) - exit_after_candles):  # İlk 50 mum indikatörler için
            row = df.iloc[i]
            score, direction = self.calculate_score(row)
            
            if not in_position:
                # Giriş kontrolü
                if score >= entry_threshold:
                    in_position = True
                    entry_price = row['close']
                    entry_time = str(row['timestamp'])
                    entry_direction = direction
                    entry_score = score
                    entry_index = i
            
            else:
                # Çıkış kontrolü
                current_price = row['close']
                candles_passed = i - entry_index
                
                # P/L hesapla
                if entry_direction == "long":
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100
                else:  # short
                    pnl_percent = ((entry_price - current_price) / entry_price) * 100
                
                should_exit = False
                
                # Çıkış koşulları
                if pnl_percent >= take_profit:
                    should_exit = True
                elif pnl_percent <= -stop_loss:
                    should_exit = True
                elif candles_passed >= exit_after_candles:
                    should_exit = True
                
                if should_exit:
                    profit_usd = balance * (pnl_percent / 100)
                    balance += profit_usd
                    
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=str(row['timestamp']),
                        symbol=symbol,
                        direction=entry_direction,
                        entry_price=entry_price,
                        exit_price=current_price,
                        score=entry_score,
                        profit_percent=pnl_percent,
                        profit_usd=profit_usd
                    ))
                    
                    in_position = False
                    
                    # Drawdown hesapla
                    max_balance = max(max_balance, balance)
                    current_dd = ((max_balance - balance) / max_balance) * 100
                    max_drawdown = max(max_drawdown, current_dd)
        
        # Sonuçları hesapla
        winning_trades = [t for t in trades if t.profit_percent > 0]
        losing_trades = [t for t in trades if t.profit_percent <= 0]
        
        result = BacktestResult(
            symbol=symbol,
            start_date=str(df.iloc[0]['timestamp']),
            end_date=str(df.iloc[-1]['timestamp']),
            initial_balance=self.initial_balance,
            final_balance=balance,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=(len(winning_trades) / len(trades) * 100) if trades else 0,
            total_profit_percent=((balance - self.initial_balance) / self.initial_balance) * 100,
            max_drawdown=max_drawdown,
            best_trade=max([t.profit_percent for t in trades]) if trades else 0,
            worst_trade=min([t.profit_percent for t in trades]) if trades else 0,
            avg_profit=np.mean([t.profit_percent for t in trades]) if trades else 0,
            trades=trades
        )
        
        return result
    
    def print_result(self, result: BacktestResult):
        """Sonucu yazdır"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST SONUCU                           ║
╠══════════════════════════════════════════════════════════════╣
║  Coin      : {result.symbol:<20}                       ║
║  Dönem     : {result.start_date[:10]} - {result.end_date[:10]}                ║
╠══════════════════════════════════════════════════════════════╣
║  Başlangıç : ${result.initial_balance:,.2f}                                    ║
║  Bitiş     : ${result.final_balance:,.2f}                                    ║
║  Kar/Zarar : {result.total_profit_percent:+.2f}%                                     ║
╠══════════════════════════════════════════════════════════════╣
║  Toplam İşlem  : {result.total_trades:<10}                               ║
║  Kazanan       : {result.winning_trades} ({result.win_rate:.1f}%)                                  ║
║  Kaybeden      : {result.losing_trades}                                        ║
║  Ort. Kar      : {result.avg_profit:+.2f}%                                     ║
╠══════════════════════════════════════════════════════════════╣
║  En İyi Trade  : {result.best_trade:+.2f}%                                     ║
║  En Kötü Trade : {result.worst_trade:+.2f}%                                    ║
║  Max Drawdown  : {result.max_drawdown:.2f}%                                     ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    def optimize_parameters(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        En iyi parametreleri bul.
        """
        print(f"\n[Backtest] {symbol} icin parametre optimizasyonu...")
        
        best_result = None
        best_params = {}
        
        # Grid search
        for threshold in [50, 55, 60, 65, 70]:
            for sl in [0.5, 1.0, 1.5]:
                for tp in [1.0, 1.5, 2.0, 2.5]:
                    result = self.run_backtest(
                        df, symbol,
                        entry_threshold=threshold,
                        stop_loss=sl,
                        take_profit=tp
                    )
                    
                    # Sharpe-like score: Kar / Drawdown
                    if result.max_drawdown > 0:
                        score = result.total_profit_percent / result.max_drawdown
                    else:
                        score = result.total_profit_percent
                    
                    if best_result is None or score > best_params.get('score', 0):
                        best_result = result
                        best_params = {
                            'threshold': threshold,
                            'stop_loss': sl,
                            'take_profit': tp,
                            'score': score
                        }
        
        print(f"\n[Backtest] En iyi parametreler: Esik={best_params['threshold']}, SL={best_params['stop_loss']}%, TP={best_params['take_profit']}%")
        
        return {
            'params': best_params,
            'result': best_result
        }


async def run_quick_backtest(symbol: str = "SOLUSDT", days: int = 7):
    """Hızlı backtest çalıştır"""
    bt = Backtester(initial_balance=1000)
    
    # Veri indir
    df = await bt.download_historical_data(symbol, interval="1m", days=days)
    
    # İndikatör hesapla
    df = bt.calculate_indicators(df)
    
    # Backtest çalıştır
    result = bt.run_backtest(df, symbol)
    
    # Sonucu göster
    bt.print_result(result)
    
    return result


if __name__ == "__main__":
    # Test
    asyncio.run(run_quick_backtest("SOLUSDT", days=7))
