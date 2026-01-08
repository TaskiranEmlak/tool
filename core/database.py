# HFT Trading Tools - Veritabanı Modülü
"""
SQLite tabanlı zaman serisi veritabanı.
Tick verisi, order book ve likidasyonları saklar.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Trade:
    """Trade veri yapısı"""
    symbol: str
    price: float
    quantity: float
    side: str  # "buy" veya "sell"
    timestamp: datetime
    exchange: str = "binance"


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]
    timestamp: datetime
    exchange: str = "binance"


@dataclass
class Liquidation:
    """Likidasyon verisi"""
    symbol: str
    side: str  # "long" veya "short"
    price: float
    quantity: float
    timestamp: datetime
    exchange: str = "binance"


class Database:
    """
    Zaman serisi veritabanı yöneticisi.
    Yüksek yazma hızı için optimize edilmiş.
    """
    
    def __init__(self, db_path: str = "data/hft_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self):
        """Veritabanı tablolarını oluştur"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Yazma hızı için
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        # Trade tablosu
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                side TEXT NOT NULL,
                exchange TEXT DEFAULT 'binance',
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Order book snapshot tablosu
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bids_json TEXT NOT NULL,
                asks_json TEXT NOT NULL,
                exchange TEXT DEFAULT 'binance',
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Likidasyon tablosu
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS liquidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                exchange TEXT DEFAULT 'binance',
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # CVD tablosu (hesaplanmış değerler)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cvd_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                cvd_value REAL NOT NULL,
                delta REAL NOT NULL,
                window_seconds INTEGER NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """)
        
        # OBI tablosu
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS obi_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                obi_value REAL NOT NULL,
                weighted_obi REAL NOT NULL,
                bid_volume REAL NOT NULL,
                ask_volume REAL NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """)
        
        # Sinyal tablosu
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                reason TEXT,
                status TEXT DEFAULT 'active',
                timestamp DATETIME NOT NULL,
                closed_at DATETIME,
                pnl REAL
            )
        """)
        
        # İndeksler
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_liquidations_symbol_time ON liquidations(symbol, timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cvd_symbol_time ON cvd_values(symbol, timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obi_symbol_time ON obi_values(symbol, timestamp)")
        
        self.conn.commit()
    
    # ============ Trade İşlemleri ============
    
    def insert_trade(self, trade: Trade):
        """Tek trade ekle"""
        self.conn.execute("""
            INSERT INTO trades (symbol, price, quantity, side, exchange, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (trade.symbol, trade.price, trade.quantity, trade.side, 
              trade.exchange, trade.timestamp))
        self.conn.commit()
    
    def insert_trades_batch(self, trades: List[Trade]):
        """Toplu trade ekleme (performans için)"""
        data = [(t.symbol, t.price, t.quantity, t.side, t.exchange, t.timestamp) 
                for t in trades]
        self.conn.executemany("""
            INSERT INTO trades (symbol, price, quantity, side, exchange, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()
    
    def get_trades(self, symbol: str, since: datetime, 
                   until: Optional[datetime] = None) -> List[Trade]:
        """Belirli zaman aralığındaki trade'leri getir"""
        if until is None:
            until = datetime.now()
        
        cursor = self.conn.execute("""
            SELECT symbol, price, quantity, side, timestamp, exchange
            FROM trades
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (symbol, since, until))
        
        return [Trade(
            symbol=row[0], price=row[1], quantity=row[2], 
            side=row[3], timestamp=datetime.fromisoformat(row[4]), 
            exchange=row[5]
        ) for row in cursor.fetchall()]
    
    # ============ Order Book İşlemleri ============
    
    def insert_orderbook(self, snapshot: OrderBookSnapshot):
        """Order book snapshot ekle"""
        self.conn.execute("""
            INSERT INTO orderbook_snapshots (symbol, bids_json, asks_json, exchange, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (snapshot.symbol, json.dumps(snapshot.bids), json.dumps(snapshot.asks),
              snapshot.exchange, snapshot.timestamp))
        self.conn.commit()
    
    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """En son order book snapshot'ı getir"""
        cursor = self.conn.execute("""
            SELECT symbol, bids_json, asks_json, timestamp, exchange
            FROM orderbook_snapshots
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol,))
        
        row = cursor.fetchone()
        if row:
            return OrderBookSnapshot(
                symbol=row[0],
                bids=json.loads(row[1]),
                asks=json.loads(row[2]),
                timestamp=datetime.fromisoformat(row[3]),
                exchange=row[4]
            )
        return None
    
    # ============ Likidasyon İşlemleri ============
    
    def insert_liquidation(self, liq: Liquidation):
        """Likidasyon ekle"""
        self.conn.execute("""
            INSERT INTO liquidations (symbol, side, price, quantity, exchange, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (liq.symbol, liq.side, liq.price, liq.quantity, liq.exchange, liq.timestamp))
        self.conn.commit()
    
    def get_liquidations(self, symbol: str, hours: int = 24) -> List[Liquidation]:
        """Son N saatteki likidasyonları getir"""
        since = datetime.now() - timedelta(hours=hours)
        
        cursor = self.conn.execute("""
            SELECT symbol, side, price, quantity, timestamp, exchange
            FROM liquidations
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (symbol, since))
        
        return [Liquidation(
            symbol=row[0], side=row[1], price=row[2], quantity=row[3],
            timestamp=datetime.fromisoformat(row[4]), exchange=row[5]
        ) for row in cursor.fetchall()]
    
    # ============ CVD İşlemleri ============
    
    def insert_cvd(self, symbol: str, cvd_value: float, delta: float, 
                   window_seconds: int, timestamp: datetime):
        """CVD değeri ekle"""
        self.conn.execute("""
            INSERT INTO cvd_values (symbol, cvd_value, delta, window_seconds, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, cvd_value, delta, window_seconds, timestamp))
        self.conn.commit()
    
    def get_cvd_history(self, symbol: str, window_seconds: int, 
                        limit: int = 100) -> List[Dict[str, Any]]:
        """CVD geçmişi"""
        cursor = self.conn.execute("""
            SELECT cvd_value, delta, timestamp
            FROM cvd_values
            WHERE symbol = ? AND window_seconds = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (symbol, window_seconds, limit))
        
        return [{"cvd": row[0], "delta": row[1], "timestamp": row[2]} 
                for row in cursor.fetchall()]
    
    # ============ OBI İşlemleri ============
    
    def insert_obi(self, symbol: str, obi_value: float, weighted_obi: float,
                   bid_volume: float, ask_volume: float, timestamp: datetime):
        """OBI değeri ekle"""
        self.conn.execute("""
            INSERT INTO obi_values (symbol, obi_value, weighted_obi, bid_volume, ask_volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, obi_value, weighted_obi, bid_volume, ask_volume, timestamp))
        self.conn.commit()
    
    def get_obi_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """OBI geçmişi"""
        cursor = self.conn.execute("""
            SELECT obi_value, weighted_obi, bid_volume, ask_volume, timestamp
            FROM obi_values
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (symbol, limit))
        
        return [{"obi": row[0], "weighted_obi": row[1], "bid_vol": row[2], 
                 "ask_vol": row[3], "timestamp": row[4]} 
                for row in cursor.fetchall()]
    
    # ============ Sinyal İşlemleri ============
    
    def insert_signal(self, symbol: str, direction: str, confidence: float,
                      entry_price: float, stop_loss: float, take_profit: float,
                      reason: str, timestamp: datetime) -> int:
        """Sinyal ekle ve ID döndür"""
        cursor = self.conn.execute("""
            INSERT INTO signals (symbol, direction, confidence, entry_price, 
                                stop_loss, take_profit, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, direction, confidence, entry_price, stop_loss, 
              take_profit, reason, timestamp))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Aktif sinyalleri getir"""
        cursor = self.conn.execute("""
            SELECT id, symbol, direction, confidence, entry_price, 
                   stop_loss, take_profit, reason, timestamp
            FROM signals
            WHERE status = 'active'
            ORDER BY timestamp DESC
        """)
        
        return [{"id": row[0], "symbol": row[1], "direction": row[2],
                 "confidence": row[3], "entry_price": row[4], "stop_loss": row[5],
                 "take_profit": row[6], "reason": row[7], "timestamp": row[8]}
                for row in cursor.fetchall()]
    
    def close_signal(self, signal_id: int, pnl: float):
        """Sinyali kapat"""
        self.conn.execute("""
            UPDATE signals 
            SET status = 'closed', closed_at = ?, pnl = ?
            WHERE id = ?
        """, (datetime.now(), pnl, signal_id))
        self.conn.commit()
    
    # ============ Temizlik İşlemleri ============
    
    def cleanup_old_data(self, days: int = 7):
        """Eski verileri temizle"""
        cutoff = datetime.now() - timedelta(days=days)
        
        self.conn.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff,))
        self.conn.execute("DELETE FROM orderbook_snapshots WHERE timestamp < ?", (cutoff,))
        self.conn.execute("DELETE FROM cvd_values WHERE timestamp < ?", (cutoff,))
        self.conn.execute("DELETE FROM obi_values WHERE timestamp < ?", (cutoff,))
        
        self.conn.execute("VACUUM")
        self.conn.commit()
    
    def get_stats(self) -> Dict[str, int]:
        """Veritabanı istatistikleri"""
        stats = {}
        for table in ["trades", "orderbook_snapshots", "liquidations", 
                      "cvd_values", "obi_values", "signals"]:
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        return stats
    
    def close(self):
        """Bağlantıyı kapat"""
        if self.conn:
            self.conn.close()
