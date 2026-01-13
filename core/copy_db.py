# Copy Trading System - Database Module
"""
Trader takibi, pozisyon ve sinyal verilerini yöneten veritabanı modülü.
SQLite kullanır, 30 günlük veri saklar.
"""

import sqlite3
import aiosqlite
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path


class CopyTradingDB:
    """Copy Trading veritabanı yöneticisi"""
    
    def __init__(self, db_path: str = "data/copy_trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Tabloları oluştur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Traderlar tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traders (
                uid TEXT PRIMARY KEY,
                nickname TEXT,
                roi REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                follower_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_scraped DATETIME
            )
        """)
        
        # Pozisyonlar tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trader_uid TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                mark_price REAL,
                size REAL,
                leverage INTEGER,
                pnl_percent REAL DEFAULT 0,
                pnl_usdt REAL DEFAULT 0,
                opened_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                closed_at DATETIME,
                close_price REAL,
                is_open INTEGER DEFAULT 1,
                FOREIGN KEY (trader_uid) REFERENCES traders(uid)
            )
        """)
        
        # Sinyaller tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                direction TEXT,
                consensus_score REAL,
                long_score REAL,
                short_score REAL,
                supporting_traders TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME,
                result TEXT DEFAULT 'PENDING',
                result_pnl REAL
            )
        """)
        
        # Scrape geçmişi (rate limiting için)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scrape_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trader_uid TEXT,
                success INTEGER,
                error_message TEXT,
                scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ============ TRADER İŞLEMLERİ ============
    
    def upsert_trader(self, uid: str, data: Dict) -> bool:
        """Trader ekle veya güncelle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO traders (uid, nickname, roi, pnl, win_rate, follower_count, last_scraped)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                nickname = excluded.nickname,
                roi = excluded.roi,
                pnl = excluded.pnl,
                win_rate = excluded.win_rate,
                follower_count = excluded.follower_count,
                last_scraped = excluded.last_scraped
        """, (
            uid,
            data.get('nickname', 'Unknown'),
            data.get('roi', 0),
            data.get('pnl', 0),
            data.get('win_rate', 0),
            data.get('follower_count', 0),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def get_active_traders(self) -> List[Dict]:
        """Aktif traderları getir"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM traders 
            WHERE is_active = 1 
            ORDER BY roi DESC
        """)
        
        traders = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return traders
    
    def deactivate_trader(self, uid: str):
        """Trader'ı pasif yap"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE traders SET is_active = 0 WHERE uid = ?", (uid,))
        conn.commit()
        conn.close()
    
    def get_trader_stats(self, uid: str) -> Optional[Dict]:
        """Trader istatistiklerini getir"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM traders WHERE uid = ?", (uid,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    # ============ POZİSYON İŞLEMLERİ ============
    
    def add_position(self, trader_uid: str, data: Dict) -> int:
        """Yeni pozisyon ekle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO positions (
                trader_uid, symbol, direction, entry_price, 
                mark_price, size, leverage, pnl_percent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trader_uid,
            data.get('symbol'),
            data.get('direction'),
            data.get('entry_price', 0),
            data.get('mark_price', 0),
            data.get('size', 0),
            data.get('leverage', 1),
            data.get('pnl_percent', 0)
        ))
        
        position_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return position_id
    
    def update_position(self, position_id: int, data: Dict):
        """Pozisyon güncelle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE positions SET
                mark_price = ?,
                pnl_percent = ?,
                pnl_usdt = ?
            WHERE id = ?
        """, (
            data.get('mark_price', 0),
            data.get('pnl_percent', 0),
            data.get('pnl_usdt', 0),
            position_id
        ))
        
        conn.commit()
        conn.close()
    
    def close_position(self, position_id: int, close_price: float, pnl_percent: float):
        """Pozisyonu kapat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE positions SET
                is_open = 0,
                closed_at = ?,
                close_price = ?,
                pnl_percent = ?
            WHERE id = ?
        """, (
            datetime.now().isoformat(),
            close_price,
            pnl_percent,
            position_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_open_positions(self, symbol: str = None) -> List[Dict]:
        """Açık pozisyonları getir"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute("""
                SELECT p.*, t.nickname, t.roi, t.win_rate
                FROM positions p
                JOIN traders t ON p.trader_uid = t.uid
                WHERE p.is_open = 1 AND p.symbol = ?
            """, (symbol,))
        else:
            cursor.execute("""
                SELECT p.*, t.nickname, t.roi, t.win_rate
                FROM positions p
                JOIN traders t ON p.trader_uid = t.uid
                WHERE p.is_open = 1
            """)
        
        positions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return positions
    
    def get_trader_positions(self, uid: str, include_closed: bool = False) -> List[Dict]:
        """Trader'ın pozisyonlarını getir"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if include_closed:
            cursor.execute("SELECT * FROM positions WHERE trader_uid = ? ORDER BY opened_at DESC", (uid,))
        else:
            cursor.execute("SELECT * FROM positions WHERE trader_uid = ? AND is_open = 1", (uid,))
        
        positions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return positions
    
    # ============ SİNYAL İŞLEMLERİ ============
    
    def add_signal(self, data: Dict) -> int:
        """Yeni sinyal ekle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO signals (
                symbol, direction, consensus_score, 
                long_score, short_score, supporting_traders
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            data.get('symbol'),
            data.get('direction'),
            data.get('consensus_score', 0),
            data.get('long_score', 0),
            data.get('short_score', 0),
            json.dumps(data.get('supporting_traders', []))
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
    
    def resolve_signal(self, signal_id: int, result: str, pnl: float):
        """Sinyal sonucunu kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE signals SET
                resolved_at = ?,
                result = ?,
                result_pnl = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), result, pnl, signal_id))
        
        conn.commit()
        conn.close()
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Son sinyalleri getir"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM signals 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return signals
    
    def get_signal_stats(self) -> Dict:
        """Sinyal istatistikleri"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN result != 'PENDING' THEN result_pnl ELSE NULL END) as avg_pnl
            FROM signals
            WHERE created_at > datetime('now', '-30 days')
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        total = row[0] or 0
        wins = row[1] or 0
        losses = row[2] or 0
        avg_pnl = row[3] or 0
        
        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'avg_pnl': avg_pnl
        }
    
    # ============ TEMİZLİK ============
    
    def cleanup_old_data(self, days: int = 30):
        """Eski verileri temizle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("DELETE FROM positions WHERE closed_at < ? AND is_open = 0", (cutoff,))
        cursor.execute("DELETE FROM signals WHERE created_at < ?", (cutoff,))
        cursor.execute("DELETE FROM scrape_log WHERE scraped_at < ?", (cutoff,))
        
        conn.commit()
        conn.close()


# Singleton instance
_db_instance = None

def get_db() -> CopyTradingDB:
    """Veritabanı instance'ı al"""
    global _db_instance
    if _db_instance is None:
        _db_instance = CopyTradingDB()
    return _db_instance
