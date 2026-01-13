# Copy Trading System - Consensus Engine
"""
Traderların pozisyonlarını analiz ederek
STRONG sinyaller üreten consensus motoru.
"""

from datetime import datetime
from typing import Dict, List, Optional
from core.copy_db import get_db
import json


class ConsensusEngine:
    """Trader konsensüsü hesaplayan motor"""
    
    def __init__(self):
        self.db = get_db()
        self.min_consensus_score = 5  # Minimum toplam puan
        self.last_signals = {}  # Duplicate önleme
    
    def calculate_trader_weight(self, trader: Dict) -> float:
        """
        Trader'ın oy ağırlığını hesapla
        ROI ve Win Rate'e göre ağırlıklandırma
        """
        roi = trader.get('roi', 0)
        win_rate = trader.get('win_rate', 50)
        
        weight = 1.0
        
        # ROI bazlı bonus
        if roi > 500:
            weight = 3.0
        elif roi > 100:
            weight = 2.0
        elif roi > 50:
            weight = 1.5
        
        # Win Rate bonus
        if win_rate > 70:
            weight *= 1.5
        elif win_rate > 60:
            weight *= 1.2
        
        # Negatif ROI ceza
        if roi < 0:
            weight *= 0.5
        
        return weight
    
    def calculate_consensus(self, symbol: str) -> Dict:
        """
        Belirli bir sembol için consensus hesapla
        
        Returns:
            {
                'symbol': 'BTCUSDT',
                'long_score': 5.5,
                'short_score': 2.0,
                'direction': 'STRONG_LONG',
                'consensus_score': 5.5,
                'supporting_traders': [...]
            }
        """
        positions = self.db.get_open_positions(symbol)
        
        if not positions:
            return {
                'symbol': symbol,
                'long_score': 0,
                'short_score': 0,
                'direction': 'NEUTRAL',
                'consensus_score': 0,
                'supporting_traders': []
            }
        
        long_score = 0.0
        short_score = 0.0
        long_traders = []
        short_traders = []
        
        for pos in positions:
            trader_data = {
                'roi': pos.get('roi', 0),
                'win_rate': pos.get('win_rate', 50)
            }
            weight = self.calculate_trader_weight(trader_data)
            
            if pos['direction'] == 'LONG':
                long_score += weight
                long_traders.append({
                    'nickname': pos.get('nickname', 'Unknown'),
                    'uid': pos['trader_uid'],
                    'weight': weight,
                    'pnl': pos.get('pnl_percent', 0)
                })
            else:
                short_score += weight
                short_traders.append({
                    'nickname': pos.get('nickname', 'Unknown'),
                    'uid': pos['trader_uid'],
                    'weight': weight,
                    'pnl': pos.get('pnl_percent', 0)
                })
        
        # Yön belirleme
        direction = 'NEUTRAL'
        consensus_score = 0
        supporting = []
        
        if long_score > short_score * 2 and long_score >= self.min_consensus_score:
            direction = 'STRONG_LONG'
            consensus_score = long_score
            supporting = long_traders
        elif short_score > long_score * 2 and short_score >= self.min_consensus_score:
            direction = 'STRONG_SHORT'
            consensus_score = short_score
            supporting = short_traders
        elif long_score > short_score and long_score >= self.min_consensus_score / 2:
            direction = 'LONG'
            consensus_score = long_score
            supporting = long_traders
        elif short_score > long_score and short_score >= self.min_consensus_score / 2:
            direction = 'SHORT'
            consensus_score = short_score
            supporting = short_traders
        
        return {
            'symbol': symbol,
            'long_score': long_score,
            'short_score': short_score,
            'direction': direction,
            'consensus_score': consensus_score,
            'supporting_traders': supporting
        }
    
    def get_all_consensus(self) -> List[Dict]:
        """
        Tüm semboller için consensus hesapla
        """
        # Açık pozisyonları olan tüm sembolleri bul
        all_positions = self.db.get_open_positions()
        symbols = set(p['symbol'] for p in all_positions)
        
        results = []
        for symbol in symbols:
            consensus = self.calculate_consensus(symbol)
            if consensus['direction'] != 'NEUTRAL':
                results.append(consensus)
        
        # Consensus score'a göre sırala
        results.sort(key=lambda x: x['consensus_score'], reverse=True)
        
        return results
    
    def check_for_new_signals(self) -> List[Dict]:
        """
        Yeni sinyalleri kontrol et ve kaydet
        Duplicate sinyalleri önle
        """
        new_signals = []
        all_consensus = self.get_all_consensus()
        
        for consensus in all_consensus:
            # Sadece STRONG sinyalleri dikkate al
            if not consensus['direction'].startswith('STRONG'):
                continue
            
            symbol = consensus['symbol']
            direction = consensus['direction'].replace('STRONG_', '')
            
            # Son sinyalle aynı mı?
            last_key = f"{symbol}_{direction}"
            if last_key in self.last_signals:
                # 30 dakika içinde aynı sinyal varsa atla
                last_time = self.last_signals[last_key]
                if (datetime.now() - last_time).seconds < 1800:
                    continue
            
            # Yeni sinyal oluştur
            signal_data = {
                'symbol': symbol,
                'direction': direction,
                'consensus_score': consensus['consensus_score'],
                'long_score': consensus['long_score'],
                'short_score': consensus['short_score'],
                'supporting_traders': consensus['supporting_traders']
            }
            
            # Veritabanına kaydet
            signal_id = self.db.add_signal(signal_data)
            signal_data['id'] = signal_id
            signal_data['created_at'] = datetime.now().isoformat()
            
            # Duplicate önleme için kaydet
            self.last_signals[last_key] = datetime.now()
            
            new_signals.append(signal_data)
            print(f"[Consensus] YENİ SİNYAL: {direction} {symbol} (Skor: {consensus['consensus_score']:.1f})")
        
        return new_signals
    
    def get_market_sentiment(self) -> Dict:
        """
        Genel piyasa duyarlılığını hesapla
        """
        all_positions = self.db.get_open_positions()
        
        if not all_positions:
            return {
                'sentiment': 'NEUTRAL',
                'long_ratio': 50,
                'short_ratio': 50,
                'total_positions': 0
            }
        
        long_count = sum(1 for p in all_positions if p['direction'] == 'LONG')
        short_count = sum(1 for p in all_positions if p['direction'] == 'SHORT')
        total = long_count + short_count
        
        long_ratio = (long_count / total * 100) if total > 0 else 50
        short_ratio = (short_count / total * 100) if total > 0 else 50
        
        if long_ratio > 70:
            sentiment = 'ÇOĞUNLUK LONG'
        elif short_ratio > 70:
            sentiment = 'ÇOĞUNLUK SHORT'
        else:
            sentiment = 'KARARSIZ'
        
        return {
            'sentiment': sentiment,
            'long_ratio': round(long_ratio, 1),
            'short_ratio': round(short_ratio, 1),
            'total_positions': total,
            'long_count': long_count,
            'short_count': short_count
        }
    
    def get_dashboard_data(self) -> Dict:
        """
        Dashboard için özet veri
        """
        traders = self.db.get_active_traders()
        positions = self.db.get_open_positions()
        signal_stats = self.db.get_signal_stats()
        sentiment = self.get_market_sentiment()
        
        # Günlük PnL hesapla
        total_pnl = sum(p.get('pnl_percent', 0) for p in positions)
        
        return {
            'active_traders': len(traders),
            'open_positions': len(positions),
            'total_pnl': round(total_pnl, 2),
            'sentiment': sentiment,
            'signal_stats': signal_stats,
            'top_consensus': self.get_all_consensus()[:5]
        }


# Singleton instance
_engine_instance = None

def get_consensus_engine() -> ConsensusEngine:
    """Consensus engine instance'ı al"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ConsensusEngine()
    return _engine_instance
