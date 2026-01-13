# Trading Brain - Öğrenen Zeka Sistemi
"""
4 Katmanlı Öğrenen Beyin:
1. Anında Hafıza - Her işlemden öğrenme
2. Isı Haritası - RSI x Volume matris
3. Bayesian Olasılık - P(Kazanç | koşullar)
4. Pattern Recognition - Öğrenilen patternler
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class TradeResult:
    """İşlem sonucu"""
    symbol: str
    direction: str  # LONG / SHORT
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl_percent: float
    is_win: bool
    # Giriş koşulları
    rsi: float
    volume_ratio: float
    trend: str  # up / down / sideways
    score: float


@dataclass 
class BrainDecision:
    """Beyin kararı"""
    action: str  # SIGNAL / WATCH / SKIP
    confidence: float  # 0-100
    reasons: List[str]
    heatmap_zone: str  # GREEN / YELLOW / RED
    bayesian_prob: float
    pattern_match: Optional[str]


class HeatmapMemory:
    """
    Isı Haritası Hafızası
    RSI (10 bölge) x Volume (5 bölge) = 50 hücre
    Her hücre kazanç/kayıp oranını tutar
    """
    
    RSI_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    VOL_BINS = [0, 0.5, 1.0, 1.5, 2.0, float('inf')]
    
    def __init__(self, data_path: str = "data/brain"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # Heatmap: {(rsi_bin, vol_bin): {'wins': 0, 'losses': 0}}
        self.heatmap: Dict[Tuple[int, int], Dict] = defaultdict(lambda: {'wins': 0, 'losses': 0})
        self._load()
    
    def _get_bin(self, value: float, bins: List) -> int:
        """Değerin hangi bin'e düştüğünü bul"""
        for i, threshold in enumerate(bins[1:], 1):
            if value < threshold:
                return i - 1
        return len(bins) - 2
    
    def update(self, rsi: float, volume_ratio: float, is_win: bool):
        """İşlem sonucuyla güncelle"""
        rsi_bin = self._get_bin(rsi, self.RSI_BINS)
        vol_bin = self._get_bin(volume_ratio, self.VOL_BINS)
        
        key = (rsi_bin, vol_bin)
        if is_win:
            self.heatmap[key]['wins'] += 1
        else:
            self.heatmap[key]['losses'] += 1
        
        self._save()
    
    def get_probability(self, rsi: float, volume_ratio: float) -> Tuple[float, str]:
        """
        Bu koşullarda kazanç olasılığı
        Returns: (probability, zone_color)
        """
        rsi_bin = self._get_bin(rsi, self.RSI_BINS)
        vol_bin = self._get_bin(volume_ratio, self.VOL_BINS)
        
        key = (rsi_bin, vol_bin)
        data = self.heatmap.get(key, {'wins': 0, 'losses': 0})
        
        total = data['wins'] + data['losses']
        if total < 3:  # Yetersiz veri - GÜVENLİ MOD
            return 0.35, "RED"  # Yeni coinlerde dikkatli ol
        
        prob = data['wins'] / total
        
        if prob >= 0.65:
            zone = "GREEN"
        elif prob >= 0.45:
            zone = "YELLOW"
        else:
            zone = "RED"
        
        return prob, zone
    
    def get_heatmap_display(self) -> str:
        """Terminal için heatmap görselleştirme"""
        lines = ["    RSI: 10  20  30  40  50  60  70  80  90"]
        vol_labels = ["0.5x", "1.0x", "1.5x", "2.0x", "2.0+"]
        
        for vol_bin, vol_label in enumerate(vol_labels):
            row = f"{vol_label} │"
            for rsi_bin in range(9):
                key = (rsi_bin, vol_bin)
                data = self.heatmap.get(key, {'wins': 0, 'losses': 0})
                total = data['wins'] + data['losses']
                
                if total < 3:
                    row += " ⚪ "
                else:
                    prob = data['wins'] / total
                    if prob >= 0.65:
                        row += " 🟢 "
                    elif prob >= 0.45:
                        row += " 🟡 "
                    else:
                        row += " 🔴 "
            lines.append(row)
        
        return "\n".join(lines)
    
    def _save(self):
        """Kaydet"""
        path = os.path.join(self.data_path, "heatmap.json")
        # Convert tuple keys to strings
        data = {f"{k[0]}_{k[1]}": v for k, v in self.heatmap.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Yükle"""
        path = os.path.join(self.data_path, "heatmap.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            # Convert string keys back to tuples
            for key, value in data.items():
                parts = key.split('_')
                self.heatmap[(int(parts[0]), int(parts[1]))] = value


class PatternMemory:
    """Pattern öğrenme ve eşleştirme"""
    
    def __init__(self, data_path: str = "data/brain"):
        self.data_path = data_path
        self.patterns: Dict[str, Dict] = {}
        self._load()
    
    def record_pattern(self, conditions: Dict, result: TradeResult):
        """Yeni pattern kaydet veya mevcut pattern güncelle"""
        # Pattern key oluştur
        key = self._make_key(conditions)
        
        if key not in self.patterns:
            self.patterns[key] = {
                'conditions': conditions,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'examples': []
            }
        
        pattern = self.patterns[key]
        if result.is_win:
            pattern['wins'] += 1
        else:
            pattern['losses'] += 1
        pattern['total_pnl'] += result.pnl_percent
        
        # Son 5 örneği tut
        pattern['examples'].append({
            'symbol': result.symbol,
            'pnl': result.pnl_percent,
            'time': result.entry_time.isoformat()
        })
        pattern['examples'] = pattern['examples'][-5:]
        
        self._save()
    
    def find_matching_pattern(self, conditions: Dict) -> Optional[Dict]:
        """Eşleşen pattern bul"""
        key = self._make_key(conditions)
        if key in self.patterns:
            p = self.patterns[key]
            total = p['wins'] + p['losses']
            if total >= 3:
                return {
                    'name': key,
                    'win_rate': p['wins'] / total,
                    'avg_pnl': p['total_pnl'] / total,
                    'sample_size': total
                }
        return None
    
    def _make_key(self, conditions: Dict) -> str:
        """Koşullardan key oluştur"""
        rsi = conditions.get('rsi', 50)
        vol = conditions.get('volume_ratio', 1)
        
        # 5 bölgeli RSI - DAHA HASSAS
        if rsi < 30: rsi_zone = "vsold"      # Very Oversold
        elif rsi < 40: rsi_zone = "sold"     # Oversold
        elif rsi > 70: rsi_zone = "vbot"     # Very Overbought
        elif rsi > 60: rsi_zone = "bot"      # Overbought
        else: rsi_zone = "neut"
        
        # 3 bölgeli Volume
        if vol > 2.0: vol_zone = "vhigh"
        elif vol > 1.5: vol_zone = "high"
        else: vol_zone = "norm"
        
        trend = conditions.get('trend', 'sideways')
        
        return f"{rsi_zone}_{vol_zone}_{trend}"
    
    def _save(self):
        path = os.path.join(self.data_path, "patterns.json")
        with open(path, 'w') as f:
            json.dump(self.patterns, f, indent=2, default=str)
    
    def _load(self):
        path = os.path.join(self.data_path, "patterns.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.patterns = json.load(f)


class TradingBrain:
    """
    Ana Öğrenen Beyin - GELİŞMİŞ VERSİYON
    
    Özellikler:
    - Trade History: Tüm işlemlerin kaydı
    - Coin-Specific Learning: Her coin için ayrı öğrenme
    - Adaptive Thresholds: Performansa göre eşikler değişir
    - Time Patterns: Saat bazlı performans takibi
    """
    
    def __init__(self, data_path: str = None):
        # Use absolute path from project root
        if data_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(project_root, "data", "brain")
        
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # Alt sistemler
        self.heatmap = HeatmapMemory(data_path)
        self.patterns = PatternMemory(data_path)
        
        # Trade History
        self.trade_history: List[Dict] = []
        
        # Coin-Specific Stats
        self.coin_stats: Dict[str, Dict] = {}
        
        # Adaptive Thresholds - DAHA AGRESİF
        self.signal_threshold = 65  # Daha düşük = daha fazla sinyal
        self.watch_threshold = 45
        
        # Time Patterns (saat bazlı)
        self.hour_stats: Dict[int, Dict] = {h: {'wins': 0, 'losses': 0} for h in range(24)}
        
        # SHADOW SIGNALS - Reddedilen sinyalleri takip et
        # Kaçırılan fırsatlardan öğrenmek için
        self.shadow_signals: Dict[str, Dict] = {}  # {symbol: {entry, direction, time, confidence}}
        self.missed_opportunities = 0  # Kaçırılan karlı fırsatlar
        self.correct_skips = 0  # Doğru skip'ler
        
        # İstatistikler
        self.stats = self._load_stats()
        self._load_history()
        self._load_coin_stats()
        
        # Adaptive threshold'ları hesapla
        self._update_thresholds()
        
        print(f"[Brain] Başlatıldı - {self.stats['total_trades']} işlem | Signal threshold: {self.signal_threshold}%")
    
    def decide(self, symbol: str, rsi: float, volume_ratio: float, 
               trend: str, score: float, direction: str, 
               current_price: float = 0.0) -> BrainDecision:
        """
        Beyin kararı ver - ADAPTIVE
        Geçmiş performansa göre kararlar evrilir
        """
        reasons = []
        
        # 1. Isı Haritası kontrolü
        heatmap_prob, zone = self.heatmap.get_probability(rsi, volume_ratio)
        reasons.append(f"Isı Haritası: {zone} ({heatmap_prob:.0%})")
        
        # 2. Pattern eşleştirme
        conditions = {'rsi': rsi, 'volume_ratio': volume_ratio, 'trend': trend}
        pattern = self.patterns.find_matching_pattern(conditions)
        pattern_match = None
        
        if pattern:
            pattern_match = pattern['name']
            reasons.append(f"Pattern: {pattern_match} ({pattern['win_rate']:.0%} win, n={pattern['sample_size']})")
        
        # 3. Coin-specific history
        coin_bonus = 0
        if symbol in self.coin_stats:
            coin = self.coin_stats[symbol]
            total = coin['wins'] + coin['losses']
            if total >= 5:  # Daha fazla veri gerekli
                coin_wr = coin['wins'] / total
                if coin_wr > 0.70:  # Daha yüksek eşik
                    coin_bonus = 0.10  # Azaltıldı (önceki 0.15)
                    reasons.append(f"🎯 {symbol} iyi performans ({coin_wr:.0%} win)")
                elif coin_wr < 0.35:
                    coin_bonus = -0.10  # Azaltıldı (önceki -0.15)
                    reasons.append(f"⚠️ {symbol} kötü performans ({coin_wr:.0%} win)")
        
        # 4. Time pattern (saat kontrolü)
        current_hour = datetime.now().hour
        hour_bonus = 0
        hour_data = self.hour_stats.get(current_hour, {'wins': 0, 'losses': 0})
        hour_total = hour_data['wins'] + hour_data['losses']
        if hour_total >= 5:
            hour_wr = hour_data['wins'] / hour_total
            if hour_wr > 0.6:
                hour_bonus = 0.10
                reasons.append(f"⏰ Saat {current_hour}:00 iyi ({hour_wr:.0%} win)")
            elif hour_wr < 0.4:
                hour_bonus = -0.10
                reasons.append(f"⏰ Saat {current_hour}:00 kötü ({hour_wr:.0%} win)")
        
        # 5. Bayesian güncellenmiş olasılık
        base_prob = 0.5
        
        # RSI etkisi
        if direction == "LONG" and rsi < 35:
            base_prob += 0.15
            reasons.append("RSI oversold (+15%)")
        elif direction == "SHORT" and rsi > 65:
            base_prob += 0.15
            reasons.append("RSI overbought (+15%)")
        
        # Volume etkisi
        if volume_ratio > 1.5:
            base_prob += 0.10
            reasons.append(f"Volume {volume_ratio:.1f}x (+10%)")
        
        # Trend etkisi
        if (direction == "LONG" and trend == "up") or (direction == "SHORT" and trend == "down"):
            base_prob += 0.10
            reasons.append("Trend hizalı (+10%)")
        
        # Heatmap etkisi (öğrenilmiş)
        if zone == "GREEN":
            base_prob += 0.15
            reasons.append("🟢 Heatmap green zone (+15%)")
        elif zone == "RED":
            base_prob -= 0.20
            reasons.append("🔴 Heatmap red zone (-20%)")
        
        # Pattern etkisi
        if pattern and pattern['win_rate'] > 0.6:
            base_prob += 0.10
        elif pattern and pattern['win_rate'] < 0.4:
            base_prob -= 0.10
        
        # Coin ve saat bonusları
        base_prob += coin_bonus + hour_bonus
        
        bayesian_prob = min(0.95, max(0.05, base_prob))
        
        # 6. Final karar (ADAPTIVE thresholds)
        confidence = bayesian_prob * 100
        
        if confidence >= self.signal_threshold and zone != "RED":
            action = "SIGNAL"
        elif confidence >= self.watch_threshold:
            action = "WATCH"
            # SHADOW: Watch'ları da takip et - belki sinyal olmalıydı?
            # current_price kullanılıyor (score değil!) - P&L hesaplaması için kritik
            if current_price > 0:
                self.shadow_signals[symbol] = {
                    'entry': current_price,  # DÜZELTME: gerçek fiyat kullanılıyor
                    'direction': direction,
                    'time': datetime.now(),
                    'confidence': confidence,
                    'was_watch': True
                }
        else:
            action = "SKIP"
            # SHADOW: Skip'leri takip et - kaçırılan fırsat mı?
            if current_price > 0:
                self.shadow_signals[symbol] = {
                    'entry': current_price,  # DÜZELTME: gerçek fiyat kullanılıyor
                    'direction': direction,
                    'time': datetime.now(),
                    'confidence': confidence,
                    'was_watch': False
                }
        
        return BrainDecision(
            action=action,
            confidence=confidence,
            reasons=reasons,
            heatmap_zone=zone,
            bayesian_prob=bayesian_prob,
            pattern_match=pattern_match
        )
    
    def track_shadow_signal(self, symbol: str, entry_price: float):
        """Shadow sinyale gerçek entry fiyatı ekle"""
        if symbol in self.shadow_signals:
            self.shadow_signals[symbol]['entry'] = entry_price
    
    def check_shadow_signals(self, current_prices: Dict[str, float]):
        """
        KAÇIRILAN FIRSATLARI KONTROL ET
        
        Eğer skip/watch ettiğimiz bir sinyal karlı olsaydı,
        threshold'u gevşet (daha agresif ol)
        """
        expired = []
        
        for symbol, shadow in list(self.shadow_signals.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            entry = shadow['entry']
            direction = shadow['direction']
            elapsed = (datetime.now() - shadow['time']).total_seconds()
            
            # 30 dakika sonra kontrol et (daha güvenilir)
            if elapsed < 1800:  # 30 dakika
                continue
            
            # P&L hesapla
            if direction == "LONG":
                pnl = ((current_price - entry) / entry) * 100
            else:
                pnl = ((entry - current_price) / entry) * 100
            
            # Analiz
            if pnl >= 1.0:  # %1 veya daha fazla kar
                self.missed_opportunities += 1
                was_watch = shadow.get('was_watch', False)
                
                if was_watch:
                    print(f"[Brain] 😤 KAÇIRILAN FIRSAT: {symbol} {direction} | +{pnl:.2f}% (WATCH idi)")
                else:
                    print(f"[Brain] 😤 KAÇIRILAN FIRSAT: {symbol} {direction} | +{pnl:.2f}% (SKIP idi)")
                
                # DENGELİ threshold ayarı: her 3 kaçırılan fırsatta 1 düşür
                if self.missed_opportunities % 3 == 0:
                    self.signal_threshold = max(55, self.signal_threshold - 1)
                
                print(f"[Brain]    → Threshold gevşetildi: {self.signal_threshold}%")
                
            elif pnl <= -1.0:  # %1 veya daha fazla zarar
                self.correct_skips += 1
                print(f"[Brain] ✅ DOĞRU SKIP: {symbol} {direction} | {pnl:.2f}%")
                # DENGELİ: Her 3 doğru skip'te 1 artır (önceki 5'ti)
                if self.correct_skips % 3 == 0:
                    self.signal_threshold = min(80, self.signal_threshold + 1)
            
            expired.append(symbol)
        
        # Temizle
        for symbol in expired:
            del self.shadow_signals[symbol]
    
    def learn(self, result: TradeResult):
        """İşlem sonucundan öğren - GELİŞMİŞ"""
        
        # 1. Heatmap güncelle
        self.heatmap.update(result.rsi, result.volume_ratio, result.is_win)
        
        # 2. Pattern kaydet
        conditions = {
            'rsi': result.rsi,
            'volume_ratio': result.volume_ratio,
            'trend': result.trend
        }
        self.patterns.record_pattern(conditions, result)
        
        # 3. Trade history'e ekle
        trade_record = {
            'symbol': result.symbol,
            'direction': result.direction,
            'entry_price': result.entry_price,
            'exit_price': result.exit_price,
            'entry_time': result.entry_time.isoformat(),
            'exit_time': result.exit_time.isoformat(),
            'pnl_percent': result.pnl_percent,
            'is_win': result.is_win,
            'rsi': result.rsi,
            'volume_ratio': result.volume_ratio,
            'trend': result.trend
        }
        self.trade_history.append(trade_record)
        self._save_history()
        
        # 4. Coin-specific stats güncelle
        if result.symbol not in self.coin_stats:
            self.coin_stats[result.symbol] = {'wins': 0, 'losses': 0, 'pnl': 0}
        
        if result.is_win:
            self.coin_stats[result.symbol]['wins'] += 1
        else:
            self.coin_stats[result.symbol]['losses'] += 1
        self.coin_stats[result.symbol]['pnl'] += result.pnl_percent
        self._save_coin_stats()
        
        # 5. Time pattern güncelle
        hour = result.entry_time.hour
        if result.is_win:
            self.hour_stats[hour]['wins'] += 1
        else:
            self.hour_stats[hour]['losses'] += 1
        
        # 6. Global stats güncelle
        self.stats['total_trades'] += 1
        if result.is_win:
            self.stats['wins'] += 1
        self.stats['total_pnl'] += result.pnl_percent
        self._save_stats()
        
        # 7. Adaptive threshold güncelle
        self._update_thresholds()
        
        outcome = "✅ WIN" if result.is_win else "❌ LOSS"
        print(f"[Brain] 📚 ÖĞRENİLDİ: {result.symbol} {outcome} {result.pnl_percent:+.2f}%")
        print(f"[Brain]    → Heatmap, Pattern, Coin Stats, Hour Stats güncellendi")
        print(f"[Brain]    → Yeni threshold: {self.signal_threshold}%")
    
    def _update_thresholds(self):
        """
        Adaptive Thresholds + Smart Pause
        
        - İyi performansta (> 75%) → öğrenmeyi duraklat, ayarları koru
        - Performans düşerse (< 60%) → öğrenmeye devam et
        - Kötü performans → daha seçici ol
        - İyi performans → daha agresif ol
        """
        total = self.stats['total_trades']
        if total < 10:
            self.learning_paused = False
            return  # Yetersiz veri
        
        # Son 20 trade'e bak
        recent = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
        recent_wins = sum(1 for t in recent if t['is_win'])
        recent_wr = recent_wins / len(recent) if recent else 0.5
        
        # Smart Pause Logic
        if recent_wr >= 0.75:
            # MÜKEMMEL performans - öğrenmeyi duraklat!
            if not getattr(self, 'learning_paused', False):
                print(f"[Brain] 🎯 MÜKEMMEL PERFORMANS ({recent_wr:.0%})! Ayarlar kilitlendi.")
                self.learning_paused = True
                self.locked_threshold = self.signal_threshold
            return  # Threshold değiştirme
        
        elif recent_wr < 0.60 and getattr(self, 'learning_paused', False):
            # Performans düştü - öğrenmeye devam et
            print(f"[Brain] ⚠️ Performans düştü ({recent_wr:.0%}). Öğrenme devam ediyor...")
            self.learning_paused = False
        
        # Learning paused değilse threshold ayarla
        if not getattr(self, 'learning_paused', False):
            if recent_wr >= 0.7:
                # İyi performans - daha agresif ol
                self.signal_threshold = max(60, self.signal_threshold - 2)
                self.watch_threshold = max(40, self.watch_threshold - 2)
            elif recent_wr <= 0.4:
                # Kötü performans - daha seçici ol
                self.signal_threshold = min(85, self.signal_threshold + 3)
                self.watch_threshold = min(60, self.watch_threshold + 2)
    
    def get_status(self) -> Dict:
        """Beyin durumu - detaylı"""
        total = self.stats['total_trades']
        wins = self.stats['wins']
        
        # Son 10 trade
        recent = self.trade_history[-10:] if self.trade_history else []
        recent_wins = sum(1 for t in recent if t['is_win'])
        
        # En iyi/kötü coinler
        best_coin = None
        worst_coin = None
        
        for symbol, data in self.coin_stats.items():
            coin_total = data['wins'] + data['losses']
            if coin_total >= 3:
                wr = data['wins'] / coin_total
                if best_coin is None or wr > best_coin[1]:
                    best_coin = (symbol, wr)
                if worst_coin is None or wr < worst_coin[1]:
                    worst_coin = (symbol, wr)
        
        # Heatmap verisi (web için)
        heatmap_data = []
        for vol_bin in range(5):
            row = []
            for rsi_bin in range(9):
                key = (rsi_bin, vol_bin)
                data = self.heatmap.heatmap.get(key, {'wins': 0, 'losses': 0})
                total_cell = data['wins'] + data['losses']
                if total_cell < 3:
                    row.append({'zone': 'empty', 'count': total_cell})
                else:
                    prob = data['wins'] / total_cell
                    zone = 'green' if prob >= 0.65 else 'yellow' if prob >= 0.45 else 'red'
                    row.append({'zone': zone, 'count': total_cell, 'prob': prob})
            heatmap_data.append(row)
        
        return {
            'total_trades': total,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pnl': self.stats['total_pnl'],
            'heatmap_zones': len(self.heatmap.heatmap),
            'patterns_learned': len(self.patterns.patterns),
            'recent_win_rate': (recent_wins / len(recent) * 100) if recent else 0,
            'signal_threshold': self.signal_threshold,
            'watch_threshold': self.watch_threshold,
            'coins_tracked': len(self.coin_stats),
            'best_coin': best_coin,
            'worst_coin': worst_coin,
            'heatmap_data': heatmap_data,
            'learning_paused': getattr(self, 'learning_paused', False)
        }
    
    def _load_stats(self) -> Dict:
        path = os.path.join(self.data_path, "stats.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {'total_trades': 0, 'wins': 0, 'total_pnl': 0}
    
    def _save_stats(self):
        path = os.path.join(self.data_path, "stats.json")
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _load_history(self):
        path = os.path.join(self.data_path, "trade_history.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.trade_history = json.load(f)
    
    def _save_history(self):
        path = os.path.join(self.data_path, "trade_history.json")
        with open(path, 'w') as f:
            json.dump(self.trade_history[-500:], f, indent=2)  # Son 500 trade
    
    def _load_coin_stats(self):
        path = os.path.join(self.data_path, "coin_stats.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.coin_stats = json.load(f)
    
    def _save_coin_stats(self):
        path = os.path.join(self.data_path, "coin_stats.json")
        with open(path, 'w') as f:
            json.dump(self.coin_stats, f, indent=2)


# Singleton
_brain_instance = None

def get_brain() -> TradingBrain:
    """Trading Brain singleton"""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = TradingBrain()
    return _brain_instance


# Test
if __name__ == "__main__":
    brain = get_brain()
    
    print("\n=== BEYİN DURUMU ===")
    status = brain.get_status()
    print(f"Toplam İşlem: {status['total_trades']}")
    print(f"Win Rate: {status['win_rate']:.1f}%")
    print(f"Öğrenilen Patternler: {status['patterns_learned']}")
    
    print("\n=== ISI HARİTASI ===")
    print(brain.heatmap.get_heatmap_display())
    
    print("\n=== TEST KARAR ===")
    decision = brain.decide(
        symbol="BTCUSDT",
        rsi=28,
        volume_ratio=1.8,
        trend="up",
        score=65,
        direction="LONG"
    )
    print(f"Karar: {decision.action}")
    print(f"Güven: {decision.confidence:.0f}%")
    print(f"Zone: {decision.heatmap_zone}")
    print("Nedenler:")
    for r in decision.reasons:
        print(f"  - {r}")
