# HFT Trading Tools - Training Data Collector
"""
Eğitim verisi toplama modülü.
Her dakika sinyal verilerini kaydeder ve 15dk sonra hedef fiyatı günceller.
"""

import asyncio
import aiohttp
import csv
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class TrainingRecord:
    """Tek bir eğitim kaydı"""
    timestamp: str
    symbol: str
    
    # Fiyat verileri
    price: float
    price_change_1m: float
    price_change_5m: float
    
    # İndikatörler
    obi: float              # Order Book Imbalance
    volume_ratio: float     # Hacim / Ortalama
    momentum_score: float   # Momentum skoru
    
    # Market verileri
    funding_rate: float
    long_percent: float
    
    # Timeframe trendleri (1=up, 0=neutral, -1=down)
    tf_15m: int
    tf_5m: int
    tf_1m: int
    
    # Mevcut sistem skoru
    prediction_score: float
    predicted_direction: str  # "up" or "down"
    
    # HEDEF (15dk sonra güncellenir)
    target_price_15m: Optional[float] = None
    target_change_percent: Optional[float] = None
    target_direction: Optional[str] = None  # Gerçekte ne oldu?
    prediction_correct: Optional[bool] = None  # Doğru tahmin mi?


class TrainingDataCollector:
    """
    Eğitim verisi toplama ve yönetimi.
    """
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        self.pending_updates: Dict[str, TrainingRecord] = {}  # 15dk sonra güncellenecekler
        
        # Klasör oluştur
        os.makedirs(data_dir, exist_ok=True)
        
        # CSV dosya yolu
        self.csv_path = os.path.join(data_dir, "signals.csv")
        self._init_csv()
    
    def _init_csv(self):
        """CSV dosyasını başlat"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'price', 'price_change_1m', 'price_change_5m',
                    'obi', 'volume_ratio', 'momentum_score',
                    'funding_rate', 'long_percent',
                    'tf_15m', 'tf_5m', 'tf_1m',
                    'prediction_score', 'predicted_direction',
                    'target_price_15m', 'target_change_percent', 'target_direction', 'prediction_correct'
                ])
    
    def save_signal(self, record: TrainingRecord):
        """Yeni sinyali kaydet"""
        # CSV'ye ekle
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                record.timestamp, record.symbol, record.price,
                record.price_change_1m, record.price_change_5m,
                record.obi, record.volume_ratio, record.momentum_score,
                record.funding_rate, record.long_percent,
                record.tf_15m, record.tf_5m, record.tf_1m,
                record.prediction_score, record.predicted_direction,
                record.target_price_15m, record.target_change_percent,
                record.target_direction, record.prediction_correct
            ])
        
        # 15dk sonra güncellenecek listeye ekle
        key = f"{record.timestamp}_{record.symbol}"
        self.pending_updates[key] = record
        
        print(f"[Training] Kayit: {record.symbol} @ ${record.price:.4f} | Skor: {record.prediction_score:.0f}")
    
    async def update_targets(self):
        """15dk geçen kayıtların hedef fiyatlarını güncelle"""
        now = datetime.now()
        updated_keys = []
        
        async with aiohttp.ClientSession() as session:
            for key, record in self.pending_updates.items():
                try:
                    record_time = datetime.strptime(record.timestamp, "%Y-%m-%d %H:%M:%S")
                    elapsed = (now - record_time).total_seconds()
                    
                    # 15 dakika geçti mi?
                    if elapsed >= 900:  # 15 * 60 = 900 saniye
                        # Güncel fiyatı al
                        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={record.symbol}"
                        async with session.get(url) as resp:
                            data = await resp.json()
                            current_price = float(data.get("price", 0))
                        
                        if current_price > 0:
                            # Hedef değerlerini hesapla
                            record.target_price_15m = current_price
                            record.target_change_percent = ((current_price - record.price) / record.price) * 100
                            
                            # Gerçek yön
                            if record.target_change_percent > 0.1:
                                record.target_direction = "up"
                            elif record.target_change_percent < -0.1:
                                record.target_direction = "down"
                            else:
                                record.target_direction = "neutral"
                            
                            # Tahmin doğru mu?
                            if record.predicted_direction == record.target_direction:
                                record.prediction_correct = True
                            elif record.target_direction == "neutral":
                                record.prediction_correct = None  # Belirsiz
                            else:
                                record.prediction_correct = False
                            
                            # CSV'yi güncelle
                            self._update_csv_record(record)
                            updated_keys.append(key)
                            
                            result = "✓" if record.prediction_correct else "✗" if record.prediction_correct is False else "?"
                            print(f"[Training] Guncellendi: {record.symbol} | Tahmin: {record.predicted_direction} | Gercek: {record.target_direction} ({record.target_change_percent:+.2f}%) {result}")
                
                except Exception as e:
                    print(f"[Training] Guncelleme hatasi: {e}")
        
        # Güncellenen kayıtları listeden çıkar
        for key in updated_keys:
            del self.pending_updates[key]
    
    def _update_csv_record(self, record: TrainingRecord):
        """CSV'deki kaydı güncelle - ATOMIC WRITE"""
        
        # Mevcut verileri oku
        rows = []
        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # İlgili satırı bul ve güncelle
        for i, row in enumerate(rows):
            if len(row) > 1 and row[0] == record.timestamp and row[1] == record.symbol:
                rows[i] = [
                    record.timestamp, record.symbol, record.price,
                    record.price_change_1m, record.price_change_5m,
                    record.obi, record.volume_ratio, record.momentum_score,
                    record.funding_rate, record.long_percent,
                    record.tf_15m, record.tf_5m, record.tf_1m,
                    record.prediction_score, record.predicted_direction,
                    record.target_price_15m, record.target_change_percent,
                    record.target_direction, record.prediction_correct
                ]
                break
        
        # DÜZELTME: Atomic write - Önce temp dosyaya yaz, sonra rename
        dir_path = os.path.dirname(self.csv_path)
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', 
                                          dir=dir_path, delete=False, suffix='.tmp') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            temp_path = f.name
        
        # Temp dosyayı asıl dosya olarak rename et
        shutil.move(temp_path, self.csv_path)
    
    def get_stats(self) -> Dict:
        """İstatistikleri al"""
        total = 0
        correct = 0
        wrong = 0
        pending = len(self.pending_updates)
        
        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row.get('prediction_correct') == 'True':
                    correct += 1
                elif row.get('prediction_correct') == 'False':
                    wrong += 1
        
        accuracy = (correct / (correct + wrong) * 100) if (correct + wrong) > 0 else 0
        
        return {
            "total_signals": total,
            "correct": correct,
            "wrong": wrong,
            "pending": pending,
            "accuracy": accuracy
        }
    
    def export_for_training(self) -> str:
        """ML eğitimi için veri export et"""
        # Sadece tamamlanmış kayıtları al
        export_path = os.path.join(self.data_dir, "ml_training.csv")
        
        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            
            with open(export_path, 'w', newline='', encoding='utf-8') as f_out:
                fieldnames = ['obi', 'volume_ratio', 'momentum_score', 'funding_rate',
                             'long_percent', 'tf_15m', 'tf_5m', 'tf_1m', 'target_change_percent']
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
                
                count = 0
                for row in reader:
                    if row.get('target_change_percent') and row.get('target_change_percent') != 'None':
                        writer.writerow({
                            'obi': row['obi'],
                            'volume_ratio': row['volume_ratio'],
                            'momentum_score': row['momentum_score'],
                            'funding_rate': row['funding_rate'],
                            'long_percent': row['long_percent'],
                            'tf_15m': row['tf_15m'],
                            'tf_5m': row['tf_5m'],
                            'tf_1m': row['tf_1m'],
                            'target_change_percent': row['target_change_percent']
                        })
                        count += 1
        
        print(f"[Training] ML verisi export edildi: {export_path} ({count} kayit)")
        return export_path
