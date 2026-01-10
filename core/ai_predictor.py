# HFT Trading Tools - AI Predictor
"""
Makine öğrenmesi tabanlı fiyat tahmini.
Random Forest modeli kullanarak gelecek hareketi tahmin eder.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import joblib
from datetime import datetime


@dataclass
class AIPrediction:
    """AI tahmin sonucu"""
    symbol: str
    predicted_change: float      # Tahmini % değişim
    confidence: float            # 0-1 arası güven
    direction: str               # "up", "down", "neutral"
    features_used: Dict          # Kullanılan özellikler
    model_version: str


class AIPredictor:
    """
    Makine öğrenmesi tabanlı tahmin modeli.
    """
    
    def __init__(self, model_path: str = "models/crypto_brain.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = [
            'obi', 'volume_ratio', 'momentum_score', 
            'funding_rate', 'long_percent',
            'tf_15m', 'tf_5m', 'tf_1m'
        ]
        self.is_trained = False
        self.model_version = "1.0"
        
        # Model klasörü oluştur
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else "models", exist_ok=True)
        
        # Mevcut model varsa yükle
        self._load_model()
    
    def _load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            if os.path.exists(self.model_path):
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data.get('scaler')
                self.model_version = data.get('version', '1.0')
                self.is_trained = True
                print(f"[AI] Model yuklendi: {self.model_path} (v{self.model_version})")
        except Exception as e:
            print(f"[AI] Model yuklenemedi: {e}")
            self.is_trained = False
    
    def train(self, csv_path: str, min_samples: int = 100) -> Dict:
        """
        Modeli eğit.
        
        Args:
            csv_path: Eğitim verisi CSV yolu
            min_samples: Min örnek sayısı
            
        Returns:
            Eğitim sonuçları
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, r2_score
        
        print(f"[AI] Egitim baslatiyor: {csv_path}")
        
        try:
            # Veri yükle
            df = pd.read_csv(csv_path)
            
            if len(df) < min_samples:
                return {
                    "success": False,
                    "error": f"Yetersiz veri: {len(df)} < {min_samples}"
                }
            
            # Eksik değerleri temizle
            df = df.dropna()
            
            # Özellikler ve hedef
            X = df[self.feature_names].values
            y = df['target_change_percent'].values
            
            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Ölçekleme
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Model eğitimi
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Değerlendirme
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Yön doğruluğu
            correct_direction = sum(
                (y_test[i] > 0 and y_pred[i] > 0) or (y_test[i] < 0 and y_pred[i] < 0)
                for i in range(len(y_test))
            )
            direction_accuracy = correct_direction / len(y_test) * 100
            
            # Feature importance
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Modeli kaydet
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M")
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'version': self.model_version,
                'feature_names': self.feature_names
            }, self.model_path)
            
            self.is_trained = True
            
            result = {
                "success": True,
                "samples": len(df),
                "mae": mae,
                "r2": r2,
                "direction_accuracy": direction_accuracy,
                "feature_importance": importance,
                "model_path": self.model_path,
                "version": self.model_version
            }
            
            print(f"""
[AI] Egitim Tamamlandi!
    Ornekler: {len(df)}
    MAE: {mae:.4f}%
    R2: {r2:.4f}
    Yon Dogrulugu: {direction_accuracy:.1f}%
    En Onemli: {max(importance, key=importance.get)}
""")
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, features: Dict) -> AIPrediction:
        """
        Tahmin yap.
        
        Args:
            features: {'obi': 0.5, 'volume_ratio': 1.5, ...}
        """
        if not self.is_trained:
            return AIPrediction(
                symbol="",
                predicted_change=0,
                confidence=0,
                direction="neutral",
                features_used=features,
                model_version="untrained"
            )
        
        try:
            # Feature vektörü oluştur
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            
            # Ölçekle
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Tahmin
            predicted_change = self.model.predict(X)[0]
            
            # Tüm ağaçların tahminleri (güven için)
            all_predictions = [tree.predict(X)[0] for tree in self.model.estimators_]
            std = np.std(all_predictions)
            
            # Güven hesapla (düşük std = yüksek güven)
            confidence = max(0, min(1, 1 - (std / 2)))
            
            # Yön
            if predicted_change > 0.1:
                direction = "up"
            elif predicted_change < -0.1:
                direction = "down"
            else:
                direction = "neutral"
            
            return AIPrediction(
                symbol=features.get('symbol', ''),
                predicted_change=predicted_change,
                confidence=confidence,
                direction=direction,
                features_used=features,
                model_version=self.model_version
            )
            
        except Exception as e:
            print(f"[AI] Tahmin hatasi: {e}")
            return AIPrediction(
                symbol="",
                predicted_change=0,
                confidence=0,
                direction="neutral",
                features_used=features,
                model_version="error"
            )
    
    def get_status(self) -> Dict:
        """Model durumu"""
        return {
            "is_trained": self.is_trained,
            "model_path": self.model_path,
            "version": self.model_version,
            "features": self.feature_names
        }


class HybridPredictor:
    """
    Mevcut kural tabanlı sistem + AI tahminlerini birleştiren hibrit sistem.
    """
    
    def __init__(self, ai_weight: float = 0.4):
        self.ai_predictor = AIPredictor()
        self.ai_weight = ai_weight  # AI'ın toplam skora etkisi
    
    def predict(self, rule_score: float, rule_direction: str, features: Dict) -> Tuple[float, str, float]:
        """
        Hibrit tahmin.
        
        Args:
            rule_score: Kural tabanlı sistemin skoru (0-100)
            rule_direction: Kural tabanlı sistemin yönü
            features: AI için özellikler
            
        Returns:
            (final_score, final_direction, ai_confidence)
        """
        if not self.ai_predictor.is_trained:
            # AI eğitilmemişse sadece kural tabanlı kullan
            return rule_score, rule_direction, 0
        
        # AI tahmini al
        ai_pred = self.ai_predictor.predict(features)
        
        # AI skoru hesapla (predicted_change'i 0-100 skalasına çevir)
        ai_score = min(100, max(0, 50 + ai_pred.predicted_change * 10))
        
        # Hibrit skor
        final_score = rule_score * (1 - self.ai_weight) + ai_score * self.ai_weight
        
        # Yön belirleme (ikisi de aynı yöndeyse güçlü, farklıysa zayıf)
        if rule_direction == ai_pred.direction:
            final_direction = rule_direction
            final_score *= 1.1  # Bonus
        else:
            # Çelişkili durumda güvene göre karar ver
            if ai_pred.confidence > 0.7:
                final_direction = ai_pred.direction
            else:
                final_direction = rule_direction
            final_score *= 0.9  # Penaltı
        
        return min(100, final_score), final_direction, ai_pred.confidence


class LightGBMPredictor:
    """
    LightGBM tabanlı tahmin modeli.
    RandomForest'tan daha hızlı ve genellikle daha doğru.
    """
    
    def __init__(self, model_path: str = "models/lgbm_crypto.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        
        # Genişletilmiş feature seti
        self.feature_names = [
            'obi', 'volume_ratio', 'momentum_score',
            'funding_rate', 'long_percent', 'oi_change_5m',
            'tf_5m', 'tf_1m', 'btc_lag', 'hour_of_day'
        ]
        
        self.is_trained = False
        self.model_version = "lgbm_1.0"
        self.training_stats = {}
        
        # Model klasörü
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else "models", exist_ok=True)
        
        # Mevcut model varsa yükle
        self._load_model()
    
    def _load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            if os.path.exists(self.model_path):
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data.get('scaler')
                self.model_version = data.get('version', 'lgbm_1.0')
                self.training_stats = data.get('stats', {})
                self.is_trained = True
                print(f"[LightGBM] Model yüklendi: {self.model_path} (v{self.model_version})")
        except Exception as e:
            print(f"[LightGBM] Model yüklenemedi: {e}")
            self.is_trained = False
    
    def train(self, df: pd.DataFrame = None, csv_path: str = None, 
              target_column: str = 'target_change_percent',
              min_samples: int = 50) -> Dict:
        """
        LightGBM modeli eğit.
        
        Args:
            df: Pandas DataFrame (opsiyonel)
            csv_path: CSV dosya yolu (opsiyonel)
            target_column: Hedef kolon adı
            min_samples: Minimum örnek sayısı
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, r2_score
        except ImportError:
            return {"success": False, "error": "lightgbm kurulu değil: pip install lightgbm"}
        
        print("[LightGBM] Eğitim başlıyor...")
        
        try:
            # Veri yükle
            if df is None and csv_path:
                df = pd.read_csv(csv_path)
            
            if df is None or len(df) < min_samples:
                return {
                    "success": False,
                    "error": f"Yetersiz veri: {len(df) if df is not None else 0} < {min_samples}"
                }
            
            # Eksik feature'ları varsayılan değerlerle doldur
            for feature in self.feature_names:
                if feature not in df.columns:
                    if feature == 'hour_of_day':
                        df[feature] = 12  # Varsayılan saat
                    elif feature == 'btc_lag':
                        df[feature] = 0
                    elif feature == 'oi_change_5m':
                        df[feature] = 0
                    else:
                        df[feature] = 0
            
            # Eksik değerleri temizle
            df = df.dropna(subset=[target_column])
            
            # Feature'ları al
            available_features = [f for f in self.feature_names if f in df.columns]
            X = df[available_features].fillna(0).values
            y = df[target_column].values
            
            # Train/Test split (time-based olmalı ama basitlik için random)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Ölçekleme
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # LightGBM parametreleri (overfitting'i önlemek için)
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 200,
                'early_stopping_rounds': 20
            }
            
            # Model eğitimi
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
            
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20)]
            )
            
            # Değerlendirme
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Yön doğruluğu
            correct_direction = sum(
                (y_test[i] > 0 and y_pred[i] > 0) or (y_test[i] < 0 and y_pred[i] < 0)
                for i in range(len(y_test))
            )
            direction_accuracy = correct_direction / len(y_test) * 100
            
            # Feature importance
            importance = dict(zip(available_features, self.model.feature_importance()))
            
            # Model kaydet
            self.model_version = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.training_stats = {
                "samples": len(df),
                "mae": mae,
                "r2": r2,
                "direction_accuracy": direction_accuracy
            }
            
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'version': self.model_version,
                'feature_names': available_features,
                'stats': self.training_stats
            }, self.model_path)
            
            self.is_trained = True
            
            result = {
                "success": True,
                "samples": len(df),
                "mae": mae,
                "r2": r2,
                "direction_accuracy": direction_accuracy,
                "feature_importance": importance,
                "model_path": self.model_path,
                "version": self.model_version
            }
            
            print(f"""
[LightGBM] Eğitim Tamamlandı!
    Örnekler: {len(df)}
    MAE: {mae:.4f}%
    R2: {r2:.4f}
    Yön Doğruluğu: {direction_accuracy:.1f}%
    En Önemli Feature: {max(importance, key=importance.get)}
""")
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def predict(self, features: Dict, symbol: str = "") -> AIPrediction:
        """
        LightGBM ile tahmin yap.
        
        Args:
            features: {'obi': 0.5, 'volume_ratio': 1.5, ...}
            symbol: Coin sembolü
        """
        if not self.is_trained or self.model is None:
            return AIPrediction(
                symbol=symbol,
                predicted_change=0,
                confidence=0,
                direction="neutral",
                features_used=features,
                model_version="untrained"
            )
        
        try:
            # Feature vektörü oluştur
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            
            # Ölçekle
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Tahmin
            predicted_change = self.model.predict(X)[0]
            
            # Güven hesapla (eğitim istatistiklerine göre)
            mae = self.training_stats.get('mae', 0.5)
            confidence = max(0, min(1, 1 - mae / 2))
            
            # Yön
            if predicted_change > 0.1:
                direction = "up"
            elif predicted_change < -0.1:
                direction = "down"
            else:
                direction = "neutral"
            
            return AIPrediction(
                symbol=symbol,
                predicted_change=float(predicted_change),
                confidence=confidence,
                direction=direction,
                features_used=features,
                model_version=self.model_version
            )
            
        except Exception as e:
            print(f"[LightGBM] Tahmin hatası: {e}")
            return AIPrediction(
                symbol=symbol,
                predicted_change=0,
                confidence=0,
                direction="neutral",
                features_used=features,
                model_version="error"
            )
    
    def get_status(self) -> Dict:
        """Model durumu"""
        return {
            "model_type": "LightGBM",
            "is_trained": self.is_trained,
            "model_path": self.model_path,
            "version": self.model_version,
            "features": self.feature_names,
            "training_stats": self.training_stats
        }
    
    def train_from_backtest(self, backtest_csv: str = "data/backtest/results_*.csv"):
        """
        Backtest sonuçlarından model eğit.
        """
        import glob
        
        # En son backtest dosyasını bul
        files = glob.glob(backtest_csv)
        if not files:
            return {"success": False, "error": "Backtest sonucu bulunamadı"}
        
        latest_file = max(files)
        print(f"[LightGBM] Backtest verisinden eğitim: {latest_file}")
        
        return self.train(csv_path=latest_file, target_column='actual_change')


def train_model_from_signals():
    """Training data'dan model eğit"""
    ai = AIPredictor()
    
    # Önce ml_training.csv oluştur
    from core.training_data import TrainingDataCollector
    collector = TrainingDataCollector()
    export_path = collector.export_for_training()
    
    # Modeli eğit
    result = ai.train(export_path)
    
    return result


if __name__ == "__main__":
    # Test
    ai = AIPredictor()
    print(f"Model durumu: {ai.get_status()}")
    
    # Örnek tahmin
    features = {
        'obi': 0.6,
        'volume_ratio': 1.8,
        'momentum_score': 70,
        'funding_rate': 0.0001,
        'long_percent': 55,
        'tf_15m': 1,
        'tf_5m': 1,
        'tf_1m': 0
    }
    
    pred = ai.predict(features)
    print(f"Tahmin: {pred.predicted_change:+.2f}% ({pred.direction}), Guven: {pred.confidence:.1%}")
