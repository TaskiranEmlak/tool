# HFT Trading Tools - Deep Learning Predictor
"""
LSTM tabanlı derin öğrenme tahmin modeli.
%70-80 win rate hedefli gelişmiş sistem.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, Bidirectional,
        BatchNormalization, Attention, Concatenate,
        GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[DeepLearning] TensorFlow not available, using fallback")

# Sklearn imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib


@dataclass
class DeepPrediction:
    """Derin öğrenme tahmin sonucu"""
    symbol: str
    direction: str  # "up", "down", "neutral"
    confidence: float  # 0-1
    predicted_change: float  # Tahmini % değişim
    probabilities: Dict  # {"up": 0.7, "down": 0.2, "neutral": 0.1}
    model_version: str


class LSTMPredictor:
    """
    LSTM (Long Short-Term Memory) tabanlı derin öğrenme modeli.
    
    Özellikler:
    - Bidirectional LSTM ile geçmiş ve gelecek bağlamı
    - Multi-head attention mekanizması
    - 50+ teknik indikatör
    - Ensemble learning desteği
    """
    
    def __init__(self, model_path: str = "models/lstm_crypto.keras"):
        self.model_path = model_path
        self.scaler_path = model_path.replace('.keras', '_scaler.pkl')
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_version = "lstm_v2.0"
        
        # Model parametreleri
        self.sequence_length = 60  # 60 mum (5 saat)
        self.n_features = 50  # Özellik sayısı
        self.n_classes = 3  # UP, DOWN, NEUTRAL
        
        # Feature listesi - TAM 50 OZELLIK
        self.feature_columns = [
            # Fiyat özellikleri (4)
            'returns', 'log_returns', 'high_low_range', 'close_open_range',
            # Hacim (3)
            'volume_ratio', 'volume_change', 'volume_ma_ratio',
            # RSI varyasyonları (3)
            'rsi', 'rsi_smooth', 'rsi_slope',
            # MACD (4)
            'macd', 'macd_signal', 'macd_hist', 'macd_slope',
            # Stochastic (3)
            'stoch_k', 'stoch_d', 'stoch_diff',
            # Bollinger Bands (3)
            'bb_position', 'bb_width', 'bb_squeeze',
            # EMA/SMA (4)
            'ema_9', 'ema_21', 'ema_50', 'ema_cross',
            # Momentum (4)
            'momentum_4', 'momentum_12', 'roc', 'williams_r',
            # Volatility (3)
            'atr', 'atr_percent', 'volatility',
            # Trend (4)
            'adx', 'plus_di', 'minus_di', 'trend_strength',
            # Price action (3)
            'upper_shadow', 'lower_shadow', 'body_ratio',
            # Pattern features (3)
            'higher_high', 'lower_low', 'inside_bar',
            # Time features (4)
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            # Lag features (3)
            'return_lag_1', 'return_lag_4', 'return_lag_12',
            # Rolling stats (2)
            'rolling_mean_12', 'zscore'
        ]  # Total: 50 features
        
        # Modeli yükle
        self._load_model()
    
    def _load_model(self):
        """Eğitilmiş modeli yükle"""
        if os.path.exists(self.model_path):
            try:
                if TF_AVAILABLE:
                    self.model = load_model(self.model_path)
                    self.is_trained = True
                    print(f"[DeepLearning] LSTM model yüklendi: {self.model_path}")
                
                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
            except Exception as e:
                print(f"[DeepLearning] Model yükleme hatası: {e}")
                self.is_trained = False
    
    def _build_model(self) -> Model:
        """LSTM model mimarisi"""
        if not TF_AVAILABLE:
            return None
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # First Bidirectional LSTM
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second Bidirectional LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Multi-Head Attention
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=32
        )(x, x)
        x = LayerNormalization()(x + attention_output)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output - 3 sınıf (UP, NEUTRAL, DOWN)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ham OHLCV verisinden 50+ özellik hesapla.
        
        Args:
            df: open, high, low, close, volume içeren DataFrame
            
        Returns:
            Özellikler eklenmiş DataFrame
        """
        df = df.copy()
        
        # Fiyat özellikleri
        df['returns'] = df['close'].pct_change() * 100
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)) * 100
        df['high_low_range'] = (df['high'] - df['low']) / df['close'] * 100
        df['close_open_range'] = (df['close'] - df['open']) / df['open'] * 100
        
        # Hacim
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(50).mean().replace(0, 1)
        
        # RSI (14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.0001)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_smooth'] = df['rsi'].rolling(3).mean()
        df['rsi_slope'] = df['rsi'].diff(3)
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_slope'] = df['macd'].diff(3)
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14).replace(0, 0.0001)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 0.0001)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20'] * 100
        df['bb_squeeze'] = df['bb_width'].rolling(20).rank(pct=True)
        
        # EMA/SMA
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['close'] * 100
        
        # Momentum
        df['momentum_4'] = df['close'].pct_change(4) * 100
        df['momentum_12'] = df['close'].pct_change(12) * 100
        df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        # Williams %R
        df['williams_r'] = (high_14 - df['close']) / (high_14 - low_14).replace(0, 0.0001) * -100
        
        # ATR (Volatility)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_percent'] = df['atr'] / df['close'] * 100
        df['volatility'] = df['returns'].rolling(20).std()
        
        # ADX (Trend Strength)
        df['plus_dm'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0), 0
        )
        df['minus_dm'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0), 0
        )
        df['plus_di'] = 100 * (pd.Series(df['plus_dm']).rolling(14).mean() / df['atr'].replace(0, 0.0001))
        df['minus_di'] = 100 * (pd.Series(df['minus_dm']).rolling(14).mean() / df['atr'].replace(0, 0.0001))
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 0.0001)
        df['adx'] = df['dx'].rolling(14).mean()
        df['trend_strength'] = df['adx'] * np.sign(df['plus_di'] - df['minus_di'])
        
        # Price action patterns
        body = abs(df['close'] - df['open'])
        total_range = (df['high'] - df['low']).replace(0, 0.0001)
        df['upper_shadow'] = (df['high'] - np.maximum(df['close'], df['open'])) / total_range
        df['lower_shadow'] = (np.minimum(df['close'], df['open']) - df['low']) / total_range
        df['body_ratio'] = body / total_range
        
        # Pattern features
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
        
        # Time features (cyclical encoding)
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.dayofweek
        else:
            df['hour'] = 12
            df['day'] = 3
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
        
        # Lag features
        df['return_lag_1'] = df['returns'].shift(1)
        df['return_lag_4'] = df['returns'].shift(4)
        df['return_lag_12'] = df['returns'].shift(12)
        
        # Rolling statistics
        df['rolling_mean_12'] = df['returns'].rolling(12).mean()
        df['rolling_std_12'] = df['returns'].rolling(12).std()
        df['zscore'] = (df['returns'] - df['rolling_mean_12']) / df['rolling_std_12'].replace(0, 0.0001)
        
        # NaN temizliği - COLD START duzeltmesi
        # Ilk 60 satiri at (en buyuk indikatör periyodu kadar)
        # fillna(0) yerine drop yapmak daha dogru - 0 ile doldurmak
        # indikatorleri bozar ve yanlis sinyaller uretir
        df = df.iloc[60:].reset_index(drop=True)  # Ilk 60 satiri at
        df = df.fillna(method='ffill')  # Kalan NaN'ler icin forward fill
        df = df.dropna()  # Hala NaN varsa at
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Veriyi LSTM için sequence formatına dönüştür.
        
        Args:
            df: Özellikler hesaplanmış DataFrame
            target_col: Hedef kolon (None ise sadece X döndür)
            
        Returns:
            X: (samples, sequence_length, features) shape'inde array
            y: (samples, n_classes) one-hot encoded labels
        """
        # Sadece gerekli kolonları al
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        if len(feature_cols) < 30:
            print(f"[DeepLearning] Uyarı: Sadece {len(feature_cols)} özellik bulundu")
        
        # Eksik özellikleri sıfırla
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        data = df[self.feature_columns].values
        
        # Scaler uygula
        if self.scaler is None:
            self.scaler = RobustScaler()
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)
        
        # Sequence oluştur
        X = []
        y = []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            
            if target_col and target_col in df.columns:
                target_value = df[target_col].iloc[i]
                
                # 3 sınıf: UP (>0.3%), NEUTRAL (-0.3% to 0.3%), DOWN (<-0.3%)
                if target_value > 0.3:
                    y.append([1, 0, 0])  # UP
                elif target_value < -0.3:
                    y.append([0, 0, 1])  # DOWN
                else:
                    y.append([0, 1, 0])  # NEUTRAL
        
        X = np.array(X)
        y = np.array(y) if y else None
        
        return X, y
    
    def train(self, df: pd.DataFrame = None, csv_path: str = None,
              epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """
        LSTM modeli eğit.
        
        Args:
            df: Pandas DataFrame (opsiyonel)
            csv_path: CSV dosya yolu (opsiyonel)
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            validation_split: Validasyon oranı
        """
        if not TF_AVAILABLE:
            print("[DeepLearning] TensorFlow yüklü değil!")
            return None
        
        print("\n" + "="*60)
        print("     LSTM MODEL EĞİTİMİ")
        print("="*60 + "\n")
        
        # Veri yükle
        if df is None and csv_path:
            df = pd.read_csv(csv_path)
        
        if df is None or len(df) < self.sequence_length + 100:
            print(f"[DeepLearning] Yetersiz veri! Min {self.sequence_length + 100} satır gerekli")
            return None
        
        # Özellikleri hesapla
        print("[DeepLearning] Özellikler hesaplanıyor...")
        df = self.calculate_features(df)
        
        # Target kolon - 4 mum sonraki değişim (20 dakika @ 5m)
        df['target'] = df['close'].shift(-4).pct_change(4) * 100
        df = df.dropna()
        
        # Sequence hazırla
        print("[DeepLearning] Sequence'lar oluşturuluyor...")
        X, y = self.prepare_sequences(df, target_col='target')
        
        if len(X) < 100:
            print(f"[DeepLearning] Yetersiz örnek: {len(X)}")
            return None
        
        print(f"[DeepLearning] Toplam örnek: {len(X)}")
        print(f"[DeepLearning] X shape: {X.shape}")
        print(f"[DeepLearning] y distribution: UP={np.sum(y[:,0])}, NEUTRAL={np.sum(y[:,1])}, DOWN={np.sum(y[:,2])}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        # Class weights (dengesizlik için)
        total = len(y_train)
        class_weights = {
            0: total / (3 * np.sum(y_train[:, 0]) + 1),  # UP
            1: total / (3 * np.sum(y_train[:, 1]) + 1),  # NEUTRAL
            2: total / (3 * np.sum(y_train[:, 2]) + 1)   # DOWN
        }
        
        # Model oluştur
        print("[DeepLearning] Model oluşturuluyor...")
        self.model = self._build_model()
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Eğitim
        print("\n[DeepLearning] Eğitim başlıyor...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Sonuçları değerlendir
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\n[DeepLearning] Validation Accuracy: {val_acc*100:.1f}%")
        print(f"[DeepLearning] Validation Loss: {val_loss:.4f}")
        
        # Scaler'ı kaydet
        joblib.dump(self.scaler, self.scaler_path)
        print(f"[DeepLearning] Scaler kaydedildi: {self.scaler_path}")
        
        self.is_trained = True
        
        return {
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "epochs_trained": len(history.history['loss']),
            "best_accuracy": max(history.history['val_accuracy'])
        }
    
    def predict(self, df: pd.DataFrame, symbol: str = "") -> DeepPrediction:
        """
        LSTM ile tahmin yap.
        
        Args:
            df: Son 60+ mumu içeren DataFrame
            symbol: Coin sembolü
            
        Returns:
            DeepPrediction objesi
        """
        if not self.is_trained or self.model is None:
            # Model yoksa basit tahmin
            return DeepPrediction(
                symbol=symbol,
                direction="neutral",
                confidence=0.33,
                predicted_change=0.0,
                probabilities={"up": 0.33, "neutral": 0.34, "down": 0.33},
                model_version="fallback"
            )
        
        # Özellikleri hesapla
        df_features = self.calculate_features(df.copy())
        
        # Sadece son sequence'ı al
        X, _ = self.prepare_sequences(df_features)
        
        if len(X) == 0:
            return DeepPrediction(
                symbol=symbol,
                direction="neutral",
                confidence=0.33,
                predicted_change=0.0,
                probabilities={"up": 0.33, "neutral": 0.34, "down": 0.33},
                model_version="error"
            )
        
        # Son sequence ile tahmin
        X_last = X[-1:].reshape(1, self.sequence_length, self.n_features)
        
        # Tahmin
        probs = self.model.predict(X_last, verbose=0)[0]
        
        # Sonuçları yorumla
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        
        direction_map = {0: "up", 1: "neutral", 2: "down"}
        direction = direction_map[class_idx]
        
        # Tahmini değişim miktarı
        if direction == "up":
            predicted_change = confidence * 0.8  # Max %0.8
        elif direction == "down":
            predicted_change = -confidence * 0.8
        else:
            predicted_change = 0.0
        
        return DeepPrediction(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            predicted_change=predicted_change,
            probabilities={
                "up": float(probs[0]),
                "neutral": float(probs[1]),
                "down": float(probs[2])
            },
            model_version=self.model_version
        )
    
    def predict_proba(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """UP, NEUTRAL, DOWN olasılıklarını döndür"""
        pred = self.predict(df)
        return (
            pred.probabilities['up'],
            pred.probabilities['neutral'],
            pred.probabilities['down']
        )
    
    def get_status(self) -> Dict:
        """Model durumu"""
        return {
            "model_type": "LSTM",
            "is_trained": self.is_trained,
            "model_path": self.model_path,
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "version": self.model_version,
            "tf_available": TF_AVAILABLE
        }


class EnsemblePredictor:
    """
    LSTM + LightGBM + Rule-based ensemble.
    En yüksek doğruluk için 3 sistemi birleştirir.
    """
    
    def __init__(self, lstm_weight: float = 0.45, lgbm_weight: float = 0.35, rule_weight: float = 0.20):
        self.lstm_predictor = LSTMPredictor()
        
        # LightGBM importu
        try:
            from core.ai_predictor import LightGBMPredictor
            self.lgbm_predictor = LightGBMPredictor()
            self.lgbm_available = True
        except:
            self.lgbm_predictor = None
            self.lgbm_available = False
        
        self.lstm_weight = lstm_weight
        self.lgbm_weight = lgbm_weight if self.lgbm_available else 0
        self.rule_weight = rule_weight
        
        # Eğer LightGBM yoksa ağırlıkları yeniden dağıt
        if not self.lgbm_available:
            self.lstm_weight = 0.65
            self.rule_weight = 0.35
    
    def predict(self, df: pd.DataFrame, rule_score: float, rule_direction: str,
                features: Dict = None, symbol: str = "") -> Tuple[str, float, float]:
        """
        Ensemble tahmin.
        
        Args:
            df: OHLCV DataFrame
            rule_score: Kural tabanlı skor (0-100)
            rule_direction: Kural tabanlı yön ("up"/"down")
            features: LightGBM için özellikler
            symbol: Coin sembolü
            
        Returns:
            (final_direction, final_confidence, predicted_change)
        """
        # LSTM tahmini
        lstm_pred = self.lstm_predictor.predict(df, symbol)
        
        # LightGBM tahmini
        lgbm_direction = "neutral"
        lgbm_confidence = 0.33
        if self.lgbm_available and features:
            lgbm_pred = self.lgbm_predictor.predict(features, symbol)
            lgbm_direction = lgbm_pred.direction
            lgbm_confidence = lgbm_pred.confidence
        
        # Yön skorları hesapla
        up_score = 0.0
        down_score = 0.0
        
        # LSTM katkısı
        up_score += lstm_pred.probabilities['up'] * self.lstm_weight
        down_score += lstm_pred.probabilities['down'] * self.lstm_weight
        
        # LightGBM katkısı
        if self.lgbm_available:
            if lgbm_direction == "up":
                up_score += lgbm_confidence * self.lgbm_weight
            elif lgbm_direction == "down":
                down_score += lgbm_confidence * self.lgbm_weight
        
        # Rule-based katkısı (normalize)
        rule_conf = min(rule_score / 100, 1.0)
        if rule_direction == "up":
            up_score += rule_conf * self.rule_weight
        elif rule_direction == "down":
            down_score += rule_conf * self.rule_weight
        
        # Final karar
        if up_score > down_score + 0.1:  # Minimum fark eşiği
            final_direction = "up"
            final_confidence = up_score / (up_score + down_score + 0.0001)
        elif down_score > up_score + 0.1:
            final_direction = "down"
            final_confidence = down_score / (up_score + down_score + 0.0001)
        else:
            final_direction = "neutral"
            final_confidence = 0.5
        
        # Predicted change
        predicted_change = lstm_pred.predicted_change
        
        return final_direction, final_confidence, predicted_change
    
    def get_status(self) -> Dict:
        return {
            "lstm": self.lstm_predictor.get_status(),
            "lgbm_available": self.lgbm_available,
            "weights": {
                "lstm": self.lstm_weight,
                "lgbm": self.lgbm_weight,
                "rule": self.rule_weight
            }
        }


async def train_lstm_model(days: int = 5, symbols: List[str] = None):
    """
    LSTM modelini historical data ile eğit.
    
    Args:
        days: Kaç günlük veri kullanılacak
        symbols: Hangi coinler kullanılacak
    """
    import aiohttp
    from datetime import datetime, timedelta
    
    if symbols is None:
        # Volatil altcoinler - daha iyi egitim icin
        symbols = [
            "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
            "LINKUSDT", "MATICUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT"
        ]
    
    print("\n" + "="*60)
    print("     LSTM EĞİTİM VERİSİ TOPLAMA")
    print(f"     {days} gün, {len(symbols)} coin")
    print("="*60 + "\n")
    
    all_data = []
    
    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            print(f"[Training] {symbol} verisi çekiliyor...")
            
            url = "https://fapi.binance.com/fapi/v1/klines"
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            all_klines = []
            current_start = start_time
            
            while current_start < end_time:
                params = {
                    "symbol": symbol,
                    "interval": "5m",
                    "startTime": current_start,
                    "limit": 1000
                }
                
                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            klines = await resp.json()
                            if not klines:
                                break
                            all_klines.extend(klines)
                            current_start = klines[-1][0] + 1
                        else:
                            break
                except Exception as e:
                    print(f"[Training] Hata: {e}")
                    break
            
            if all_klines:
                df = pd.DataFrame(all_klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                df['symbol'] = symbol
                all_data.append(df)
                print(f"[Training] {symbol}: {len(df)} mum")
    
    if not all_data:
        print("[Training] Veri toplanamadı!")
        return None
    
    # Tüm veriyi birleştir
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n[Training] Toplam: {len(combined_df)} mum")
    
    # Model eğit
    predictor = LSTMPredictor()
    results = predictor.train(combined_df, epochs=30, batch_size=64)
    
    if results:
        print("\n" + "="*60)
        print("     EĞİTİM TAMAMLANDI!")
        print("="*60)
        print(f"   Best Accuracy: {results['best_accuracy']*100:.1f}%")
        print(f"   Final Val Accuracy: {results['val_accuracy']*100:.1f}%")
        print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    import asyncio
    
    # Test
    print("[Test] Deep Learning Predictor")
    
    predictor = LSTMPredictor()
    status = predictor.get_status()
    print(f"Model durumu: {json.dumps(status, indent=2)}")
    
    # Eğitim varsa test tahmini
    if predictor.is_trained:
        # Örnek DataFrame oluştur
        import numpy as np
        n = 100
        df_test = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='5min'),
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.rand(n) * 1000 + 500
        })
        
        pred = predictor.predict(df_test, "TEST")
        print(f"\nTahmin: {pred.direction} ({pred.confidence*100:.1f}%)")
        print(f"Olasılıklar: {pred.probabilities}")
