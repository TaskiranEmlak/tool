# HFT Trading Tools - Benford Yasası Wash Trading Filtresi
"""
Benford Yasası ile sahte hacim (Wash Trading) tespiti.
"""

import math
from typing import List, Tuple
from collections import Counter


# Benford Yasası referans dağılımı
BENFORD_DISTRIBUTION = {
    1: 0.301,
    2: 0.176,
    3: 0.125,
    4: 0.097,
    5: 0.079,
    6: 0.067,
    7: 0.058,
    8: 0.051,
    9: 0.046
}


def get_leading_digit(number: float) -> int:
    """Sayının ilk basamağını al"""
    if number <= 0:
        return 0
    
    # Normalize et
    while number >= 10:
        number /= 10
    while number < 1:
        number *= 10
    
    return int(number)


def calculate_benford_score(volumes: List[float]) -> Tuple[float, bool]:
    """
    Hacim listesinin Benford Yasası'na uyumunu hesapla.
    
    Args:
        volumes: Hacim listesi
        
    Returns:
        (score, is_organic) - score: 0-1 arası (yüksek = organik), is_organic: bool
    """
    if len(volumes) < 100:
        return 0.5, True  # Yetersiz veri
    
    # İlk basamakları çıkar
    leading_digits = [get_leading_digit(v) for v in volumes if v > 0]
    
    if len(leading_digits) < 50:
        return 0.5, True
    
    # Dağılımı hesapla
    total = len(leading_digits)
    observed = Counter(leading_digits)
    
    # Chi-square benzeri sapma hesapla
    total_deviation = 0.0
    
    for digit in range(1, 10):
        expected = BENFORD_DISTRIBUTION[digit]
        observed_freq = observed.get(digit, 0) / total
        deviation = abs(observed_freq - expected)
        total_deviation += deviation
    
    # Normalize et (max sapma ~1.4)
    score = 1.0 - min(total_deviation / 1.4, 1.0)
    
    # Eşik: 0.6'nın altı şüpheli
    is_organic = score > 0.6
    
    return score, is_organic


class WashTradingFilter:
    """
    Wash Trading tespit filtresi.
    """
    
    def __init__(self):
        self._volume_buffer = {}  # symbol -> [volumes]
        self._scores = {}  # symbol -> (score, is_organic)
        self.buffer_size = 1000
    
    def add_volume(self, symbol: str, volume: float):
        """Hacim ekle"""
        if symbol not in self._volume_buffer:
            self._volume_buffer[symbol] = []
        
        self._volume_buffer[symbol].append(volume)
        
        # Buffer boyutunu sınırla
        if len(self._volume_buffer[symbol]) > self.buffer_size:
            self._volume_buffer[symbol] = self._volume_buffer[symbol][-self.buffer_size:]
        
        # Her 100 hacimde bir skoru güncelle
        if len(self._volume_buffer[symbol]) % 100 == 0:
            self._scores[symbol] = calculate_benford_score(self._volume_buffer[symbol])
    
    def get_score(self, symbol: str) -> Tuple[float, bool]:
        """Sembol için Benford skoru"""
        return self._scores.get(symbol, (0.5, True))
    
    def is_suspicious(self, symbol: str) -> bool:
        """Şüpheli mi?"""
        score, is_organic = self.get_score(symbol)
        return not is_organic
