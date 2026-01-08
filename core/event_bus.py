# HFT Trading Tools - Event Bus
"""
Olay tabanlı iletişim sistemi.
Modüller arası veri akışını yönetir.
"""

import asyncio
from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(Enum):
    """Olay tipleri"""
    # Veri olayları
    TRADE = "trade"
    ORDERBOOK_UPDATE = "orderbook_update"
    LIQUIDATION = "liquidation"
    
    # Gösterge olayları
    CVD_UPDATE = "cvd_update"
    OBI_UPDATE = "obi_update"
    HEATMAP_UPDATE = "heatmap_update"
    
    # Sinyal olayları
    SIGNAL_LONG = "signal_long"
    SIGNAL_SHORT = "signal_short"
    SIGNAL_CLOSE = "signal_close"
    
    # Sistem olayları
    CONNECTION_UP = "connection_up"
    CONNECTION_DOWN = "connection_down"
    ERROR = "error"


@dataclass
class Event:
    """Olay veri yapısı"""
    type: EventType
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"


class EventBus:
    """
    Merkezi olay yönetim sistemi.
    Pub/Sub pattern ile modüller arası iletişim.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        # Singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not EventBus._initialized:
            self._subscribers: Dict[EventType, List[Callable]] = {}
            self._event_queue: asyncio.Queue = None
            self._running = False
            self._stats = {
                "events_published": 0,
                "events_processed": 0,
                "errors": 0
            }
            EventBus._initialized = True
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """
        Bir olay tipine abone ol.
        
        Args:
            event_type: Dinlenecek olay tipi
            callback: Olay geldiğinde çağrılacak fonksiyon
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Aboneliği iptal et"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)
    
    async def publish(self, event: Event):
        """
        Olay yayınla.
        
        Args:
            event: Yayınlanacak olay
        """
        self._stats["events_published"] += 1
        
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                    self._stats["events_processed"] += 1
                except Exception as e:
                    self._stats["errors"] += 1
                    print(f"[EventBus] Hata: {event.type.value} - {e}")
    
    def publish_sync(self, event: Event):
        """Senkron olay yayını (async olmayan ortamlar için)"""
        self._stats["events_published"] += 1
        
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    if not asyncio.iscoroutinefunction(callback):
                        callback(event)
                        self._stats["events_processed"] += 1
                except Exception as e:
                    self._stats["errors"] += 1
                    print(f"[EventBus] Hata: {event.type.value} - {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """İstatistikleri döndür"""
        return self._stats.copy()
    
    def reset(self):
        """Event bus'ı sıfırla (test için)"""
        self._subscribers.clear()
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "errors": 0
        }


# Global event bus instance
event_bus = EventBus()
