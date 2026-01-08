# HFT Trading Tools - WebSocket Data Collector
"""
Binance WebSocket üzerinden canlı veri toplama modülü.
Trade, Order Book ve Likidasyon verilerini yakalar.
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from core.event_bus import EventBus, Event, EventType, event_bus
from core.database import Database, Trade, OrderBookSnapshot, Liquidation
from config.settings import settings


@dataclass
class OrderBook:
    """Canlı order book verisi"""
    symbol: str
    bids: Dict[float, float]  # {price: quantity}
    asks: Dict[float, float]
    last_update_id: int = 0
    last_update_time: datetime = None


class DataCollector:
    """
    Çoklu WebSocket bağlantı yöneticisi.
    Veri akışını Event Bus'a ve Database'e yönlendirir.
    """
    
    def __init__(self, symbols: List[str] = None, use_database: bool = True):
        self.symbols = symbols or settings.SYMBOLS
        self.db = Database() if use_database else None
        self.event_bus = event_bus
        
        # Bağlantı durumu
        self._running = False
        self._connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._tasks: List[asyncio.Task] = []
        
        # Canlı veri
        self.order_books: Dict[str, OrderBook] = {}
        self.latest_prices: Dict[str, float] = {}
        self.trade_buffer: List[Trade] = []
        
        # İstatistikler
        self.stats = {
            "trades_received": 0,
            "orderbook_updates": 0,
            "liquidations_received": 0,
            "reconnects": 0,
            "errors": 0
        }
        
        # Callback'ler
        self._on_trade_callbacks: List[Callable] = []
        self._on_orderbook_callbacks: List[Callable] = []
        self._on_liquidation_callbacks: List[Callable] = []
    
    # ============ Callback Yönetimi ============
    
    def on_trade(self, callback: Callable):
        """Trade callback ekle"""
        self._on_trade_callbacks.append(callback)
    
    def on_orderbook(self, callback: Callable):
        """Order book callback ekle"""
        self._on_orderbook_callbacks.append(callback)
    
    def on_liquidation(self, callback: Callable):
        """Likidasyon callback ekle"""
        self._on_liquidation_callbacks.append(callback)
    
    # ============ Veri İşleme ============
    
    def _process_trade(self, data: dict):
        """
        Binance aggTrade mesajını işle.
        Format: {"e":"aggTrade","E":1234,"s":"BTCUSDT","p":"50000.00",...}
        """
        try:
            symbol = data.get("s", "")
            price = float(data.get("p", 0))
            quantity = float(data.get("q", 0))
            is_buyer_maker = data.get("m", False)
            timestamp = datetime.fromtimestamp(data.get("T", 0) / 1000)
            
            side = "sell" if is_buyer_maker else "buy"
            
            trade = Trade(
                symbol=symbol,
                price=price,
                quantity=quantity,
                side=side,
                timestamp=timestamp,
                exchange="binance"
            )
            
            # Fiyatı güncelle
            self.latest_prices[symbol] = price
            
            # Veritabanına ekle (buffer ile)
            self.trade_buffer.append(trade)
            if len(self.trade_buffer) >= 50:  # 50 trade'de bir flush
                if self.db:
                    self.db.insert_trades_batch(self.trade_buffer)
                self.trade_buffer.clear()
            
            # Event yayınla
            self.event_bus.publish_sync(Event(
                type=EventType.TRADE,
                symbol=symbol,
                data={
                    "price": price,
                    "quantity": quantity,
                    "side": side,
                    "value": price * quantity
                },
                timestamp=timestamp
            ))
            
            # Callback'leri çağır
            for callback in self._on_trade_callbacks:
                try:
                    callback(trade)
                except Exception as e:
                    print(f"[DataCollector] Trade callback hatası: {e}")
            
            self.stats["trades_received"] += 1
            
        except Exception as e:
            self.stats["errors"] += 1
            print(f"[DataCollector] Trade işleme hatası: {e}")
    
    def _process_orderbook(self, data: dict):
        """
        Binance depth mesajını işle.
        Format: {"e":"depthUpdate","s":"BTCUSDT","b":[[price,qty],...],"a":[[price,qty],...]}
        """
        try:
            symbol = data.get("s", "")
            bids = data.get("b", []) or data.get("bids", [])
            asks = data.get("a", []) or data.get("asks", [])
            
            if symbol not in self.order_books:
                self.order_books[symbol] = OrderBook(
                    symbol=symbol,
                    bids={},
                    asks={}
                )
            
            ob = self.order_books[symbol]
            
            # Bids güncelle
            for bid in bids:
                price = float(bid[0])
                qty = float(bid[1])
                if qty == 0:
                    ob.bids.pop(price, None)
                else:
                    ob.bids[price] = qty
            
            # Asks güncelle
            for ask in asks:
                price = float(ask[0])
                qty = float(ask[1])
                if qty == 0:
                    ob.asks.pop(price, None)
                else:
                    ob.asks[price] = qty
            
            ob.last_update_time = datetime.now()
            
            # En iyi 10 seviyeyi al
            sorted_bids = sorted(ob.bids.items(), reverse=True)[:10]
            sorted_asks = sorted(ob.asks.items())[:10]
            
            # Event yayınla
            self.event_bus.publish_sync(Event(
                type=EventType.ORDERBOOK_UPDATE,
                symbol=symbol,
                data={
                    "bids": sorted_bids,
                    "asks": sorted_asks,
                    "best_bid": sorted_bids[0] if sorted_bids else (0, 0),
                    "best_ask": sorted_asks[0] if sorted_asks else (0, 0),
                    "spread": (sorted_asks[0][0] - sorted_bids[0][0]) if sorted_bids and sorted_asks else 0
                },
                timestamp=ob.last_update_time
            ))
            
            # Callback'leri çağır
            for callback in self._on_orderbook_callbacks:
                try:
                    callback(symbol, sorted_bids, sorted_asks)
                except Exception as e:
                    print(f"[DataCollector] Orderbook callback hatası: {e}")
            
            self.stats["orderbook_updates"] += 1
            
        except Exception as e:
            self.stats["errors"] += 1
            print(f"[DataCollector] Orderbook işleme hatası: {e}")
    
    def _process_liquidation(self, data: dict):
        """
        Binance forceOrder mesajını işle.
        """
        try:
            order = data.get("o", {})
            symbol = order.get("s", "")
            side = "long" if order.get("S", "") == "SELL" else "short"
            price = float(order.get("p", 0))
            quantity = float(order.get("q", 0))
            timestamp = datetime.fromtimestamp(order.get("T", 0) / 1000)
            
            liq = Liquidation(
                symbol=symbol,
                side=side,
                price=price,
                quantity=quantity,
                timestamp=timestamp,
                exchange="binance"
            )
            
            # Veritabanına ekle
            if self.db:
                self.db.insert_liquidation(liq)
            
            # Event yayınla
            self.event_bus.publish_sync(Event(
                type=EventType.LIQUIDATION,
                symbol=symbol,
                data={
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                    "value": price * quantity
                },
                timestamp=timestamp
            ))
            
            # Callback'leri çağır
            for callback in self._on_liquidation_callbacks:
                try:
                    callback(liq)
                except Exception as e:
                    print(f"[DataCollector] Likidasyon callback hatası: {e}")
            
            self.stats["liquidations_received"] += 1
            
        except Exception as e:
            self.stats["errors"] += 1
            print(f"[DataCollector] Likidasyon işleme hatası: {e}")
    
    # ============ WebSocket Bağlantıları ============
    
    async def _connect_trades(self, symbol: str):
        """Trade stream'e bağlan"""
        url = f"{settings.EXCHANGES['binance']['ws_base']}/{symbol.lower()}@aggTrade"
        
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._connections[f"trade_{symbol}"] = ws
                    print(f"[DataCollector] Trade bağlantısı kuruldu: {symbol}")
                    
                    await self.event_bus.publish(Event(
                        type=EventType.CONNECTION_UP,
                        symbol=symbol,
                        data={"stream": "trade"}
                    ))
                    
                    async for message in ws:
                        if not self._running:
                            break
                        data = json.loads(message)
                        self._process_trade(data)
                        
            except websockets.ConnectionClosed:
                self.stats["reconnects"] += 1
                print(f"[DataCollector] Trade bağlantısı koptu, yeniden bağlanılıyor: {symbol}")
                await asyncio.sleep(1)
            except Exception as e:
                self.stats["errors"] += 1
                print(f"[DataCollector] Trade bağlantı hatası {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _connect_orderbook(self, symbol: str):
        """Order book stream'e bağlan"""
        depth_levels = min(settings.OBI_DEPTH, 20)  # Max 20 seviye
        url = f"{settings.EXCHANGES['binance']['ws_base']}/{symbol.lower()}@depth{depth_levels}@100ms"
        
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._connections[f"depth_{symbol}"] = ws
                    print(f"[DataCollector] Order book bağlantısı kuruldu: {symbol}")
                    
                    async for message in ws:
                        if not self._running:
                            break
                        data = json.loads(message)
                        self._process_orderbook(data)
                        
            except websockets.ConnectionClosed:
                self.stats["reconnects"] += 1
                print(f"[DataCollector] Order book bağlantısı koptu: {symbol}")
                await asyncio.sleep(1)
            except Exception as e:
                self.stats["errors"] += 1
                print(f"[DataCollector] Order book hatası {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _connect_liquidations(self):
        """Likidasyon stream'e bağlan"""
        url = settings.EXCHANGES['binance']['ws_futures'] + "/!forceOrder@arr"
        
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._connections["liquidations"] = ws
                    print("[DataCollector] Likidasyon bağlantısı kuruldu")
                    
                    async for message in ws:
                        if not self._running:
                            break
                        data = json.loads(message)
                        self._process_liquidation(data)
                        
            except websockets.ConnectionClosed:
                self.stats["reconnects"] += 1
                print("[DataCollector] Likidasyon bağlantısı koptu")
                await asyncio.sleep(1)
            except Exception as e:
                self.stats["errors"] += 1
                print(f"[DataCollector] Likidasyon hatası: {e}")
                await asyncio.sleep(5)
    
    # ============ Kontrol ============
    
    async def start(self):
        """Veri toplamayı başlat"""
        if self._running:
            return
        
        self._running = True
        print(f"[DataCollector] Başlatılıyor... Semboller: {self.symbols}")
        
        # Her sembol için trade ve order book stream'leri
        for symbol in self.symbols:
            self._tasks.append(asyncio.create_task(self._connect_trades(symbol)))
            self._tasks.append(asyncio.create_task(self._connect_orderbook(symbol)))
        
        # Likidasyon stream
        self._tasks.append(asyncio.create_task(self._connect_liquidations()))
        
        print(f"[DataCollector] {len(self._tasks)} stream başlatıldı")
    
    async def stop(self):
        """Veri toplamayı durdur"""
        self._running = False
        
        # Bağlantıları kapat
        for name, ws in self._connections.items():
            try:
                await ws.close()
            except:
                pass
        
        # Task'ları iptal et
        for task in self._tasks:
            task.cancel()
        
        # Buffer'ı flush et
        if self.trade_buffer and self.db:
            self.db.insert_trades_batch(self.trade_buffer)
            self.trade_buffer.clear()
        
        print("[DataCollector] Durduruldu")
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """Sembol için order book döndür"""
        return self.order_books.get(symbol)
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Son fiyatı döndür"""
        return self.latest_prices.get(symbol)
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        return {
            **self.stats,
            "active_connections": len(self._connections),
            "symbols": self.symbols,
            "running": self._running
        }


# Yardımcı fonksiyon: Kısa süreli test için
async def test_collector(duration: int = 10):
    """Data collector test"""
    collector = DataCollector(symbols=["BTCUSDT"], use_database=False)
    
    # Trade callback
    def on_trade(trade: Trade):
        print(f"Trade: {trade.symbol} {trade.side.upper()} {trade.quantity:.4f} @ ${trade.price:,.2f}")
    
    collector.on_trade(on_trade)
    
    await collector.start()
    await asyncio.sleep(duration)
    await collector.stop()
    
    print(f"\nİstatistikler: {collector.get_stats()}")


if __name__ == "__main__":
    print("Data Collector Test (10 saniye)")
    asyncio.run(test_collector(10))
