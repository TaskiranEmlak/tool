# Trading Brain Web Dashboard - FastAPI + WebSocket Backend
"""
Real-time trading brain dashboard with WebSocket support.
Tarayıcıda açılır, anlık güncelleme.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Set
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
import aiohttp
import numpy as np

from core.trading_brain import get_brain, TradingBrain, TradeResult
from core.smart_watchlist import get_watchlist, SmartWatchlist
from core.scanner import VolatilityScanner
from core.predictor import Predictor


# ============ GLOBALS ============
brain: TradingBrain = None
watchlist: SmartWatchlist = None
scanner: VolatilityScanner = None
predictor: Predictor = None

# Active signals and websocket connections
active_signals: Dict[str, Dict] = {}
closed_trades: List[Dict] = []
connected_clients: Set[WebSocket] = set()
is_running = False


# ============ LIFESPAN ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain, watchlist, scanner, predictor
    
    print("🧠 Trading Brain Web Dashboard başlatılıyor...")
    
    brain = get_brain()
    watchlist = get_watchlist()
    scanner = VolatilityScanner()
    predictor = Predictor()
    
    print(f"✅ Brain yüklendi - {brain.stats['total_trades']} işlem hafızada")
    
    yield
    
    print("👋 Kapatılıyor...")


# ============ APP ============
app = FastAPI(title="Trading Brain Dashboard", lifespan=lifespan)

# Get absolute paths based on this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Create directories if needed
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "sounds"), exist_ok=True)

# Static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ============ HELPERS ============
def calc_rsi(closes, period=14):
    """RSI hesapla"""
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    # DÜZELTME: Her iki ortalama da 0 ise (yatay piyasa) RSI 50
    if avg_loss == 0 and avg_gain == 0:
        return 50
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


async def get_market_sentiment() -> dict:
    """
    Piyasa durumu analizi - Binance API kullanarak
    BTC trendi, altcoin performansı, hacim analizi
    """
    try:
        async with aiohttp.ClientSession() as session:
            # 24h değişimler
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            async with session.get(url) as resp:
                tickers = await resp.json()
            
            # BTC analizi
            btc = next((t for t in tickers if t['symbol'] == 'BTCUSDT'), None)
            btc_change = float(btc['priceChangePercent']) if btc else 0
            
            # ETH analizi
            eth = next((t for t in tickers if t['symbol'] == 'ETHUSDT'), None)
            eth_change = float(eth['priceChangePercent']) if eth else 0
            
            # Altcoin analizi (USDT pairs, top 50 by volume)
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') 
                         and t['symbol'] not in ['BTCUSDT', 'ETHUSDT']]
            usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            top_alts = usdt_pairs[:50]
            
            positive_alts = sum(1 for t in top_alts if float(t['priceChangePercent']) > 0)
            negative_alts = len(top_alts) - positive_alts
            avg_alt_change = sum(float(t['priceChangePercent']) for t in top_alts) / len(top_alts) if top_alts else 0
            
            # Top gainers & losers - DÜZELTME: boş liste kontrolü
            if top_alts:
                top_gainer = max(top_alts, key=lambda x: float(x['priceChangePercent']))
                top_loser = min(top_alts, key=lambda x: float(x['priceChangePercent']))
            else:
                top_gainer = {'symbol': 'N/A', 'priceChangePercent': 0}
                top_loser = {'symbol': 'N/A', 'priceChangePercent': 0}
            
            # Piyasa durumu belirleme
            if btc_change > 2 and positive_alts > negative_alts:
                mood = "🟢 YUKARIŞ (BULL)"
                description = f"BTC +{btc_change:.1f}% ile yükselişte. Alıcı baskısı hakim."
            elif btc_change < -2 and negative_alts > positive_alts:
                mood = "🔴 DÜŞÜŞ (BEAR)"
                description = f"BTC {btc_change:.1f}% ile düşüşte. Satıcı baskısı hakim."
            elif abs(btc_change) < 1:
                mood = "🟡 YATAY (SIDEWAYS)"
                description = f"BTC yatay seyrediyor. Kararsız piyasa."
            else:
                mood = "🟠 KARIŞIK"
                description = f"BTC {btc_change:+.1f}%, piyasa karışık sinyaller veriyor."
            
            # Altcoin sezonu mu?
            if avg_alt_change > btc_change + 2:
                alt_season = "🚀 ALT SEASON!"
            elif avg_alt_change < btc_change - 2:
                alt_season = "₿ BTC Dominance yükseliyor"
            else:
                alt_season = "Dengeli piyasa"
            
            return {
                'mood': mood,
                'description': description,
                'btc_change': btc_change,
                'eth_change': eth_change,
                'alt_avg_change': avg_alt_change,
                'positive_alts': positive_alts,
                'negative_alts': negative_alts,
                'alt_season': alt_season,
                'top_gainer': {
                    'symbol': top_gainer['symbol'].replace('USDT', ''),
                    'change': float(top_gainer['priceChangePercent'])
                },
                'top_loser': {
                    'symbol': top_loser['symbol'].replace('USDT', ''),
                    'change': float(top_loser['priceChangePercent'])
                }
            }
    except Exception as e:
        print(f"[Market] Sentiment error: {e}")
        return {
            'mood': '⚪ Veri alınamadı',
            'description': 'Piyasa verisi yüklenemedi',
            'btc_change': 0,
            'eth_change': 0
        }


async def broadcast(event: str, data: dict):
    """Tüm clientlara mesaj gönder"""
    message = json.dumps({"event": event, "data": data})
    disconnected = set()
    
    # DÜZELTME: Race condition önlemek için copy kullan
    for client in list(connected_clients):
        try:
            await client.send_text(message)
        except:
            disconnected.add(client)
    
    connected_clients.difference_update(disconnected)


async def get_current_prices() -> Dict[str, float]:
    """Tüm fiyatları çek"""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            async with session.get(url) as resp:
                prices = await resp.json()
                return {p['symbol']: float(p['price']) for p in prices}
    except:
        return {}


async def update_pnl():
    """Aktif sinyallerin P&L güncellemesi"""
    global active_signals, closed_trades
    
    if not active_signals:
        return
    
    prices = await get_current_prices()
    total_pnl = 0
    signals_to_close = []
    
    for symbol, info in list(active_signals.items()):
        if symbol not in prices:
            continue
        
        current_price = prices[symbol]
        entry = info['entry']
        direction = info['direction']
        sl = info['sl']
        tp = info['tp']
        
        # PnL hesapla
        if direction == "LONG":
            pnl_pct = ((current_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - current_price) / entry) * 100
        
        total_pnl += pnl_pct
        
        # Güncelle
        info['current_price'] = current_price
        info['pnl'] = pnl_pct
        
        # SL/TP kontrolü
        hit_sl = (direction == "LONG" and current_price <= sl) or \
                 (direction == "SHORT" and current_price >= sl)
        hit_tp = (direction == "LONG" and current_price >= tp) or \
                 (direction == "SHORT" and current_price <= tp)
        
        if hit_tp or hit_sl:
            info['status'] = 'TP_HIT' if hit_tp else 'SL_HIT'
            info['closed_at'] = datetime.now().isoformat()
            signals_to_close.append(symbol)
            
            # Brain'e öğret
            result = TradeResult(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                exit_price=current_price,
                entry_time=datetime.fromisoformat(info['opened_at']),
                exit_time=datetime.now(),
                pnl_percent=pnl_pct,
                is_win=hit_tp,
                rsi=info.get('rsi', 50),
                volume_ratio=info.get('volume_ratio', 1.0),
                trend="up" if direction == "LONG" else "down",
                score=info.get('confidence', 70)
            )
            brain.learn(result)
            
            # Closed trades'e ekle
            closed_trades.append({
                'symbol': symbol,
                'direction': direction,
                'pnl': pnl_pct,
                'status': info['status'],
                'closed_at': info['closed_at']
            })
            closed_trades = closed_trades[-50:]  # Son 50
            
            # Broadcast
            await broadcast("trade_closed", {
                'symbol': symbol,
                'pnl': pnl_pct,
                'status': info['status']
            })
    
    # Kapananları sil
    for symbol in signals_to_close:
        del active_signals[symbol]
    
    return total_pnl


async def scan_loop():
    """Ana tarama döngüsü"""
    global is_running, active_signals
    
    while is_running:
        try:
            # Scanner ile coin bul
            results = await scanner.scan_1m_volatility()
            
            if results:
                async with aiohttp.ClientSession() as session:
                    for coin in results[:15]:
                        try:
                            # Klines al
                            kline_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={coin.symbol}&interval=5m&limit=30"
                            async with session.get(kline_url) as resp:
                                klines = await resp.json()
                                closes = [float(k[4]) for k in klines]
                                volumes = [float(k[5]) for k in klines]
                                
                                rsi = calc_rsi(closes)
                                avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1
                                vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
                            
                            # Order book
                            depth_url = f"https://fapi.binance.com/fapi/v1/depth?symbol={coin.symbol}&limit=10"
                            async with session.get(depth_url) as resp:
                                data = await resp.json()
                                bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
                                asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
                            
                            if bids and asks:
                                pred = await predictor.analyze(coin.symbol, coin.price, bids, asks)
                                
                                # === VOTING TABANLI YÖN BELİRLEME ===
                                # Her sinyal kaynağı oy kullanır, çoğunluk kazanır
                                votes = {"LONG": 0, "SHORT": 0}
                                
                                # 1. RSI oyu (2 oy - güçlü sinyal)
                                if rsi < 30:
                                    votes["LONG"] += 2
                                elif rsi < 40:
                                    votes["LONG"] += 1
                                elif rsi > 70:
                                    votes["SHORT"] += 2
                                elif rsi > 60:
                                    votes["SHORT"] += 1
                                
                                # 2. Predictor oyu (2 oy - AI analizi)
                                if pred.predicted_direction == "up":
                                    votes["LONG"] += 2
                                elif pred.predicted_direction == "down":
                                    votes["SHORT"] += 2
                                
                                # 3. TF uyumu oyu (3 oy - en güçlü sinyal)
                                if pred.tf_alignment:
                                    if pred.tf_5m_trend == "up":
                                        votes["LONG"] += 3
                                    elif pred.tf_5m_trend == "down":
                                        votes["SHORT"] += 3
                                
                                # 4. Market sentiment oyu (1 oy)
                                if pred.market_score > 60:
                                    votes["LONG"] += 1
                                elif pred.market_score < 40:
                                    votes["SHORT"] += 1
                                
                                # Yön belirleme - en az 2 oy farkı gerekli
                                vote_diff = abs(votes["LONG"] - votes["SHORT"])
                                if vote_diff < 2:
                                    # Yetersiz konsensüs - sinyal verme
                                    direction = None
                                    trend = "sideways"
                                elif votes["LONG"] > votes["SHORT"]:
                                    direction = "LONG"
                                    trend = "up"
                                else:
                                    direction = "SHORT"
                                    trend = "down"
                                
                                
                                # Yetersiz konsensüs - bu coini atla
                                if direction is None:
                                    continue
                                
                                # Beyin kararı - DÜZELTME: current_price eklendi
                                decision = brain.decide(
                                    symbol=coin.symbol,
                                    rsi=rsi,
                                    volume_ratio=vol_ratio,
                                    trend=trend,
                                    score=pred.total_score,
                                    direction=direction,
                                    current_price=coin.price  # Shadow signal P&L için kritik
                                )
                                
                                if decision.action == "SIGNAL" and coin.symbol not in active_signals:
                                    # Yeni sinyal! - ANINDA bildir
                                    entry = coin.price
                                    sl = entry * (0.99 if direction == "LONG" else 1.01)
                                    tp = entry * (1.02 if direction == "LONG" else 0.98)
                                    
                                    signal_data = {
                                        'symbol': coin.symbol,
                                        'direction': direction,
                                        'entry': entry,
                                        'current_price': entry,
                                        'sl': sl,
                                        'tp': tp,
                                        'pnl': 0,
                                        'confidence': decision.confidence,
                                        'reasons': decision.reasons,
                                        'rsi': rsi,
                                        'volume_ratio': vol_ratio,
                                        'opened_at': datetime.now().isoformat(),
                                        'status': 'OPEN'
                                    }
                                    
                                    active_signals[coin.symbol] = signal_data
                                    
                                    # ANINDA konsola ve UI'a bildir
                                    print(f"🚀 SİNYAL: {coin.symbol} {direction} @{entry:.4f} | Güven: {decision.confidence:.0f}%")
                                    
                                    # Hemen broadcast et
                                    await broadcast("new_signal", signal_data)
                                    await broadcast("signals_update", {
                                        'signals': list(active_signals.values()),
                                        'total_pnl': 0,
                                        'count': len(active_signals)
                                    })
                                
                                elif decision.action == "WATCH":
                                    # Shadow signal için fiyat kaydet
                                    brain.track_shadow_signal(coin.symbol, coin.price)
                                    
                                    watchlist.add(
                                        symbol=coin.symbol,
                                        direction=direction,
                                        current_price=coin.price,
                                        current_rsi=rsi,
                                        current_volume=vol_ratio,
                                        score=pred.total_score,
                                        market_score=pred.market_score
                                    )
                                
                                elif decision.action == "SKIP":
                                    # Shadow signal için fiyat kaydet
                                    brain.track_shadow_signal(coin.symbol, coin.price)
                        except Exception as e:
                            continue
            
            # P&L güncelle
            total_pnl = await update_pnl()
            
            # Shadow signals kontrol et (kaçırılan fırsatlar)
            prices = await get_current_prices()
            brain.check_shadow_signals(prices)
            
            # Broadcast updates
            await broadcast("signals_update", {
                'signals': list(active_signals.values()),
                'total_pnl': total_pnl,
                'count': len(active_signals)
            })
            
            await broadcast("watchlist_update", {
                'items': watchlist.get_display_data()
            })
            
            await broadcast("brain_status", brain.get_status())
            
            # Market sentiment (her 5 döngüde bir)
            if not hasattr(scan_loop, 'sentiment_counter'):
                scan_loop.sentiment_counter = 0
            scan_loop.sentiment_counter += 1
            if scan_loop.sentiment_counter >= 5:
                sentiment = await get_market_sentiment()
                await broadcast("market_sentiment", sentiment)
                scan_loop.sentiment_counter = 0
            
            await broadcast("closed_trades", {
                'trades': closed_trades[-10:]
            })
            
            await asyncio.sleep(3)  # Daha hızlı güncelleme
            
        except Exception as e:
            print(f"[Brain] Scan error: {e}")
            await asyncio.sleep(5)


# ============ ROUTES ============
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Ana dashboard sayfası"""
    return templates.TemplateResponse("brain_dashboard.html", {
        "request": request,
        "brain_status": brain.get_status() if brain else {}
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket bağlantısı"""
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🔌 Yeni client bağlandı. Toplam: {len(connected_clients)}")
    
    try:
        # İlk verileri gönder
        await websocket.send_text(json.dumps({
            "event": "init",
            "data": {
                "signals": list(active_signals.values()),
                "watchlist": watchlist.get_display_data() if watchlist else [],
                "brain_status": brain.get_status() if brain else {},
                "closed_trades": closed_trades[-10:]
            }
        }))
        
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get("action") == "start":
                global is_running
                if not is_running:
                    is_running = True
                    asyncio.create_task(scan_loop())
                    await websocket.send_text(json.dumps({"event": "status", "data": "started"}))
            
            elif msg.get("action") == "stop":
                is_running = False
                await websocket.send_text(json.dumps({"event": "status", "data": "stopped"}))
            
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        print(f"🔌 Client ayrıldı. Kalan: {len(connected_clients)}")


@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "brain": brain.get_status() if brain else {},
        "active_signals": len(active_signals),
        "watchlist_count": len(watchlist) if watchlist else 0,
        "is_running": is_running
    }


# ============ MAIN ============
if __name__ == "__main__":
    import uvicorn
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🧠 TRADING BRAIN - WEB DASHBOARD                       ║
    ║   ─────────────────────────────────                       ║
    ║                                                           ║
    ║   🌐 http://localhost:8000                                ║
    ║   📊 Real-time WebSocket updates                          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
