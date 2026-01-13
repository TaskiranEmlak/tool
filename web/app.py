# Professional Signal Dashboard - Web App
"""
FastAPI tabanlı profesyonel sinyal dashboard.
30 dakika timeframe, Binance Futures sinyalleri.
"""

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "sounds").mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(
    title="Kriptol Signal Station",
    description="Profesyonel Binance Futures Sinyal Sistemi",
    version="2.0.0"
)

# Static files & templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Signal generator instance
signal_generator = None
signal_task = None


def get_signal_gen():
    """Lazy load signal generator"""
    global signal_generator
    if signal_generator is None:
        from core.signal_generator import get_signal_generator
        signal_generator = get_signal_generator()
    return signal_generator


# ============ SAYFA ROUTE'LARI ============

@app.get("/", response_class=HTMLResponse)
async def signal_dashboard(request: Request):
    """Profesyonel sinyal dashboard"""
    return templates.TemplateResponse("signal_dashboard.html", {
        "request": request
    })


@app.get("/old-dashboard", response_class=HTMLResponse)
async def old_dashboard(request: Request):
    """Eski dashboard (opsiyonel)"""
    try:
        from core.consensus import get_consensus_engine
        consensus = get_consensus_engine()
        data = consensus.get_dashboard_data()
    except:
        data = {}
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "data": data,
        "now": datetime.now()
    })


# ============ API ROUTE'LARI ============

@app.get("/api/signals")
async def api_signals():
    """Aktif sinyaller - Dashboard için ana endpoint"""
    gen = get_signal_gen()
    
    try:
        # Sinyalleri üret
        signals = await gen.generate_signals("30m")  # 30 dakika timeframe
        
        # Format signals for frontend
        formatted = []
        for s in signals:
            formatted.append({
                "symbol": s.symbol,
                "type": s.signal_type.value,
                "entry": s.entry_price,
                "sl": s.stop_loss,
                "tp": s.take_profit,
                "confidence": int(s.confidence),
                "rsi": s.indicators.get('rsi', 0),
                "reason": s.reason,
                "timestamp": s.timestamp.isoformat()
            })
        
        # Top volatile coins - directly from fetcher
        top_coins = []
        try:
            volatiles = await gen.data_fetcher.fetch_top_volatile_coins(10)
            top_coins = volatiles  # Already has symbol and change
        except:
            pass
        
        return {
            "signals": formatted,
            "top_coins": top_coins,
            "coin_count": len(gen.data_fetcher.SYMBOLS),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "signals": [],
            "top_coins": [],
            "coin_count": 0,
            "error": str(e)
        }


@app.get("/api/scan")
async def api_scan():
    """Manuel tarama tetikle"""
    gen = get_signal_gen()
    
    # Coin listesini güncelle
    await gen.data_fetcher.update_coin_list(limit=50)
    
    # Sinyal tara
    signals = await gen.generate_signals("30m")
    
    return {
        "success": True,
        "signal_count": len(signals),
        "coin_count": len(gen.data_fetcher.SYMBOLS)
    }


@app.get("/api/stats")
async def api_stats():
    """Strateji istatistikleri"""
    from core.proven_strategy import get_strategy
    strategy = get_strategy()
    
    return {
        "rsi_oversold": strategy.RSI_OVERSOLD,
        "rsi_overbought": strategy.RSI_OVERBOUGHT,
        "stop_loss_pct": strategy.STOP_LOSS_PCT * 100,
        "take_profit_pct": strategy.TAKE_PROFIT_PCT * 100,
        "min_score": strategy.MIN_SIGNAL_SCORE,
        "timeframe": "30m"
    }


@app.post("/api/signal-loop/start")
async def start_signal_loop():
    """Sürekli sinyal tarama başlat"""
    global signal_task
    gen = get_signal_gen()
    
    if signal_task is None or signal_task.done():
        signal_task = asyncio.create_task(gen.run_loop())
        return {"status": "started"}
    
    return {"status": "already_running"}


@app.post("/api/signal-loop/stop")
async def stop_signal_loop():
    """Sinyal tarama durdur"""
    gen = get_signal_gen()
    gen.stop()
    return {"status": "stopped"}


# ============ BAŞLATMA ============

def run_dashboard(host: str = "127.0.0.1", port: int = 8000):
    """Dashboard'u başlat"""
    print(f"\n🚀 Kriptol Signal Station başlatılıyor...")
    print(f"📡 http://{host}:{port}")
    print("-" * 40)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()

