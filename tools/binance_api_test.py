# Binance Copy Trading API Test
"""
Binance Copy Trading API'sini test eder.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import sys
import os
from typing import Dict, List
from urllib.parse import urlencode

# Path fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config'den API key'leri al
try:
    from config.api_keys import API_KEY, API_SECRET
    print(f"[OK] API Key yuklendi: {API_KEY[:10]}...")
except ImportError as e:
    print(f"[HATA] Import hatasi: {e}")
    API_KEY = ""
    API_SECRET = ""

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"  # Futures API farkli URL

def create_signature(params: dict, secret: str) -> str:
    """HMAC SHA256 signature olustur"""
    query_string = urlencode(params)
    signature = hmac.new(
        secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

async def binance_request(endpoint: str, params: dict = None, signed: bool = True, futures: bool = False) -> dict:
    """Binance API request"""
    if params is None:
        params = {}
    
    headers = {
        "X-MBX-APIKEY": API_KEY
    }
    
    if signed:
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = create_signature(params, API_SECRET)
    
    # Futures veya Spot URL sec
    base_url = BINANCE_FUTURES_URL if futures else BINANCE_BASE_URL
    url = f"{base_url}{endpoint}"
    query_string = urlencode(params)
    full_url = f"{url}?{query_string}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(full_url, headers=headers) as resp:
            print(f"\n[{resp.status}] {endpoint}")
            
            try:
                data = await resp.json()
            except:
                text = await resp.text()
                data = {"error": text[:200]}
            
            return {"status": resp.status, "data": data}

async def main():
    print("=" * 60)
    print("BINANCE COPY TRADING API TESTI")
    print("=" * 60)
    
    if not API_KEY:
        print("[HATA] API Key bulunamadi!")
        return
    
    print(f"\nAPI Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    
    # 1. Hesap bilgisi
    print("\n[1] Hesap Bilgisi...")
    result = await binance_request("/api/v3/account")
    if result["status"] == 200:
        data = result["data"]
        print(f"  Hesap Tipi: {data.get('accountType', 'N/A')}")
        balances = [b for b in data.get('balances', []) if float(b['free']) > 0]
        print(f"  Bakiyeler: {len(balances)} varlik")
        for b in balances[:5]:
            print(f"    - {b['asset']}: {b['free']}")
    else:
        print(f"  Hata: {result['data']}")
    
    # 2. Copy Trading durumu
    print("\n[2] Copy Trading Durumu...")
    result = await binance_request("/sapi/v1/copyTrading/futures/userStatus")
    print(f"  Sonuc: {result['data']}")
    
    # 3. Futures hesap durumu
    print("\n[3] Futures Hesap...")
    result = await binance_request("/fapi/v2/account", futures=True)
    if result["status"] == 200:
        data = result["data"]
        print(f"  Toplam Bakiye: {data.get('totalWalletBalance', 'N/A')} USDT")
        print(f"  Kullanilabilir: {data.get('availableBalance', 'N/A')} USDT")
    else:
        print(f"  Hata: {result['data']}")
    
    # 4. Acik pozisyonlar
    print("\n[4] Futures Pozisyonlar...")
    result = await binance_request("/fapi/v2/positionRisk", futures=True)
    if result["status"] == 200:
        positions = [p for p in result["data"] if float(p.get('positionAmt', 0)) != 0]
        if positions:
            print(f"  {len(positions)} acik pozisyon:")
            for p in positions:
                print(f"    - {p['symbol']}: {p['positionAmt']} @ {p['entryPrice']}")
        else:
            print("  Acik pozisyon yok")
    else:
        print(f"  Hata: {result['data']}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
