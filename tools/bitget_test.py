# Bitget Copy Trading API Test V2
"""
Bitget V2 API endpointlerini test eder.
"""

import asyncio
import aiohttp
from typing import Dict, List

BITGET_BASE_URL = "https://api.bitget.com"

async def test_endpoint(url: str, params: dict = None) -> dict:
    """Endpoint'i test et"""
    headers = {
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers) as resp:
            print(f"\n[Test] URL: {url}")
            print(f"[Test] Params: {params}")
            print(f"[Test] Status: {resp.status}")
            
            if resp.status == 200:
                data = await resp.json()
                return data
            else:
                text = await resp.text()
                print(f"[Test] Error: {text[:300]}")
                return {"error": text}

async def main():
    print("=" * 60)
    print("BITGET API ENDPOINT TESTI")
    print("=" * 60)
    
    # Farkli endpoint versiyonlarini dene
    endpoints = [
        # V2 Copy Trading
        f"{BITGET_BASE_URL}/api/v2/copy/mix-trader/query-trader-list",
        # V2 alternatif
        f"{BITGET_BASE_URL}/api/v2/copytrading/futures/trader/list",
        # V2 spot copy
        f"{BITGET_BASE_URL}/api/v2/copy/spot-trader/query-trader-list",
        # Public market data
        f"{BITGET_BASE_URL}/api/v2/mix/market/tickers",
    ]
    
    params = {
        "pageNo": "1",
        "pageSize": "10"
    }
    
    for endpoint in endpoints:
        result = await test_endpoint(endpoint, params)
        if result.get("code") == "00000":
            print(f"\n*** BASARILI: {endpoint} ***")
            print(f"Data: {str(result.get('data', ''))[:500]}")
            break
        else:
            print(f"Hata: {result.get('msg', result.get('error', 'Unknown'))[:200]}")
    
    # Market data testi
    print("\n\n[Market Data Testi]")
    market_url = f"{BITGET_BASE_URL}/api/v2/mix/market/tickers"
    market_params = {"productType": "USDT-FUTURES"}
    
    result = await test_endpoint(market_url, market_params)
    if result.get("code") == "00000":
        data = result.get("data", [])
        print(f"\nMarket data BASARILI! {len(data)} coin bulundu")
        for d in data[:3]:
            print(f"  - {d.get('symbol')}: {d.get('lastPr')}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
