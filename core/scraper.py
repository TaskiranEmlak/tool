# Copy Trading System - Binance Leaderboard Scraper (v4)
"""
Selenium kullanarak Binance Leaderboard verilerini ceker.
API artik cookie gerektirdigi icin Selenium ile sayfadan cekiyoruz.
"""

import asyncio
import time
import re
import json
from datetime import datetime
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("[UYARI] Selenium yuklu degil!")

from core.copy_db import get_db


class BinanceScraper:
    """Binance Copy Trading Selenium Scraper"""
    
    # Yeni URL - Copy Trading sayfasi
    COPY_TRADING_URL = "https://www.binance.com/en/copy-trading"
    
    def __init__(self, headless: bool = False):  # Headless=False daha guvenilir
        self.headless = headless
        self.driver = None
        self.db = get_db()
        self.scrape_interval = 120  # 2 dakika (cok sik isteme)
        self.is_running = False
    
    def _init_driver(self):
        """Chrome driver baslat"""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium yuklu degil!")
        
        options = Options()
        
        if self.headless:
            options.add_argument('--headless=new')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--lang=en-US')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-extensions')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Anti-detection
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        
        # Anti-detection script
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print("[Scraper] Chrome baslatildi")
    
    def _close_driver(self):
        """Driver'i kapat"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
    
    def scrape_leaderboard(self, limit: int = 10) -> List[Dict]:
        """
        Copy Trading sayfasindan traderlari cek
        """
        if not self.driver:
            self._init_driver()
        
        traders = []
        
        try:
            print(f"[Scraper] Sayfa yukleniyor: {self.COPY_TRADING_URL}")
            self.driver.get(self.COPY_TRADING_URL)
            
            # Sayfanin yuklenmesini bekle
            print("[Scraper] Sayfa yuklenmesi bekleniyor (15sn)...")
            time.sleep(15)
            
            # Cookie popup'i kapat
            try:
                cookie_btns = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Accept')]")
                for btn in cookie_btns:
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(2)
                        break
            except:
                pass
            
            # Sayfayi asagi kaydir
            print("[Scraper] Sayfa kaydiriliyor...")
            for i in range(5):
                self.driver.execute_script(f"window.scrollTo(0, {400 * (i+1)})")
                time.sleep(1.5)
            
            # Tekrar bekle
            time.sleep(5)
            
            # HTML'i al
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Debug: Sayfa basligini kontrol et
            title = soup.find('title')
            print(f"[Scraper] Sayfa: {title.text if title else 'Bilinmiyor'}")
            
            # Trader linklerini bul - FARKLI FORMATLAR
            all_links = soup.find_all('a', href=True)
            
            # encryptedUid veya portfolioId iceren linkler
            uid_links = [a for a in all_links if 'encryptedUid' in a.get('href', '') or 'portfolioId' in a.get('href', '')]
            
            print(f"[Scraper] {len(uid_links)} trader linki bulundu")
            
            if not uid_links:
                # Debug: Sayfa icerigini kontrol et
                body_text = soup.get_text()[:1000]
                print(f"[Scraper] Sayfa icerigi: {body_text[:300]}...")
                
                # Tum href'leri kontrol et
                all_hrefs = [a.get('href', '') for a in all_links]
                copy_hrefs = [h for h in all_hrefs if 'copy' in h.lower() or 'portfolio' in h.lower() or 'trader' in h.lower()]
                print(f"[Scraper] Copy/Portfolio linkleri: {copy_hrefs[:5]}")
            
            seen_uids = set()
            for link in uid_links[:limit*2]:
                try:
                    href = link.get('href', '')
                    
                    # encryptedUid veya portfolioId cek
                    uid_match = re.search(r'encryptedUid=([A-Za-z0-9]+)', href)
                    if not uid_match:
                        uid_match = re.search(r'portfolioId=([A-Za-z0-9-]+)', href)
                    if not uid_match:
                        uid_match = re.search(r'/portfolio/([A-Za-z0-9-]+)', href)
                    
                    if not uid_match:
                        continue
                    
                    uid = uid_match.group(1)
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)
                    
                    # Isim
                    nickname = link.get_text(strip=True)[:20] or f"Trader_{uid[:6]}"
                    
                    # Parent'tan ROI bulmaya calis
                    roi = 0
                    parent = link.find_parent('tr') or link.find_parent('div')
                    if parent:
                        text = parent.get_text()
                        roi_match = re.search(r'([-+]?\d+\.?\d*)%', text)
                        if roi_match:
                            roi = float(roi_match.group(1))
                    
                    traders.append({
                        'uid': uid,
                        'nickname': nickname,
                        'roi': roi,
                        'pnl': 0,
                        'follower_count': 0
                    })
                    
                    if len(traders) >= limit:
                        break
                except:
                    continue
            
            print(f"[Scraper] {len(traders)} trader bulundu!")
            for t in traders[:3]:
                print(f"  - {t['nickname']}: ROI={t['roi']}%, UID={t['uid'][:8]}...")
            
        except Exception as e:
            print(f"[Scraper] Hata: {e}")
            import traceback
            traceback.print_exc()
        
        return traders
    
    def scrape_trader_positions(self, uid: str, nickname: str = "") -> List[Dict]:
        """
        Trader'in pozisyonlarini cek
        """
        if not self.driver:
            self._init_driver()
        
        positions = []
        
        try:
            url = f"https://www.binance.com/en/futures-activity/leaderboard/user/um?encryptedUid={uid}"
            self.driver.get(url)
            time.sleep(5)
            
            # Position tab'ina tikla
            try:
                tabs = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Position')]")
                for tab in tabs:
                    if tab.is_displayed():
                        tab.click()
                        time.sleep(3)
                        break
            except:
                pass
            
            # Sayfa icerigini al
            html = self.driver.page_source
            
            # USDT ciftlerini bul
            usdt_pattern = r'([A-Z]{2,10}USDT)'
            matches = re.findall(usdt_pattern, html)
            symbols = list(dict.fromkeys(matches))[:5]
            
            for symbol in symbols:
                # Long/Short belirle
                idx = html.find(symbol)
                if idx == -1:
                    continue
                
                context = html[max(0, idx-200):idx+200]
                
                direction = None
                if 'Long' in context or 'long' in context:
                    direction = 'LONG'
                elif 'Short' in context or 'short' in context:
                    direction = 'SHORT'
                
                if not direction:
                    continue
                
                # PnL bul
                pnl = 0
                pnl_match = re.search(r'([-+]?\d+\.?\d*)%', context)
                if pnl_match:
                    pnl = float(pnl_match.group(1))
                
                positions.append({
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': 0,
                    'mark_price': 0,
                    'pnl_percent': pnl,
                    'leverage': 10
                })
            
            if positions:
                print(f"[Scraper] {nickname}: {len(positions)} pozisyon")
                for p in positions:
                    print(f"  - {p['direction']} {p['symbol']}: {p['pnl_percent']:.1f}%")
            
        except Exception as e:
            print(f"[Scraper] Pozisyon hatasi: {e}")
        
        return positions
    
    async def discover_top_traders(self, count: int = 10) -> List[str]:
        """Top traderlari kesfet"""
        print(f"[Scraper] Top {count} trader kesfediliyor...")
        
        traders = self.scrape_leaderboard(limit=count)
        
        if not traders:
            print("[Scraper] UYARI: Trader bulunamadi!")
            return []
        
        # Veritabanina kaydet
        for trader in traders:
            self.db.upsert_trader(trader['uid'], trader)
        
        print(f"[Scraper] {len(traders)} trader kaydedildi")
        return [t['uid'] for t in traders]
    
    async def update_all_positions(self):
        """Tum traderlarin pozisyonlarini guncelle"""
        traders = self.db.get_active_traders()
        
        for trader in traders:
            try:
                positions = self.scrape_trader_positions(trader['uid'], trader.get('nickname', ''))
                
                # Pozisyonlari kaydet
                for pos in positions:
                    existing = self.db.get_trader_positions(trader['uid'])
                    keys = {f"{p['symbol']}_{p['direction']}" for p in existing}
                    
                    key = f"{pos['symbol']}_{pos['direction']}"
                    if key not in keys:
                        self.db.add_position(trader['uid'], pos)
                
                await asyncio.sleep(3)  # Rate limiting
            except Exception as e:
                print(f"[Scraper] Guncelleme hatasi: {e}")
        
        # Consensus kontrol et
        from core.consensus import get_consensus_engine
        engine = get_consensus_engine()
        signals = engine.check_for_new_signals()
        
        for sig in signals:
            print(f"[SINYAL] {sig['direction']} {sig['symbol']} (Skor: {sig['consensus_score']:.1f})")
    
    async def run_loop(self):
        """Surekli tarama dongusu"""
        self.is_running = True
        print("[Scraper] Tarama dongusu basladi")
        
        try:
            self._init_driver()
            
            # Ilk kesfif
            await self.discover_top_traders(count=10)
            
            while self.is_running:
                await self.update_all_positions()
                
                print(f"[Scraper] {self.scrape_interval}sn bekleniyor...")
                await asyncio.sleep(self.scrape_interval)
                
        except Exception as e:
            print(f"[Scraper] Dongu hatasi: {e}")
        finally:
            self._close_driver()
            self.is_running = False
    
    def stop(self):
        """Taramayi durdur"""
        self.is_running = False
        self._close_driver()


# Test
if __name__ == "__main__":
    async def test():
        scraper = BinanceScraper(headless=False)
        traders = await scraper.discover_top_traders(count=5)
        
        if traders:
            await scraper.update_all_positions()
        
        scraper.stop()
    
    asyncio.run(test())
