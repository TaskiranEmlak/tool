# 🚀 HFT Trading Tools

Kripto piyasaları için yüksek frekanslı trading araç seti.

## Özellikler

- **CVD (Cumulative Volume Delta)**: Alış/satış baskısını ölçer
- **OBI (Order Book Imbalance)**: Emir defteri dengesizliğini hesaplar
- **Likidasyon Heatmap**: Likidasyon bölgelerini gösterir
- **Otomatik Sinyal Üretimi**: CVD + OBI birleşik analiz
- **God Mode Dashboard**: Canlı görselleştirme

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Konsol Modu
```bash
python main.py
```

### Dashboard (God Mode)
```bash
streamlit run dashboard/app.py
```

## Proje Yapısı

```
kriptol/
├── config/          # Konfigürasyon
├── core/            # Temel altyapı
│   ├── data_collector.py  # WebSocket veri toplama
│   ├── database.py        # Zaman serisi DB
│   └── event_bus.py       # Olay sistemi
├── indicators/      # Göstergeler
│   ├── cvd.py            # Cumulative Volume Delta
│   ├── obi.py            # Order Book Imbalance
│   └── liquidation.py    # Likidasyon Heatmap
├── signals/         # Sinyal sistemi
│   └── signal_manager.py
├── dashboard/       # Streamlit UI
│   └── app.py
└── main.py          # Ana giriş
```

## Göstergeler

### CVD (Cumulative Volume Delta)
Piyasa emri hacmini takip eder:
- Pozitif: Alış ağırlıklı
- Negatif: Satış ağırlıklı

### OBI (Order Book Imbalance)
Emir defteri dengesizliği:
- +0.3 üzeri: Güçlü alış desteği → LONG sinyal
- -0.3 altı: Güçlü satış baskısı → SHORT sinyal

### Likidasyon Heatmap
Tahmini likidasyon seviyeleri ve "mıknatıs bölgeleri".

## Sinyal Koşulları

### LONG
- OBI > +0.3
- CVD yükseliş trendi
- CVD-OBI uyumu

### SHORT
- OBI < -0.3
- CVD düşüş trendi
- CVD-OBI uyumu

## Lisans

MIT
