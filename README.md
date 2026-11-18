# Yapay Zekâ Destekli Gerçek Zamanlı Araç Rota Optimizasyonu Platformu

Bitirme projesi — Nazar Demir, Yazılım Mühendisliği

Teslimat filosundaki araçların rotalarını yapay zekâ ile planlayan ve trafik, kaza ya da yeni sipariş olduğunda rotaları gerçek zamanlı olarak yeniden hesaplayan platform.

## Nasıl çalışır?

1. **Süre tahmini:** NYC Taxi verisiyle eğitilen XGBoost modeli, iki nokta arasındaki yolculuk süresini tahmin eder.
2. **Rota optimizasyonu:** Google OR-Tools, bu sürelerle siparişleri araçlara dağıtır ve durak sırasını belirler.
3. **Gerçek zamanlı platform:** Araç konumları Kafka üzerinden işlenir ve canlı haritada gösterilir.

## İlk sonuçlar (1. hafta)

| Model | RMSLE | Ortalama hata |
|---|---|---|
| Taban çizgi (mesafe / hız) | 0,542 | 6,9 dk |
| Linear Regression | 0,518 | 5,8 dk |
| **XGBoost** | **0,296** | **2,7 dk** |

Rota demosu (3 araç, 10 teslimat): OR-Tools, en yakın komşu (greedy) yöntemine göre toplam süreyi %25 kısalttı.

## Kurulum

```
pip install -r requirements.txt
```

Veri seti: [Kaggle – New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (`train.zip`). Veri dosyası boyutu nedeniyle repoya eklenmemiştir.

## Proje yapısı

```
01_eda_ve_eta_modeli.ipynb   Veri analizi, model karşılaştırması, OR-Tools demosu
requirements.txt             Python paketleri
```

## Teknolojiler

Python, Pandas, Scikit-learn, XGBoost, Google OR-Tools, FastAPI, Apache Kafka, PostgreSQL, MongoDB, Redis, React, Leaflet.js
