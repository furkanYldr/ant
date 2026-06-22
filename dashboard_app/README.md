# 🏘️ Antalya Mahalle Karar Destek Sistemi

Antalya ili 913 mahallesi için veri tabanlı yaşanılabilirlik analizi ve interaktif harita dashboard'u.
T
## Özellikler
- **Tam ekran interaktif harita** — Skor, Küme veya Geo Tip'e göre renklendirme
- **Hamburger menü** — Sol üstte açılıp kapanan panel (Genel/Karşılaştırma/Öneri/Kümeleme sekmeleri)
- **Harita modu switch** — Sağ üstte tıkla, Skor↔Küme↔Geo Tip döngüsü
- **Mahalle detay paneli** — Haritada tıklanan veya aranan mahalle için kapsamlı bilgiler (sağ altta)
- **Persona bazlı öneri** — 10 farklı yaşam profili (Aile, Öğrenci, Emekli, vb.)
- **Özel filtreleme** — POI ağırlıkları ve yaşam tercihleriyle kişisel sıralama
- **6-Tier kümeleme** — Mahalle kalite sınıflandırması
H
## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma
A
```bash
streamlit run app.py
```

## Dosya Yapısı
A
```
dashboard_app/
├── app.py                                    # Ana Streamlit uygulaması
├── recommendation_engine.py                  # Persona bazlı öneri motoru
├── requirements.txt                          # Python bağımlılıkları
├── 00_base_mahalle_final_913_clean.geojson   # Mahalle sınırları (GeoJSON)
├── README.md
└── outputs/
    ├── scoring_results.csv                   # Mahalle skorları + küme etiketleri
    ├── X_raw_clean.csv                       # Ham özellikler (140+ değişken)
    ├── future_scores.csv                     # 5 yıllık tahmin skorları
    └── sub_scores.csv                        # Alt kategori skorları
```

## Teknolojiler
- Python 3.10+
- Streamlit 1.57+
- Folium + streamlit-folium
- Plotly
- GeoPandas
- Shapely
