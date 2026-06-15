"""
RECOMMENDATION ENGINE — Persona Bazli Mahalle Oneri Sistemi
=============================================================
Kullanicinin sectiği yasam tercihine (persona) gore mahalleleri
yeniden agirliklandirir ve Top 10 oneri listesi uretir.

Her persona icin farkli alt-skor agirliklari tanimlanmistir.
Sistem mevcut skoru, alt blok skorlarini ve raw feature'lari
kullanarak kişiye özel mahalle siralamalari uretir.

Personalar:
  - Aile: egitim, saglik, yesil alan, dusuk gurultu, dusuk risk
  - Ogrenci: ucuz, ulasim, sosyal, POI erisim
  - Emekli: sessiz, saglik, yesil, guvenli
  - Yatirim: gelisim potansiyeli, yapilaşma artisi, NTL artisi
  - Sessiz: dusuk gurultu, dusuk nufus yogunlugu
  - Sosyal: yuksek POI, gece isigi, yuruyebilirlik
  - Sahile yakin: dusuk rakim, dusuk lat
  - Dusuk butce: dusuk yogunluk, kucuk kasaba/kirsal
  - Dusuk risk: dusuk sel, dusuk sicaklik, dusuk gurultu
  - Yesil alan: yuksek yesil alan, agac orani
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "outputs")


def quantile_rank(arr, lo=0, hi=100):
    """Rank-based uniform normalization -> [lo, hi]."""
    n = len(arr)
    if n <= 1:
        return np.full(n, (lo + hi) / 2)
    ranks = rankdata(arr, method='average')
    return lo + (ranks - 1) / (n - 1 + 1e-9) * (hi - lo)


def safe_col(df, candidates, default=0):
    """Birden fazla olasi kolon ismi icinden ilk bulanani dondurur."""
    for c in candidates:
        if c in df.columns:
            return df[c].fillna(0).values
    return np.full(len(df), default)


def build_sub_scores(df_raw):
    """
    Raw feature'lardan persona sistemi icin alt-skorlar uret.
    Her skor 0-100 arasi (yuksek = iyi).
    """
    scores = {}

    # 1. Egitim Erisimi (0-100, yuksek=iyi)
    edu = safe_col(df_raw, ['walk_education_within_15min_share',
                             'walk_education_within_10min_share'])
    school_count = safe_col(df_raw, ['type_primary_school_count', 'type_school_count'])
    scores['education_score'] = quantile_rank(edu * 0.6 + np.log1p(school_count) * 0.4)

    # 2. Saglik Erisimi (0-100)
    health = safe_col(df_raw, ['walk_health_within_15min_share',
                                'walk_health_within_10min_share'])
    hospital = safe_col(df_raw, ['type_hospital_count', 'type_doctor_count'])
    pharmacy = safe_col(df_raw, ['type_pharmacy_count'])
    scores['health_score'] = quantile_rank(
        health * 0.5 + np.log1p(hospital) * 0.3 + np.log1p(pharmacy) * 0.2
    )

    # 3. Yesil Alan Skoru (0-100)
    green = safe_col(df_raw, ['green_natural_share', 'vegetated_share'])
    tree = safe_col(df_raw, ['tree_share'])
    park = safe_col(df_raw, ['type_park_count'])
    scores['green_score'] = quantile_rank(green * 0.5 + tree * 0.3 + np.log1p(park) * 0.2)

    # 4. Gurultu Skoru (0-100, yuksek = SESSIZ = iyi)
    noise = safe_col(df_raw, ['noise_density_per_km2'])
    scores['quiet_score'] = 100 - quantile_rank(np.log1p(noise))

    # 5. Sel/Risk Skoru (0-100, yuksek = GUVENLI = iyi)
    flood = safe_col(df_raw, ['raw_flood_exposure_proxy_v2'])
    low_hand = safe_col(df_raw, ['low_hand_5_share', 'low_hand_10_share'])
    scores['safety_score'] = 100 - quantile_rank(flood * 0.6 + low_hand * 0.4)

    # 6. Sicaklik Riski (0-100, yuksek = SERIN = iyi)
    lst = safe_col(df_raw, ['mean_summer_lst_c', 'median_summer_lst_c'])
    scores['heat_comfort'] = 100 - quantile_rank(lst)

    # 7. Ulasim / Transit Erisimi (0-100)
    transit = safe_col(df_raw, ['walk_transit_within_15min_share',
                                 'walk_transit_within_10min_share'])
    bus = safe_col(df_raw, ['type_bus_stop_count', 'type_transit_station_count'])
    scores['transport_score'] = quantile_rank(transit * 0.6 + np.log1p(bus) * 0.4)

    # 8. Yuruyebilirlik / Walkability (0-100)
    walk_cov = safe_col(df_raw, ['walk_category_coverage_within_10min_mean',
                                   'walk_category_coverage_within_15min_mean'])
    street_d = safe_col(df_raw, ['street_density_km_per_km2'])
    scores['walkability_score'] = quantile_rank(walk_cov * 0.6 + np.log1p(street_d) * 0.4)

    # 9. Sosyal Yasam / POI Zenginligi (0-100)
    poi_ent = safe_col(df_raw, ['poi_type_entropy_norm', 'poi_type_entropy'])
    poi_den = safe_col(df_raw, ['poi_density_per_km2'])
    cafe = safe_col(df_raw, ['type_cafe_count', 'type_restaurant_count'])
    scores['social_score'] = quantile_rank(
        np.log1p(poi_den) * 0.4 + poi_ent * 0.3 + np.log1p(cafe) * 0.3
    )

    # 10. Gece Isigi / Aktivite Seviyesi (0-100)
    ntl = safe_col(df_raw, ['ntl_annual_2024_mean', 'ntl_annual_2023_mean'])
    scores['nightlight_score'] = quantile_rank(np.log1p(ntl))

    # 11. Nufus Skoru (0-100, orta nufus = iyi)
    pop = safe_col(df_raw, ['pop', 'population'])
    # Cok dusuk ve cok yuksek kotu, orta ideal → U-shaped penalty
    log_pop = np.log1p(pop)
    median_pop = np.median(log_pop[log_pop > 0])
    pop_dev = np.abs(log_pop - median_pop)
    scores['population_balance'] = 100 - quantile_rank(pop_dev)

    # 12. Market/Gunluk Ihtiyac Erisimi (0-100)
    grocery = safe_col(df_raw, ['walk_grocery_within_15min_share',
                                  'walk_grocery_within_10min_share'])
    market = safe_col(df_raw, ['type_grocery_store_count', 'type_supermarket_count',
                                'type_market_count'])
    scores['daily_needs_score'] = quantile_rank(grocery * 0.6 + np.log1p(market) * 0.4)

    # 13. Sahile Yakinlik (0-100, yuksek = sahile yakin)
    lat = safe_col(df_raw, ['centroid_lat'])
    elev = safe_col(df_raw, ['mean_elevation_m', 'elevation_m_mean'])
    coastal = np.clip((36.95 - lat) / 0.85, 0, 1) * 0.5 + (1 - np.clip(elev / 500, 0, 1)) * 0.5
    scores['coastal_proximity'] = quantile_rank(coastal)

    # 14. Uygun Fiyat / Dusuk Butce (0-100, yuksek = uygun fiyatli)
    # DUZELTME: Eski formul sehir merkezlerini cezalandiriyordu.
    # Yeni mantik: Erisilebilirlik YUKSEK ama luks yapi DUSUK = uygun fiyat
    # Sehir merkezleri yuksek erisilebilirlik sunduklari icin artik 
    # makul skor aliyorlar. Asiri luks/yuksek katli binalar ceza aliyor.
    pop_density = safe_col(df_raw, ['ghsl_pop_density_2020'])
    built_h = safe_col(df_raw, ['built_height_2018_mean'])
    poi_den = safe_col(df_raw, ['poi_density_per_km2'])
    walk_cov_a = safe_col(df_raw, ['walk_category_coverage_within_15min_mean',
                                     'walk_category_coverage_within_10min_mean'])
    # Erisilebilirlik bonusu (yuksek POI + yuruyebilirlik = iyi fiyat/performans)
    access_bonus = quantile_rank(np.log1p(poi_den) * 0.5 + walk_cov_a * 0.5)
    # Luks cezasi (cok yuksek bina = luks, cok dusuk = hic altyapi yok)
    # Orta bina yuksekligi en iyi (8-15m arasi ideal)
    height_penalty = np.abs(np.log1p(built_h) - np.log1p(10)) # 10m'den sapma
    luxury_penalty = quantile_rank(height_penalty)
    # Asiri dusuk yogunluk cezasi (koy = altyapi yok = yasanilabilirlik dusuk)
    too_sparse = quantile_rank(-np.log1p(pop_density + 1))
    # Kombinasyon: erisim bonusu yuksek, luks ceza dusuk, seyreklik ceza dusuk
    affordability = access_bonus * 0.50 + (100 - luxury_penalty) * 0.25 + (100 - too_sparse) * 0.25
    scores['affordability_score'] = quantile_rank(affordability)

    # 15. Yapilaşma Buyumesi (0-100)
    built_chg = safe_col(df_raw, ['built_share_change_2015_2020'])
    vol_chg = safe_col(df_raw, ['volume_density_change_2015_2020'])
    scores['built_growth_score'] = quantile_rank(built_chg * 0.6 + vol_chg * 0.4)

    return scores


# ── PERSONA TANIMLARI ─────────────────────────────────────────────────────────
PERSONAS = {
    "Aile": {
        "description": "Çocuklu aileler için uygun, güvenli ve eğitim/sağlık erişimi yüksek mahalleler",
        "weights": {
            'education_score': 0.20, 'health_score': 0.15, 'green_score': 0.15,
            'quiet_score': 0.12, 'safety_score': 0.12, 'daily_needs_score': 0.10,
            'walkability_score': 0.08, 'heat_comfort': 0.08,
        },
        "explanation": "yüksek eğitim/sağlık erişimi ve düşük gürültü/risk nedeniyle aile için uygundur"
    },
    "Öğrenci": {
        "description": "Uygun fiyatlı, sosyal yaşamı canlı ve ulaşımı kolay mahalleler",
        "weights": {
            'affordability_score': 0.20, 'transport_score': 0.18, 'social_score': 0.18,
            'daily_needs_score': 0.15, 'walkability_score': 0.12, 'nightlight_score': 0.10,
            'education_score': 0.07,
        },
        "explanation": "uygun fiyat, kolay ulaşım ve canlı sosyal yaşam nedeniyle öğrenci için idealdir"
    },
    "Emekli": {
        "description": "Sessiz, sağlık erişimi yüksek, yeşil ve güvenli mahalleler",
        "weights": {
            'quiet_score': 0.20, 'health_score': 0.18, 'green_score': 0.18,
            'safety_score': 0.15, 'heat_comfort': 0.12, 'walkability_score': 0.10,
            'daily_needs_score': 0.07,
        },
        "explanation": "düşük gürültü, yüksek sağlık erişimi ve yeşil alan nedeniyle emekliler için huzurlu bir yaşam sunar"
    },
    "Yatırım": {
        "description": "5 yıllık gelişim potansiyeli yüksek, yapılaşma ve ekonomik aktivite artışı olan mahalleler",
        "weights": {
            'built_growth_score': 0.25, 'nightlight_score': 0.20,
            'transport_score': 0.15, 'social_score': 0.15,
            'walkability_score': 0.10, 'population_balance': 0.10,
            'daily_needs_score': 0.05,
        },
        "explanation": "yapılaşma artışı ve ekonomik aktivite büyümesi nedeniyle yatırım potansiyeli yüksektir"
    },
    "Sessiz Mahalle": {
        "description": "Gürültü seviyesi düşük, sakin ve huzurlu mahalleler",
        "weights": {
            'quiet_score': 0.35, 'green_score': 0.20, 'safety_score': 0.15,
            'heat_comfort': 0.10, 'health_score': 0.10, 'daily_needs_score': 0.10,
        },
        "explanation": "düşük gürültü seviyesi ve yeşil alan zenginliği ile sessiz ve huzurlu bir yaşam sunar"
    },
    "Sosyal Yaşam": {
        "description": "Kafe, restoran, eğlence ve sosyal aktivite açısından zengin mahalleler",
        "weights": {
            'social_score': 0.30, 'nightlight_score': 0.20, 'walkability_score': 0.15,
            'transport_score': 0.12, 'daily_needs_score': 0.10, 'coastal_proximity': 0.08,
            'population_balance': 0.05,
        },
        "explanation": "yüksek POI çeşitliliği ve canlı gece/gündüz aktivitesi ile sosyal yaşam açısından öne çıkar"
    },
    "Sahile Yakın": {
        "description": "Akdeniz kıyısına yakın, düşük rakımlı mahalleler",
        "weights": {
            'coastal_proximity': 0.35, 'green_score': 0.15, 'walkability_score': 0.12,
            'social_score': 0.12, 'heat_comfort': 0.08, 'daily_needs_score': 0.10,
            'safety_score': 0.08,
        },
        "explanation": "sahile yakın konumu ve düşük rakımı ile deniz yaşamına uygun bir lokasyondadır"
    },
    "Düşük Bütçe": {
        "description": "Yaşam maliyeti düşük, temel ihtiyaçlara erişimi olan mahalleler",
        "weights": {
            'affordability_score': 0.35, 'daily_needs_score': 0.15, 'transport_score': 0.15,
            'health_score': 0.10, 'safety_score': 0.10, 'education_score': 0.08,
            'walkability_score': 0.07,
        },
        "explanation": "düşük yaşam maliyeti ve temel ihtiyaçlara erişim imkanı ile bütçe dostu bir seçenektir"
    },
    "Düşük Risk": {
        "description": "Sel, sıcaklık ve gürültü riski düşük, güvenli mahalleler",
        "weights": {
            'safety_score': 0.30, 'heat_comfort': 0.20, 'quiet_score': 0.20,
            'green_score': 0.10, 'health_score': 0.10, 'daily_needs_score': 0.10,
        },
        "explanation": "düşük sel riski, ılıman sıcaklık ve sessiz çevre ile güvenli bir yaşam alanı sunar"
    },
    "Yeşil Alan": {
        "description": "Yeşil alan, ağaç örtüsü ve doğal çevre oranı yüksek mahalleler",
        "weights": {
            'green_score': 0.35, 'heat_comfort': 0.15, 'quiet_score': 0.15,
            'safety_score': 0.12, 'walkability_score': 0.10, 'health_score': 0.08,
            'daily_needs_score': 0.05,
        },
        "explanation": "yüksek yeşil alan oranı ve doğal çevre zenginliği ile doğa severler için idealdir"
    },
}


def recommend(df_raw, persona_name, top_n=10, geo_filter=None, ilce_filter=None):
    """
    Belirli bir persona icin mahalleleri siralar ve Top N onerisi dondurur.

    Args:
        df_raw: X_raw_clean.csv DataFrame
        persona_name: PERSONAS dict'teki anahtar
        top_n: Kac mahalle onerilecek
        geo_filter: Opsiyonel geo_type filtresi (orn: "Sehir Merkezi")
        ilce_filter: Opsiyonel ilce filtresi

    Returns:
        DataFrame: Top N mahalle onerisi detaylarla
    """
    if persona_name not in PERSONAS:
        raise ValueError(f"Bilinmeyen persona: {persona_name}. Secenekler: {list(PERSONAS.keys())}")

    persona = PERSONAS[persona_name]
    sub_scores = build_sub_scores(df_raw)

    # Agirlikli skor hesapla
    total_weight = sum(persona['weights'].values())
    persona_score = np.zeros(len(df_raw))
    for score_name, weight in persona['weights'].items():
        if score_name in sub_scores:
            persona_score += sub_scores[score_name] * (weight / total_weight)

    # Filtrele
    mask = np.ones(len(df_raw), dtype=bool)
    if ilce_filter:
        mask &= df_raw['ilce_name'].str.lower().str.contains(ilce_filter.lower(), na=False)
    # Geo filter icin scoring_results gerekli
    geo_df = None
    if geo_filter:
        scores_path = os.path.join(OUT_DIR, "scoring_results.csv")
        if os.path.exists(scores_path):
            geo_df = pd.read_csv(scores_path)[['mah_id', 'geo_type']]
            merged_geo = df_raw[['mah_id']].merge(geo_df, on='mah_id', how='left')
            mask &= merged_geo['geo_type'].str.lower().str.contains(geo_filter.lower(), na=False)

    # Top N
    persona_score_masked = np.where(mask, persona_score, -1)
    top_indices = np.argsort(-persona_score_masked)[:top_n]

    results = []
    for idx in top_indices:
        if not mask[idx]:
            continue
        row = {
            'sira': len(results) + 1,
            'mah_name': df_raw.iloc[idx].get('mah_name', '?'),
            'ilce_name': df_raw.iloc[idx].get('ilce_name', '?'),
            'mah_id': df_raw.iloc[idx]['mah_id'],
            'persona_score': round(persona_score[idx], 1),
        }
        # Persona'daki her alt skoru ekle
        for score_name in persona['weights']:
            if score_name in sub_scores:
                row[score_name] = round(sub_scores[score_name][idx], 1)
        # Aciklama
        row['aciklama'] = f"Bu mahalle {persona['explanation']}."
        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("RECOMMENDATION ENGINE — Persona Bazli Mahalle Oneri")
    print("=" * 60)

    df_raw = pd.read_csv(os.path.join(OUT_DIR, "X_raw_clean.csv"))
    print(f"\nLoaded: {len(df_raw)} mahalle")

    # Alt skorlari uret ve kaydet
    print("\n--- Alt Skorlar Uretiliyor ---")
    sub_scores = build_sub_scores(df_raw)
    df_sub = pd.DataFrame({'mah_id': df_raw['mah_id'].values})
    for name, vals in sub_scores.items():
        df_sub[name] = np.round(vals, 2)
        print(f"  {name:25s}: mean={vals.mean():.1f}  std={vals.std():.1f}")

    df_sub.to_csv(os.path.join(OUT_DIR, "sub_scores.csv"), index=False)
    print(f"\n  [OK] sub_scores.csv kaydedildi ({len(df_sub)} satir, {len(sub_scores)} skor)")

    # Her persona icin ornek Top 5
    print("\n--- Persona Oneri Ornekleri ---")
    for persona_name in PERSONAS:
        print(f"\n  [{persona_name}] — {PERSONAS[persona_name]['description']}")
        result = recommend(df_raw, persona_name, top_n=5)
        for _, r in result.iterrows():
            print(f"    {r['sira']}. {r['mah_name']:25s} ({r['ilce_name']:12s}) "
                  f"| skor={r['persona_score']:.0f}")

    print("\n[RECOMMENDATION ENGINE COMPLETE]")


if __name__ == "__main__":
    main()
