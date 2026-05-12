"""
STREAMLIT APP v4 â€” Antalya Mahalle Karar Destek Sistemi
Tam ekran harita + hamburger menu + sag alt detay paneli
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
import streamlit.components.v1 as stc
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from recommendation_engine import PERSONAS, build_sub_scores, recommend

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
GEO = os.path.join(BASE, "00_base_mahalle_final_913_clean.geojson")

st.set_page_config(page_title="Antalya Mahalle", page_icon="ğŸ˜ï¸", layout="wide",
                    initial_sidebar_state="collapsed")

# â”€â”€ CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/* Force dark theme everywhere (Streamlit Cloud fix) */
.stApp, [data-testid="stAppViewContainer"] {background:#0e0e14!important;color:#d0d0d0!important;}
.stMainBlockContainer {padding:0!important;max-width:100%!important;background:#0e0e14!important;}
header[data-testid="stHeader"] {background:transparent!important;}
div[data-testid="stToolbar"] {display:none!important;}
section[data-testid="stSidebar"] {
    background:rgba(12,12,18,0.97)!important;
    backdrop-filter:blur(16px);
    border-right:1px solid rgba(255,255,255,0.08);
    width:400px!important;
    color:#d0d0d0!important;
}
section[data-testid="stSidebar"] * {color:inherit!important;}
section[data-testid="stSidebar"] > div {padding-top:1.2rem;}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {color:#fff!important;}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {color:#aaa!important;}
/* Hide ALL native sidebar toggle buttons */
button[data-testid="stBaseButton-headerNoPadding"],
button[data-testid="stSidebarCollapseButton"],
div[data-testid="collapsedControl"] {display:none!important;}
.stElementContainer {margin:0!important;padding:0!important;}
div[data-testid="stVerticalBlock"] > div {gap:0!important;}
iframe {width:100%!important;border:none!important;}
::-webkit-scrollbar {width:5px;}
::-webkit-scrollbar-thumb {background:rgba(255,255,255,0.15);border-radius:3px;}
@media (max-width:768px) {
    section[data-testid="stSidebar"] {width:100vw!important;}
}
</style>
""", unsafe_allow_html=True)

# â”€â”€ HAMBURGER + MAP SWITCH (js via components.html) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if 'color_by' not in st.session_state:
    st.session_state.color_by = 'Skor'
color_by = st.session_state.color_by

MODE_ICONS = {'Skor':'ğŸ¯','Kume':'ğŸ·ï¸','Geo Tip':'ğŸŒ'}
MODE_CYCLE = {'Skor':'Kume','Kume':'Geo Tip','Geo Tip':'Skor'}
cur_icon = MODE_ICONS.get(color_by,'ğŸ¯')
next_mode = MODE_CYCLE.get(color_by,'Skor')

stc.html(f"""
<script>
(function() {{
    var pd = window.parent.document;
    // --- Hamburger (left) ---
    var old = pd.getElementById('custom-hamburger');
    if (old) old.remove();
    var btn = pd.createElement('div');
    btn.id = 'custom-hamburger';
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" style="width:22px;height:22px;"><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/></svg>';
    btn.style.cssText = 'position:fixed;top:14px;left:14px;z-index:999999;background:rgba(20,20,28,0.9);border:1px solid rgba(255,255,255,0.18);border-radius:50%;width:48px;height:48px;cursor:pointer;backdrop-filter:blur(12px);box-shadow:0 4px 24px rgba(0,0,0,0.55);display:flex;align-items:center;justify-content:center;transition:all 0.25s ease;';
    btn.onmouseenter = function(){{btn.style.background='rgba(40,40,55,0.95)';btn.style.borderColor='rgba(100,181,246,0.5)';btn.style.transform='scale(1.08)';}};
    btn.onmouseleave = function(){{btn.style.background='rgba(20,20,28,0.9)';btn.style.borderColor='rgba(255,255,255,0.18)';btn.style.transform='scale(1)';}};
    btn.onclick = function(){{
        var sels = ['button[data-testid="stBaseButton-headerNoPadding"]','button[data-testid="stSidebarCollapseButton"]','[data-testid="collapsedControl"] button','[data-testid="stSidebarCollapsedControl"] button'];
        for(var i=0;i<sels.length;i++){{var t=pd.querySelector(sels[i]);if(t){{t.style.display='block';t.click();t.style.display='none';return;}}}}
        var hb=pd.querySelectorAll('header button');for(var j=0;j<hb.length;j++){{hb[j].click();return;}}
    }};
    pd.body.appendChild(btn);
    // --- Map mode (right) ---
    var old2 = pd.getElementById('map-mode-sw');
    if (old2) old2.remove();
    var sw = pd.createElement('div');
    sw.id = 'map-mode-sw';
    sw.innerHTML = '<span style="font-size:16px">{cur_icon}</span><span style="font-size:9px;color:#bbb;margin-top:1px">{color_by}</span>';
    sw.title = 'Tikla: {next_mode}';
    sw.style.cssText = 'position:fixed;top:14px;right:14px;z-index:999999;background:rgba(20,20,28,0.9);border:1px solid rgba(255,255,255,0.18);border-radius:24px;min-width:52px;height:48px;cursor:pointer;backdrop-filter:blur(12px);box-shadow:0 4px 24px rgba(0,0,0,0.55);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 12px;transition:all 0.25s ease;';
    sw.onmouseenter = function(){{sw.style.background='rgba(40,40,55,0.95)';sw.style.borderColor='rgba(100,181,246,0.5)';sw.style.transform='scale(1.08)';}};
    sw.onmouseleave = function(){{sw.style.background='rgba(20,20,28,0.9)';sw.style.borderColor='rgba(255,255,255,0.18)';sw.style.transform='scale(1)';}};
    sw.onclick = function(){{var all=pd.querySelectorAll('button');for(var k=0;k<all.length;k++){{if(all[k].textContent.indexOf('CYCLE_MAP')>=0){{all[k].click();return;}}}}}};
    pd.body.appendChild(sw);
}})();
</script>
""", height=0)




# â”€â”€ DATA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data
def load_all():
    s = pd.read_csv(os.path.join(OUT, "scoring_results.csv"))
    r = pd.read_csv(os.path.join(OUT, "X_raw_clean.csv"))
    f = pd.read_csv(os.path.join(OUT, "future_scores.csv")) if os.path.exists(os.path.join(OUT, "future_scores.csv")) else None
    sb = pd.read_csv(os.path.join(OUT, "sub_scores.csv")) if os.path.exists(os.path.join(OUT, "sub_scores.csv")) else None
    g = gpd.read_file(GEO)
    return s, r, f, sb, g

scores_df, raw_df, future_df, sub_df, gdf = load_all()

master = scores_df.copy()
if future_df is not None:
    fc = [c for c in future_df.columns if c not in master.columns or c == 'mah_id']
    master = master.merge(future_df[fc], on='mah_id', how='left')
if sub_df is not None:
    sc = [c for c in sub_df.columns if c not in master.columns or c == 'mah_id']
    master = master.merge(sub_df[sc], on='mah_id', how='left')

poi_cols = sorted([c for c in raw_df.columns if c.startswith('type_')])
poi_labels = {c: c.replace('type_','').replace('_count','').replace('_',' ').title() for c in poi_cols}
block_cols = [c for c in ['demographics','built_activity','accessibility','morphology','environment','seasonality'] if c in master.columns]

name_options = sorted([f"{r['mah_name']} ({r['ilce_name']})" for _, r in master[['mah_name','ilce_name']].drop_duplicates().iterrows()])
name_to_id = {f"{r['mah_name']} ({r['ilce_name']})": r['mah_id'] for _, r in master.iterrows()}

CLUSTER_COLORS = {'A - En Iyi':'#00c853','B - Iyi':'#2979ff','C - Ortanin Ustu':'#ffd600',
                  'D - Orta':'#ff6d00','E - Dusuk':'#aa00ff','F - En Dusuk':'#d50000'}
GEO_COLORS = {'Sehir Merkezi':'#e53935','Sahil / Turistik':'#1e88e5','Sehir Siniri':'#43a047',
              'Turistik Sahil Kasabasi':'#00acc1','Kucuk Kasaba':'#fb8c00',
              'Kirsal / Ovacik':'#8d6e63','DaglÄ±k / Yukari Yerlesim':'#757575'}

def score_color(s):
    if s >= 70: return '#1a9641'
    if s >= 50: return '#a6d96a'
    if s >= 30: return '#fee08b'
    if s >= 15: return '#f46d43'
    return '#d73027'




# â”€â”€ SIDEBAR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("### ğŸ˜ï¸ Antalya Mahalle")
    page = st.radio("Sekme:", ["ğŸ—ºï¸ Genel","âš–ï¸ Karsilastirma","ğŸ’¡ Oneri","ğŸ“Š Kumeleme"],
                     label_visibility="collapsed")
    st.divider()
    srch_col, clr_col = st.columns([5,1])
    with srch_col:
        search = st.selectbox("ğŸ” Mahalle Ara:", [""] + name_options, key="detail_search")
    with clr_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("âœ•", key="clear_search", help="AramayÄ± temizle"):
            st.session_state.detail_search = ""
            st.session_state.pop('show_detail', None)
            st.rerun()
    st.divider()

    # â”€â”€ TAB CONTENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if page == "ğŸ—ºï¸ Genel":
        st.subheader("ğŸ“Š Genel BakÄ±ÅŸ")
        st.caption(f"Toplam: {len(master)} mahalle")
        mc1, mc2 = st.columns(2)
        with mc1: st.metric("Ort. Skor", f"{master['score_final'].mean():.1f}")
        with mc2: st.metric("Max Skor", f"{master['score_final'].max():.1f}")
        if 'pop' in master.columns:
            st.metric("Toplam NÃ¼fus", f"{master['pop'].sum():,.0f}")
        st.markdown("**KÃ¼me DaÄŸÄ±lÄ±mÄ±**")
        c_order = ['A - En Iyi','B - Iyi','C - Ortanin Ustu','D - Orta','E - Dusuk','F - En Dusuk']
        rows = []
        for cl in c_order:
            sub = master[master['cluster_label']==cl]
            if len(sub)==0: continue
            rows.append({'KÃ¼me':cl,'N':len(sub),'Skor':f"{sub['score_final'].mean():.0f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        fig_hist = px.histogram(master, x='score_final', nbins=30, color='cluster_label',
                                 color_discrete_map=CLUSTER_COLORS, category_orders={'cluster_label':c_order})
        fig_hist.update_layout(height=250, xaxis_title="Skor", yaxis_title="SayÄ±",
                                margin=dict(l=0,r=0,t=10,b=30))
        st.plotly_chart(fig_hist, use_container_width=True)

    elif page == "âš–ï¸ Karsilastirma":
        st.subheader("âš–ï¸ KarÅŸÄ±laÅŸtÄ±rma")
        if 'compare_list' not in st.session_state: st.session_state.compare_list = []
        pick = st.selectbox("Mahalle seÃ§:", [""] + name_options, key="cmp_pick")
        if st.button("â• Ekle", type="primary"):
            if pick and pick not in st.session_state.compare_list and len(st.session_state.compare_list) < 5:
                st.session_state.compare_list.append(pick)
        if st.session_state.compare_list:
            for i, name in enumerate(st.session_state.compare_list):
                cc1, cc2 = st.columns([5,1])
                with cc1: st.write(f"**{i+1}.** {name}")
                with cc2:
                    if st.button("âŒ", key=f"rm_{i}"):
                        st.session_state.compare_list.pop(i); st.rerun()
        if len(st.session_state.compare_list) >= 2:
            sel_ids = [name_to_id.get(n) for n in st.session_state.compare_list if n in name_to_id]
            comp = master[master['mah_id'].isin(sel_ids)].copy()
            comp['display'] = comp.apply(lambda r: f"{r['mah_name']} ({r['ilce_name']})", axis=1)
            cols = st.columns(len(comp))
            for i, (_, r) in enumerate(comp.iterrows()):
                with cols[i]:
                    st.metric(r['mah_name'][:10], f"{r['score_final']:.0f}")
            radar_cols = [c for c in block_cols if c in comp.columns]
            if radar_cols:
                fig = go.Figure()
                for _, row in comp.iterrows():
                    vals = [row.get(c,0) for c in radar_cols]
                    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=radar_cols+[radar_cols[0]],
                                                   fill='toself', name=row['display'][:15], opacity=0.6))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), height=300,
                                  margin=dict(l=20,r=20,t=20,b=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("En az 2 mahalle ekleyin.")

    elif page == "ğŸ’¡ Oneri":
        st.subheader("ğŸ’¡ Mahalle Ã–ner")
        tab1, tab2 = st.tabs(["HazÄ±r Profil", "Ã–zel Filtre"])
        with tab1:
            sel_p = st.selectbox("Profil:", list(PERSONAS.keys()))
            st.info(PERSONAS[sel_p]['description'])
            ilce_f = st.selectbox("Ä°lÃ§e:", ['TÃ¼mÃ¼'] + sorted(raw_df['ilce_name'].dropna().unique()))
            top_n = st.slider("SayÄ±:", 5, 20, 10)
            if st.button("Ã–nerileri GÃ¶ster", type="primary", key="btn_p"):
                res = recommend(raw_df, sel_p, top_n=top_n,
                                ilce_filter=None if ilce_f=='TÃ¼mÃ¼' else ilce_f)
                for _, r in res.iterrows():
                    st.write(f"**#{int(r['sira'])}** {r['mah_name']} ({r['ilce_name']}) â€” **{r['persona_score']:.0f}**")
        with tab2:
            st.markdown("**Kendi filtrelerinizi ayarlayÄ±n**")
            sub_scores = build_sub_scores(raw_df)
            w_opts = {'education_score':'EÄŸitim','health_score':'SaÄŸlÄ±k','green_score':'YeÅŸil',
                       'quiet_score':'Sessizlik','safety_score':'GÃ¼venlik','transport_score':'UlaÅŸÄ±m',
                       'walkability_score':'YÃ¼rÃ¼nebilirlik','social_score':'Sosyal','affordability_score':'Uygun Fiyat',
                       'coastal_proximity':'Sahil'}
            weights = {}
            wc = st.columns(2)
            for i, (wk, wl) in enumerate(w_opts.items()):
                with wc[i%2]: weights[wk] = st.slider(wl, 0, 10, 5, key=f"w_{wk}")
            fc1, fc2 = st.columns(2)
            with fc1: ilce_f2 = st.multiselect("Ä°lÃ§e:", sorted(raw_df['ilce_name'].dropna().unique()), key="of_ilce")
            with fc2: min_score2 = st.slider("Min Skor:", 0, 80, 0)
            if st.button("Filtrele", type="primary", key="btn_o"):
                mask = np.ones(len(raw_df), dtype=bool)
                if ilce_f2: mask &= raw_df['ilce_name'].isin(ilce_f2)
                if min_score2 > 0:
                    sc_m = pd.read_csv(os.path.join(OUT,"scoring_results.csv"))[['mah_id','score_final']]
                    merged = raw_df[['mah_id']].merge(sc_m, on='mah_id', how='left')
                    mask &= merged['score_final'].fillna(0) >= min_score2
                total_w = sum(weights.values()) + 1e-9
                custom = np.zeros(len(raw_df))
                for wk, wv in weights.items():
                    if wk in sub_scores and wv > 0: custom += sub_scores[wk] * (wv / total_w)
                score_m = np.where(mask, custom, -1)
                top_idx = np.argsort(-score_m)[:15]
                st.write(f"**{mask.sum()} mahalle uygun.** En iyi 15:")
                for rank, idx in enumerate(top_idx, 1):
                    if not mask[idx]: break
                    r = raw_df.iloc[idx]
                    st.write(f"**#{rank}** {r['mah_name']} ({r['ilce_name']}) â€” {custom[idx]:.0f}")

    elif page == "ğŸ“Š Kumeleme":
        st.subheader("ğŸ“Š KÃ¼meleme (6-Tier)")
        CDESC = {'A - En Iyi':"Merkezi, hizmet yoÄŸun.","B - Iyi":"Åehir sÄ±nÄ±rÄ±, orta-yÃ¼ksek.",
                  'C - Ortanin Ustu':"Sahil/turistik.",'D - Orta':"KÃ¼Ã§Ã¼k kasaba.",
                  'E - Dusuk':"KÄ±rsal, sÄ±nÄ±rlÄ±.",'F - En Dusuk':"DaÄŸlÄ±k, eriÅŸim zor."}
        c_order = ['A - En Iyi','B - Iyi','C - Ortanin Ustu','D - Orta','E - Dusuk','F - En Dusuk']
        for cl in c_order:
            sub = master[master['cluster_label']==cl]
            if len(sub)==0: continue
            clr = CLUSTER_COLORS.get(cl,'#888')
            with st.expander(f"{cl} ({len(sub)} mah.)", expanded=(cl=='A - En Iyi')):
                st.caption(CDESC.get(cl,''))
                if block_cols:
                    avg = sub[block_cols].mean()
                    fig = px.bar(x=block_cols, y=avg.values, color_discrete_sequence=[clr])
                    fig.update_layout(height=180, margin=dict(l=0,r=0,t=10,b=30))
                    st.plotly_chart(fig, use_container_width=True)
                ex = sub.nlargest(5,'score_final')[['mah_name','ilce_name','score_final']]
                st.dataframe(ex, hide_index=True, use_container_width=True)

    # â”€â”€ HARITA AYARLARI (sidebar en alt) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.divider()
    st.markdown("**ğŸ¨ Harita Filtreleri**")
    cluster_filter = st.multiselect("KÃ¼me:", list(CLUSTER_COLORS.keys()),
                                     default=list(CLUSTER_COLORS.keys()), key="sb_cl")
    geo_filter = st.multiselect("Geo Tip:", list(GEO_COLORS.keys()),
                                 default=list(GEO_COLORS.keys()), key="sb_geo")
    # Hidden cycle button (JS finds this by text)
    if st.button("CYCLE_MAP", key="cycle_mode_btn", use_container_width=True):
        st.session_state.color_by = next_mode
        st.rerun()

# â”€â”€ MAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
gdf_map = gdf[['mah_id','geometry']].merge(master, on='mah_id', how='left')
mask_map = gdf_map['cluster_label'].isin(cluster_filter) & gdf_map['geo_type'].isin(geo_filter)
gdf_show = gdf_map[mask_map].copy()

# Searched mahalle
highlight_geom = None
if search and search in name_to_id:
    mid = name_to_id[search]
    highlight_geom = gdf_map[gdf_map['mah_id'] == mid]

def build_map():
    m = folium.Map(location=[36.9, 30.7], zoom_start=9, tiles='CartoDB dark_matter')
    def style_fn(feat):
        p = feat['properties']
        s = p.get('score_final')
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = 0
        if color_by=="Kume": fc = CLUSTER_COLORS.get(p.get('cluster_label',''),'#888')
        elif color_by=="Geo Tip": fc = GEO_COLORS.get(p.get('geo_type',''),'#888')
        else: fc = score_color(s)
        return {'fillColor':fc,'color':'#333','weight':0.5,'fillOpacity':0.72}
    tf = [f for f in ['mah_name','ilce_name','score_final','cluster_label','geo_type'] if f in gdf_show.columns]
    folium.GeoJson(gdf_show, style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tf,
            aliases=['Mahalle:','Ä°lÃ§e:','Skor:','KÃ¼me:','Tip:'],
            style="font-size:12px;padding:8px;background:rgba(0,0,0,0.85);color:#fff;border-radius:8px;border:1px solid rgba(255,255,255,0.15);")
    ).add_to(m)
    # Highlight searched
    if highlight_geom is not None and len(highlight_geom) > 0:
        folium.GeoJson(highlight_geom,
            style_function=lambda x: {'fillColor':'#ffe600','color':'#ffe600','weight':3,'fillOpacity':0.45,'dashArray':'6'},
            name="Aranan"
        ).add_to(m)
        b = highlight_geom.total_bounds
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m

map_data = st_folium(build_map(), width=None, height=930, key="main_map",
                      returned_objects=["last_object_clicked"])

# â”€â”€ FIND CLICKED MAHALLE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
detail_mah_id = None

# From search
if search and search in name_to_id:
    detail_mah_id = name_to_id[search]
    st.session_state.show_detail = True

# From map click (override)
if map_data and map_data.get("last_object_clicked"):
    click = map_data["last_object_clicked"]
    if click and 'lat' in click and 'lng' in click:
        pt = Point(click['lng'], click['lat'])
        gdf_check = gdf_map.copy()
        gdf_check['contains'] = gdf_check.geometry.contains(pt)
        hit = gdf_check[gdf_check['contains']]
        if len(hit) > 0:
            detail_mah_id = hit.iloc[0]['mah_id']
            st.session_state.show_detail = True

# Check if panel was closed
if not st.session_state.get('show_detail', True):
    detail_mah_id = None

# â”€â”€ DETAIL PANEL (bottom-right overlay - comprehensive) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if detail_mah_id:
    r = master[master['mah_id']==detail_mah_id].iloc[0]
    raw_r = raw_df[raw_df['mah_id']==detail_mah_id].iloc[0]
    sc = r['score_final']
    sc_col = score_color(sc)

    # â”€â”€ Section builder helpers
    def m(label, val):
        return f'<div style="display:inline-block;margin:3px 4px;padding:5px 10px;background:rgba(255,255,255,0.06);border-radius:6px;font-size:0.8rem;"><span style="color:#888;font-size:0.65rem;">{label}</span><br><b style="color:#e0e0e0;">{val}</b></div>'
    def section(title):
        return f'<div style="color:#64b5f6;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px 0;">{title}</div>'
    def divider():
        return '<div style="border-top:1px solid rgba(255,255,255,0.07);margin:8px 0;"></div>'

    # â”€â”€ Header (with close button)
    html = f"""<div id="detail-panel" style="position:fixed;bottom:16px;right:16px;width:400px;max-height:80vh;
        background:rgba(14,14,20,0.95);backdrop-filter:blur(16px);
        border:1px solid rgba(255,255,255,0.1);border-radius:14px;padding:16px 18px;
        z-index:99999;overflow-y:auto;box-shadow:0 8px 40px rgba(0,0,0,0.6);
        font-family:'Inter',sans-serif;color:#d0d0d0;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div><b style="color:#fff;font-size:1.1rem;">ğŸ“ {r['mah_name']}</b>
            <span style="color:#777;font-size:0.8rem;"> ({r['ilce_name']})</span></div>
            <div style="display:flex;gap:8px;align-items:center;">
                <div style="background:{sc_col};color:#fff;padding:4px 12px;border-radius:8px;font-weight:700;font-size:1rem;">{sc:.0f}</div>
                <div onclick="this.closest('#detail-panel').style.display='none'" style="cursor:pointer;color:#888;font-size:1.2rem;width:28px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:rgba(255,255,255,0.06);transition:all 0.2s;" onmouseenter="this.style.background='rgba(255,255,255,0.15)';this.style.color='#fff'" onmouseleave="this.style.background='rgba(255,255,255,0.06)';this.style.color='#888'">âœ•</div>
            </div>
        </div>
        <div style="margin-top:6px;">
            <span style="background:rgba(255,255,255,0.08);padding:3px 8px;border-radius:5px;font-size:0.75rem;margin-right:4px;">ğŸ·ï¸ {r.get('cluster_label','-')}</span>
            <span style="background:rgba(255,255,255,0.08);padding:3px 8px;border-radius:5px;font-size:0.75rem;">ğŸŒ {r.get('geo_type','-')}</span>
        </div>"""

    # â”€â”€ 5Y Future prediction
    if pd.notna(r.get('predicted_score_5y')):
        c = r.get('score_change_5y',0)
        ar = 'â–²' if c>0 else 'â–¼' if c<0 else 'â–¸'
        col = '#4caf50' if c>0 else '#f44336' if c<0 else '#999'
        html += f"""<div style="margin-top:8px;padding:8px 12px;background:rgba(255,255,255,0.05);border-radius:8px;">
            <span style="color:#aaa;font-size:0.75rem;">5 YÄ±llÄ±k Tahmin</span>
            <b style="color:{col};margin-left:6px;font-size:1rem;">{r['predicted_score_5y']:.1f}</b>
            <span style="color:{col};font-size:0.85rem;"> {ar} {abs(c):.1f}</span>
            <span style="color:#aaa;margin-left:8px;font-size:0.75rem;">| {r.get('future_class','-')}</span>
        </div>"""

    html += divider()

    # â”€â”€ Temel bilgiler
    html += section('ğŸ“‹ Temel Bilgiler')
    raw_items = [
        ('NÃ¼fus', f"{raw_r.get('pop',0):,.0f}"),
        ('Alan', f"{raw_r.get('area_m2_gee',0)/1e6:.2f} kmÂ²"),
        ('RakÄ±m', f"{raw_r.get('mean_elevation_m',0):.0f} m"),
        ('NÃ¼fus YoÄŸ.', f"{raw_r.get('ghsl_pop_density_2020',0):.0f}/kmÂ²"),
        ('POI YoÄŸ.', f"{raw_r.get('poi_density_per_km2',0):.1f}"),
        ('POI Ã‡eÅŸitlilik', f"{raw_r.get('poi_type_entropy_norm',0):.2f}"),
        ('YeÅŸil Oran', f"%{raw_r.get('green_natural_share',0)*100:.1f}"),
        ('AÄŸaÃ§ Oran', f"%{raw_r.get('tree_share',0)*100:.1f}"),
        ('GÃ¼rÃ¼ltÃ¼', f"{raw_r.get('noise_density_per_km2',0):.1f}"),
        ('Yaz SÄ±caklÄ±k', f"{raw_r.get('mean_summer_lst_c',0):.1f}Â°C"),
        ('Bina YÃ¼k.', f"{raw_r.get('built_height_2018_mean',0):.1f} m"),
        ('Sokak YoÄŸ.', f"{raw_r.get('street_density_km_per_km2',0):.1f} km/kmÂ²"),
    ]
    html += ''.join([m(l,v) for l,v in raw_items])

    # â”€â”€ Block scores
    if block_cols:
        bvals = {c: r.get(c,0) for c in block_cols if pd.notna(r.get(c))}
        if bvals:
            html += divider() + section('ğŸ“Š Blok SkorlarÄ±')
            max_v = max(bvals.values()) if max(bvals.values()) > 0 else 1
            for k, v in bvals.items():
                pct = min(100, v / max_v * 100)
                label = k.replace('_',' ').title()
                html += f'<div style="margin:4px 0;"><div style="display:flex;justify-content:space-between;font-size:0.75rem;"><span style="color:#bbb;">{label}</span><span style="color:#64b5f6;font-weight:600;">{v:.1f}</span></div><div style="height:6px;background:rgba(255,255,255,0.08);border-radius:3px;margin-top:2px;"><div style="height:100%;width:{pct}%;background:linear-gradient(to right,#1565c0,#64b5f6);border-radius:3px;"></div></div></div>'

    # â”€â”€ Sub scores
    sub_cols_show = [c for c in r.index if c.endswith('_score') and c not in ['score_final','score_within_type'] and pd.notna(r.get(c))]
    if sub_cols_show:
        html += divider() + section('ğŸ“ˆ Alt Skorlar (0-100)')
        for c in sub_cols_show:
            v = r[c]
            label = c.replace('_score','').replace('_',' ').title()
            col = '#4caf50' if v >= 70 else '#ffb300' if v >= 40 else '#f44336'
            html += f'<div style="display:inline-block;margin:2px 3px;padding:4px 8px;background:rgba(255,255,255,0.05);border-radius:5px;font-size:0.75rem;border-left:3px solid {col};"><span style="color:#999;">{label}</span> <b style="color:#e0e0e0;">{v:.0f}</b></div>'

    # â”€â”€ POI counts
    poi_vals = {poi_labels[c]: int(raw_r.get(c,0)) for c in poi_cols if raw_r.get(c,0) > 0}
    if poi_vals:
        html += divider() + section(f'ğŸ“ POI SayÄ±larÄ± ({len(poi_vals)} tÃ¼r)')
        for k, v in sorted(poi_vals.items(), key=lambda x: -x[1]):
            html += m(k, str(v))

    # â”€â”€ Walkability
    walk_cols = [c for c in raw_r.index if c.startswith('walk_') and raw_r.get(c,0) > 0]
    if walk_cols:
        html += divider() + section('ğŸš¶ YÃ¼rÃ¼nebilirlik')
        for c in walk_cols[:8]:
            label = c.replace('walk_','').replace('_within_','<').replace('min_share','dk').replace('min_mean','dk').replace('_',' ').title()
            v = raw_r.get(c, 0)
            if isinstance(v, float): v_str = f"{v:.2f}"
            else: v_str = str(v)
            html += m(label, v_str)

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
