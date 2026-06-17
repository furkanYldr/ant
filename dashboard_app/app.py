"""
STREAMLIT APP v4 — Antalya Mahalle Karar Destek Sistemi
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

# Path: hem lokal hem Streamlit Cloud icin calisir
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_SCRIPT_DIR, "outputs")
GEO = os.path.join(_SCRIPT_DIR, "00_base_mahalle_final_913_clean.geojson")

st.set_page_config(page_title="Antalya Neighborhood Livability", page_icon="🏘️", layout="wide",
                    initial_sidebar_state="collapsed")

# ── CSS ───────────────────────────────────────────────────────────────
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
/* Sidebar widget dark backgrounds */
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="popover"] {background:#1a1a24!important;border-color:rgba(255,255,255,0.1)!important;}
section[data-testid="stSidebar"] [data-baseweb="tag"] {background:#1565c0!important;color:#fff!important;}
section[data-testid="stSidebar"] [data-baseweb="menu"],
section[data-testid="stSidebar"] [role="listbox"] {background:#1a1a24!important;}
section[data-testid="stSidebar"] [role="option"] {background:#1a1a24!important;color:#d0d0d0!important;}
section[data-testid="stSidebar"] [role="option"]:hover {background:#252536!important;}
section[data-testid="stSidebar"] [data-testid="stExpander"] {background:#15151f!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:8px!important;}
section[data-testid="stSidebar"] [data-testid="stDataFrame"],
section[data-testid="stSidebar"] .stDataFrame {background:#15151f!important;}
section[data-testid="stSidebar"] [data-testid="stDataFrame"] * {background:transparent!important;color:#d0d0d0!important;}
section[data-testid="stSidebar"] [data-testid="stAlert"] {background:#15151f!important;border-color:rgba(255,255,255,0.08)!important;}
section[data-testid="stSidebar"] [data-testid="stAlert"] p {color:#ccc!important;}
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {background:transparent!important;color:#999!important;}
section[data-testid="stSidebar"] .stTabs [aria-selected="true"] {color:#64b5f6!important;border-bottom-color:#64b5f6!important;}
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-panel"] {background:transparent!important;}
section[data-testid="stSidebar"] button {background:#1a1a24!important;border-color:rgba(255,255,255,0.1)!important;color:#d0d0d0!important;}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {background:#1565c0!important;color:#fff!important;border:none!important;}
section[data-testid="stSidebar"] [data-testid="stSlider"] * {color:#ccc!important;}
/* Hide ALL native sidebar toggle buttons */
button[data-testid="stBaseButton-headerNoPadding"],
button[data-testid="stSidebarCollapseButton"],
div[data-testid="collapsedControl"] {display:none!important;}
.stElementContainer {margin:0!important;padding:0!important;}
div[data-testid="stVerticalBlock"] > div {gap:0!important;}
iframe {width:100%!important;border:none!important;}
/* Map fills entire viewport — target iframe + all parent containers */
iframe[title="streamlit_folium.st_folium"] {height:100vh!important;min-height:100vh!important;}
iframe[title="streamlit_folium.st_folium"],
iframe[title="streamlit_folium.st_folium"] ~ div,
div:has(> iframe[title="streamlit_folium.st_folium"]) {height:100vh!important;min-height:100vh!important;}
[data-testid="stAppViewContainer"] > section > div {padding:0!important;}
header[data-testid="stHeader"] {height:0!important;padding:0!important;overflow:hidden!important;}
/* Remove bottom padding from main block */
.stMainBlockContainer, [data-testid="stMainBlockContainer"] {padding-top:0!important;padding-bottom:0!important;}
::-webkit-scrollbar {width:5px;}
::-webkit-scrollbar-thumb {background:rgba(255,255,255,0.15);border-radius:3px;}
@media (max-width:768px) {
    section[data-testid="stSidebar"] {width:100vw!important;}
    #detail-panel {left:50%!important;right:auto!important;transform:translateX(-50%)!important;width:calc(100vw - 24px)!important;max-height:60vh!important;bottom:8px!important;font-size:0.85rem!important;}
}
</style>
""", unsafe_allow_html=True)

# ── HAMBURGER + MAP SWITCH (js via components.html) ──────────────────
if 'color_by' not in st.session_state:
    st.session_state.color_by = 'Skor'
color_by = st.session_state.color_by

MODE_ICONS = {'Skor':'🎯','Kume':'🏷️','Geo Tip':'🌍'}
MODE_CYCLE = {'Skor':'Kume','Kume':'Geo Tip','Geo Tip':'Skor'}
cur_icon = MODE_ICONS.get(color_by,'🎯')
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




# ── DATA ──────────────────────────────────────────────────────────────
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

# ── Translate Turkish labels to English in data ──
_LABEL_TR_EN = {
    'A - En Iyi':'A - Best','B - Iyi':'B - Good','C - Ortanin Ustu':'C - Above Avg',
    'D - Orta':'D - Average','E - Dusuk':'E - Low','F - En Dusuk':'F - Lowest',
    'Sehir Merkezi':'City Center','Sahil / Turistik':'Coastal / Touristic',
    'Sehir Siniri':'City Fringe','Turistik Sahil Kasabasi':'Touristic Coastal Town',
    'Kucuk Kasaba':'Small Town','Kirsal / Ovacik':'Rural / Flatland',
    'Dagl\u0131k / Yukari Yerlesim':'Mountainous / Highland',
    'Yuksek Potansiyel':'High Potential','Orta Potansiyel':'Mid Potential',
    'Dusuk Potansiyel':'Low Potential','Stabil':'Stable','Duragan':'Stagnant',
}
for col in ['cluster_label','geo_type','future_class']:
    if col in master.columns:
        master[col] = master[col].map(lambda x: _LABEL_TR_EN.get(str(x), x) if pd.notna(x) else x)

name_options = sorted([f"{r['mah_name']} ({r['ilce_name']})" for _, r in master[['mah_name','ilce_name']].drop_duplicates().iterrows()])
name_to_id = {f"{r['mah_name']} ({r['ilce_name']})": r['mah_id'] for _, r in master.iterrows()}

CLUSTER_COLORS = {'A - Best':'#00c853','B - Good':'#2979ff','C - Above Avg':'#ffd600',
                  'D - Average':'#ff6d00','E - Low':'#aa00ff','F - Lowest':'#d50000'}
GEO_COLORS = {'City Center':'#e53935','Coastal / Touristic':'#1e88e5','City Fringe':'#43a047',
              'Touristic Coastal Town':'#00acc1','Small Town':'#fb8c00',
              'Rural / Flatland':'#8d6e63','Mountainous / Highland':'#757575'}

def score_color(s):
    if s >= 70: return '#1a9641'
    if s >= 50: return '#a6d96a'
    if s >= 30: return '#fee08b'
    if s >= 15: return '#f46d43'
    return '#d73027'




# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏘️ Antalya Neighborhoods")
    page = st.radio("Tab:", ["🗺️ Overview","⚖️ Compare","💡 Recommend","📊 Clusters"],
                     label_visibility="collapsed")
    st.divider()
    srch_col, clr_col = st.columns([5,1])
    with srch_col:
        search = st.selectbox("🔍 Search Neighborhood:", [""] + name_options, key="detail_search")
    with clr_col:
        st.markdown("<br>", unsafe_allow_html=True)
        def _clear_search():
            st.session_state["detail_search"] = ""
            st.session_state.pop('show_detail', None)
        st.button("✕", key="clear_search", help="Clear search", on_click=_clear_search)
    st.divider()

    # ── TAB CONTENT ───────────────────────────────────────────────
    if page == "🗺️ Overview":
        st.subheader("📊 Overview")
        st.caption(f"Total: {len(master)} neighborhoods")
        mc1, mc2 = st.columns(2)
        with mc1: st.metric("Avg. Score", f"{master['score_final'].mean():.1f}")
        with mc2: st.metric("Max Score", f"{master['score_final'].max():.1f}")
        if 'pop' in master.columns:
            st.metric("Total Population", f"{master['pop'].sum():,.0f}")
        st.markdown("**Cluster Distribution**")
        c_order = ['A - Best','B - Good','C - Above Avg','D - Average','E - Low','F - Lowest']
        rows = []
        for cl in c_order:
            sub = master[master['cluster_label']==cl]
            if len(sub)==0: continue
            rows.append({'Cluster':cl,'N':len(sub),'Score':f"{sub['score_final'].mean():.0f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        fig_hist = px.histogram(master, x='score_final', nbins=30, color='cluster_label',
                                 color_discrete_map=CLUSTER_COLORS, category_orders={'cluster_label':c_order})
        fig_hist.update_layout(height=250, xaxis_title="Score", yaxis_title="Count",
                                margin=dict(l=0,r=0,t=10,b=30))
        st.plotly_chart(fig_hist, use_container_width=True)

    elif page == "⚖️ Compare":
        st.subheader("⚖️ Comparison")
        if 'compare_list' not in st.session_state: st.session_state.compare_list = []

        pick = st.selectbox("Select neighborhood:", [""] + name_options, key="cmp_pick")
        def _add_compare():
            p = st.session_state.get("cmp_pick", "")
            if p and p not in st.session_state.compare_list and len(st.session_state.compare_list) < 5:
                st.session_state.compare_list.append(p)
            st.session_state["cmp_pick"] = ""
        st.button("➕ Add", type="primary", on_click=_add_compare)
        # Show selected neighborhoods — click X to remove
        if st.session_state.compare_list:
            kept = st.multiselect("Selected (click X to remove):",
                                   st.session_state.compare_list,
                                   default=st.session_state.compare_list,
                                   key="cmp_kept")
            if set(kept) != set(st.session_state.compare_list):
                st.session_state.compare_list = list(kept)
                st.rerun()
        if len(st.session_state.compare_list) >= 2:
            sel_ids = [name_to_id.get(n) for n in st.session_state.compare_list if n in name_to_id]
            comp = master[master['mah_id'].isin(sel_ids)].copy()
            comp['display'] = comp.apply(lambda r: f"{r['mah_name']} ({r['ilce_name']})", axis=1)
            # Score cards with district name
            cols = st.columns(len(comp))
            palette = ['#42a5f5','#ef5350','#66bb6a','#ffa726','#ab47bc']
            for i, (_, r) in enumerate(comp.iterrows()):
                with cols[i]:
                    clr = palette[i % len(palette)]
                    st.markdown(f"""<div style="background:rgba(255,255,255,0.05);border-left:4px solid {clr};
                        border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                        <div style="font-size:0.85rem;font-weight:600;color:#fff;">{r['mah_name']}</div>
                        <div style="font-size:0.65rem;color:#888;">{r['ilce_name']}</div>
                        <div style="font-size:1.5rem;font-weight:700;color:{clr};margin-top:4px;">{r['score_final']:.0f}</div>
                    </div>""", unsafe_allow_html=True)
            radar_cols = [c for c in block_cols if c in comp.columns]
            if radar_cols:
                # Readable labels
                label_map = {'demographics':'Demographics','built_activity':'Built Activity',
                             'accessibility':'Accessibility','morphology':'Morphology',
                             'environment':'Environment','seasonality':'Seasonality'}
                labels = [label_map.get(c, c.replace('_',' ').title()) for c in radar_cols]
                # Radar chart
                fig = go.Figure()
                for i, (_, row) in enumerate(comp.iterrows()):
                    vals = [row.get(c,0) for c in radar_cols]
                    clr = palette[i % len(palette)]
                    fig.add_trace(go.Scatterpolar(
                        r=vals+[vals[0]], theta=labels+[labels[0]],
                        fill='toself', name=row['display'],
                        opacity=0.35, line=dict(color=clr, width=2),
                        fillcolor=clr
                    ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, showline=False, gridcolor='rgba(255,255,255,0.1)',
                                       tickfont=dict(size=9, color='#888')),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                                        tickfont=dict(size=11, color='#ccc')),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    height=380, margin=dict(l=40,r=40,t=30,b=30),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(size=10, color='#ccc'), orientation='h',
                                yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
                # Grouped bar chart for easier comparison
                import plotly.express as px
                bar_data = []
                for _, row in comp.iterrows():
                    for c, l in zip(radar_cols, labels):
                        bar_data.append({'Neighborhood': row['display'], 'Block': l, 'Score': row.get(c,0)})
                bar_df = pd.DataFrame(bar_data)
                fig_bar = px.bar(bar_df, x='Block', y='Score', color='Neighborhood',
                                barmode='group', color_discrete_sequence=palette[:len(comp)])
                fig_bar.update_layout(
                    height=280, margin=dict(l=0,r=0,t=10,b=30),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#ccc')),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickfont=dict(color='#999'),
                              title='', zeroline=True, zerolinecolor='rgba(255,255,255,0.15)'),
                    legend=dict(font=dict(size=9, color='#ccc'), orientation='h',
                                yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Add at least 2 neighborhoods.")

    elif page == "💡 Recommend":
        st.subheader("💡 Neighborhood Recommender")
        tab1, tab2 = st.tabs(["Ready Profiles", "Custom Filter"])
        with tab1:
            sel_p = st.selectbox("Profile:", list(PERSONAS.keys()))
            st.info(PERSONAS[sel_p]['description'])
            ilce_f = st.selectbox("District:", ['All'] + sorted(raw_df['ilce_name'].dropna().unique()))
            top_n = st.slider("Count:", 5, 20, 10)
            if st.button("Show Results", type="primary", key="btn_p"):
                res = recommend(raw_df, sel_p, top_n=top_n,
                                ilce_filter=None if ilce_f=='All' else ilce_f)
                for _, r in res.iterrows():
                    st.write(f"**#{int(r['sira'])}** {r['mah_name']} ({r['ilce_name']}) — **{r['persona_score']:.0f}**")
        with tab2:
            st.markdown("**Set your own preferences**")
            sub_scores = build_sub_scores(raw_df)
            w_opts = {'education_score':'Education','health_score':'Healthcare','green_score':'Green Area',
                       'quiet_score':'Quietness','safety_score':'Safety','transport_score':'Transit',
                       'walkability_score':'Walkability','social_score':'Social','affordability_score':'Affordability',
                       'coastal_proximity':'Coastal'}
            weights = {}
            wc = st.columns(2)
            for i, (wk, wl) in enumerate(w_opts.items()):
                with wc[i%2]: weights[wk] = st.slider(wl, 0, 10, 5, key=f"w_{wk}")
            fc1, fc2 = st.columns(2)
            with fc1: ilce_f2 = st.multiselect("District:", sorted(raw_df['ilce_name'].dropna().unique()), key="of_ilce")
            with fc2: min_score2 = st.slider("Min Score:", 0, 80, 0)
            if st.button("Filter", type="primary", key="btn_o"):
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
                st.write(f"**{mask.sum()} neighborhoods matched.** Top 15:")
                for rank, idx in enumerate(top_idx, 1):
                    if not mask[idx]: break
                    r = raw_df.iloc[idx]
                    st.write(f"**#{rank}** {r['mah_name']} ({r['ilce_name']}) — {custom[idx]:.0f}")

    elif page == "📊 Clusters":
        st.subheader("📊 Clustering (6-Tier)")
        CDESC = {'A - Best':"Central, service-dense, high population.","B - Good":"City fringe, mid-high accessibility.",
                  'C - Above Avg':"Coastal / touristic areas.",'D - Average':"Small towns.",
                  'E - Low':"Rural, limited infrastructure.",'F - Lowest':"Mountainous, hard to access."}
        c_order = ['A - Best','B - Good','C - Above Avg','D - Average','E - Low','F - Lowest']
        for cl in c_order:
            sub = master[master['cluster_label']==cl]
            if len(sub)==0: continue
            clr = CLUSTER_COLORS.get(cl,'#888')
            with st.expander(f"{cl} ({len(sub)} nbhd.)", expanded=(cl=='A - Best')):
                st.caption(CDESC.get(cl,''))
                if block_cols:
                    avg = sub[block_cols].mean()
                    fig = px.bar(x=block_cols, y=avg.values, color_discrete_sequence=[clr])
                    fig.update_layout(height=180, margin=dict(l=0,r=0,t=10,b=30))
                    st.plotly_chart(fig, use_container_width=True)
                ex = sub.nlargest(5,'score_final')[['mah_name','ilce_name','score_final']]
                st.dataframe(ex, hide_index=True, use_container_width=True)

    # ── HARITA AYARLARI (sidebar en alt) ────────────────────────────
    st.divider()
    st.markdown("**🎨 Map Filters**")
    cluster_filter = st.multiselect("Cluster:", list(CLUSTER_COLORS.keys()),
                                     default=list(CLUSTER_COLORS.keys()), key="sb_cl")
    geo_filter = st.multiselect("Geo Type:", list(GEO_COLORS.keys()),
                                 default=list(GEO_COLORS.keys()), key="sb_geo")
    # Hidden cycle button (JS finds this by text)
    if st.button("CYCLE_MAP", key="cycle_mode_btn", use_container_width=True):
        st.session_state.color_by = next_mode
        st.rerun()
    # Hidden close-detail button (JS finds this by text)
    if st.button("CLOSE_DETAIL", key="close_detail_btn", use_container_width=True):
        st.session_state.detail_closed_click = (st.session_state.get('detail_closed_click', 0) or 0) + 1
        st.rerun()

# ── MAP ───────────────────────────────────────────────────────────────
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
    al = ['Neighborhood:','District:','Score:','Cluster:','Type:']
    if len(gdf_show) > 0 and tf:
        folium.GeoJson(gdf_show, style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=tf,
                aliases=al[:len(tf)],
                style="font-size:12px;padding:8px;background:rgba(0,0,0,0.85);color:#fff;border-radius:8px;border:1px solid rgba(255,255,255,0.15);")
        ).add_to(m)
    elif len(gdf_show) > 0:
        folium.GeoJson(gdf_show, style_function=style_fn).add_to(m)
    # Highlight searched
    if highlight_geom is not None and len(highlight_geom) > 0:
        folium.GeoJson(highlight_geom,
            style_function=lambda x: {'fillColor':'#ffe600','color':'#ffe600','weight':3,'fillOpacity':0.45,'dashArray':'6'},
            name="Searched"
        ).add_to(m)
        b = highlight_geom.total_bounds
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m

map_data = st_folium(build_map(), width=None, height=2500, key="main_map",
                      returned_objects=["last_object_clicked"])

# ── FIND CLICKED MAHALLE ──────────────────────────────────────────────
detail_mah_id = None
panel_visible = True

# From search
if search and search in name_to_id:
    detail_mah_id = name_to_id[search]
    st.session_state.last_detail_source = 'search'
    st.session_state.pop('detail_closed_click', None)  # Arama yapilinca panel acilsin

# From map click (override)
if map_data and map_data.get("last_object_clicked"):
    click = map_data["last_object_clicked"]
    click_key = f"{click.get('lat',0):.6f}_{click.get('lng',0):.6f}" if click else ''
    if click and 'lat' in click and 'lng' in click:
        # Yeni tiklama mi kontrol et
        if click_key != st.session_state.get('last_click_key', ''):
            st.session_state.last_click_key = click_key
            st.session_state.pop('detail_closed_click', None)  # Panel tekrar acilsin
        pt = Point(click['lng'], click['lat'])
        gdf_check = gdf_map.copy()
        gdf_check['contains'] = gdf_check.geometry.contains(pt)
        hit = gdf_check[gdf_check['contains']]
        if len(hit) > 0:
            detail_mah_id = hit.iloc[0]['mah_id']
            st.session_state.last_detail_source = 'map'

# Panel kapali mi? (gizli buton tiklanmissa)
if st.session_state.get('detail_closed_click', 0):
    panel_visible = False
    # Ama arama secildiyse her zaman goster
    if st.session_state.get('last_detail_source') == 'search' and search:
        panel_visible = True

# ── DETAIL PANEL (bottom-right overlay - comprehensive) ───────────────
if detail_mah_id and panel_visible:
    r = master[master['mah_id']==detail_mah_id].iloc[0]
    raw_r = raw_df[raw_df['mah_id']==detail_mah_id].iloc[0]
    sc = r['score_final']
    sc_col = score_color(sc)

    # ── Section builder helpers
    def m(label, val):
        return f'<div style="display:inline-block;margin:3px 4px;padding:5px 10px;background:rgba(255,255,255,0.06);border-radius:6px;font-size:0.8rem;"><span style="color:#888;font-size:0.65rem;">{label}</span><br><b style="color:#e0e0e0;">{val}</b></div>'
    def section(title):
        return f'<div style="color:#64b5f6;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px 0;">{title}</div>'
    def divider():
        return '<div style="border-top:1px solid rgba(255,255,255,0.07);margin:8px 0;"></div>'

    # Close button - onclick JS ile ekleniyor (stc.html icinde)
    close_btn = '<div id="detail-close-btn" style="cursor:pointer;color:#888;font-size:1.2rem;width:28px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:rgba(255,255,255,0.06);transition:all 0.2s;">✕</div>'

    # ── Header (with close button)
    html = f"""<div id="detail-panel" style="position:fixed;bottom:16px;right:16px;width:400px;max-width:calc(100vw - 32px);max-height:80vh;
        background:rgba(14,14,20,0.95);backdrop-filter:blur(16px);
        border:1px solid rgba(255,255,255,0.1);border-radius:14px;padding:16px 18px;
        z-index:99999;overflow-y:auto;box-shadow:0 8px 40px rgba(0,0,0,0.6);
        font-family:'Inter',sans-serif;color:#d0d0d0;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div><b style="color:#fff;font-size:1.1rem;">📍 {r['mah_name']}</b>
            <span style="color:#777;font-size:0.8rem;"> ({r['ilce_name']})</span></div>
            <div style="display:flex;gap:8px;align-items:center;">
                <div style="background:{sc_col};color:#fff;padding:4px 12px;border-radius:8px;font-weight:700;font-size:1rem;">{sc:.0f}</div>
                {close_btn}
            </div>
        </div>
        <div style="margin-top:6px;">
            <span style="background:rgba(255,255,255,0.08);padding:3px 8px;border-radius:5px;font-size:0.75rem;margin-right:4px;">🏷️ {r.get('cluster_label','-')}</span>
            <span style="background:rgba(255,255,255,0.08);padding:3px 8px;border-radius:5px;font-size:0.75rem;">🌍 {r.get('geo_type','-')}</span>
        </div>"""

    # ── 5Y Future prediction
    if pd.notna(r.get('predicted_score_5y')):
        c = r.get('score_change_5y',0)
        ar = '▲' if c>0 else '▼' if c<0 else '▸'
        col = '#4caf50' if c>0 else '#f44336' if c<0 else '#999'
        html += f"""<div style="margin-top:8px;padding:8px 12px;background:rgba(255,255,255,0.05);border-radius:8px;">
            <span style="color:#aaa;font-size:0.75rem;">5-Year Forecast</span>
            <b style="color:{col};margin-left:6px;font-size:1rem;">{r['predicted_score_5y']:.1f}</b>
            <span style="color:{col};font-size:0.85rem;"> {ar} {abs(c):.1f}</span>
            <span style="color:#aaa;margin-left:8px;font-size:0.75rem;">| {r.get('future_class','-')}</span>
        </div>"""

    html += divider()

    # ── Temel bilgiler
    html += section('📋 Key Facts')
    raw_items = [
        ('Population', f"{raw_r.get('pop',0):,.0f}"),
        ('Area', f"{raw_r.get('area_m2_gee',0)/1e6:.2f} km²"),
        ('Elevation', f"{raw_r.get('mean_elevation_m',0):.0f} m"),
        ('Pop. Density', f"{raw_r.get('ghsl_pop_density_2020',0):.0f}/km²"),
        ('POI Density', f"{raw_r.get('poi_density_per_km2',0):.1f}"),
        ('POI Diversity', f"{raw_r.get('poi_type_entropy_norm',0):.2f}"),
        ('Green Ratio', f"{raw_r.get('green_natural_share',0)*100:.1f}%"),
        ('Tree Cover', f"{raw_r.get('tree_share',0)*100:.1f}%"),
        ('Noise', f"{raw_r.get('noise_density_per_km2',0):.1f}"),
        ('Summer Temp', f"{raw_r.get('mean_summer_lst_c',0):.1f}°C"),
        ('Bldg Height', f"{raw_r.get('built_height_2018_mean',0):.1f} m"),
        ('Street Dens.', f"{raw_r.get('street_density_km_per_km2',0):.1f} km/km²"),
    ]
    html += ''.join([m(l,v) for l,v in raw_items])

    # ── Block scores (diverging bar: negatif=kirmizi sola, pozitif=yesil saga)
    if block_cols:
        bvals = {c: r.get(c,0) for c in block_cols if pd.notna(r.get(c))}
        if bvals:
            html += divider() + section('📊 Block Scores')
            SCALE = 10.0  # sabit olcek: -10 ile +10 arasi
            for k, v in bvals.items():
                label = k.replace('_',' ').title()
                pct = min(50, abs(v) / SCALE * 50)  # max %50 (bir yon)
                val_color = '#4caf50' if v >= 0 else '#f44336'
                val_label_color = '#81c784' if v >= 0 else '#ef9a9a'
                if v >= 0:
                    bar_html = f'<div style="position:relative;height:8px;background:rgba(255,255,255,0.06);border-radius:4px;"><div style="position:absolute;left:50%;height:100%;width:{pct}%;background:{val_color};border-radius:0 4px 4px 0;"></div><div style="position:absolute;left:50%;top:-2px;width:1px;height:12px;background:rgba(255,255,255,0.2);"></div></div>'
                else:
                    bar_html = f'<div style="position:relative;height:8px;background:rgba(255,255,255,0.06);border-radius:4px;"><div style="position:absolute;right:50%;height:100%;width:{pct}%;background:{val_color};border-radius:4px 0 0 4px;"></div><div style="position:absolute;left:50%;top:-2px;width:1px;height:12px;background:rgba(255,255,255,0.2);"></div></div>'
                html += f'<div style="margin:5px 0;"><div style="display:flex;justify-content:space-between;font-size:0.75rem;"><span style="color:#bbb;">{label}</span><span style="color:{val_label_color};font-weight:600;">{v:+.1f}</span></div>{bar_html}</div>'

    # ── Sub scores
    sub_cols_show = [c for c in r.index if c.endswith('_score') and c not in ['score_final','score_within_type'] and pd.notna(r.get(c))]
    if sub_cols_show:
        html += divider() + section('📈 Sub-Scores (0-100)')
        for c in sub_cols_show:
            v = r[c]
            label = c.replace('_score','').replace('_',' ').title()
            col = '#4caf50' if v >= 70 else '#ffb300' if v >= 40 else '#f44336'
            html += f'<div style="display:inline-block;margin:2px 3px;padding:4px 8px;background:rgba(255,255,255,0.05);border-radius:5px;font-size:0.75rem;border-left:3px solid {col};"><span style="color:#999;">{label}</span> <b style="color:#e0e0e0;">{v:.0f}</b></div>'

    # ── POI counts
    poi_vals = {poi_labels[c]: int(raw_r.get(c,0)) for c in poi_cols if raw_r.get(c,0) > 0}
    if poi_vals:
        html += divider() + section(f'📍 POI Counts ({len(poi_vals)} types)')
        for k, v in sorted(poi_vals.items(), key=lambda x: -x[1]):
            html += m(k, str(v))

    # ── Walkability
    walk_cols = [c for c in raw_r.index if c.startswith('walk_') and raw_r.get(c,0) > 0]
    if walk_cols:
        html += divider() + section('🚶 Walkability')
        for c in walk_cols[:8]:
            label = c.replace('walk_','').replace('_within_','<').replace('min_share','min').replace('min_mean','min').replace('_',' ').title()
            v = raw_r.get(c, 0)
            if isinstance(v, float): v_str = f"{v:.2f}"
            else: v_str = str(v)
            html += m(label, v_str)

    html += '</div>'
    # Escape for JS string (tek tirnak icindeki tek tirnaklari escape et)
    html_escaped = html.replace('\\', '\\\\').replace("'", "\\'").replace('\n', ' ')
    # stc.html ile parent document'a ekle — hamburger butonu ile AYNI mekanizma
    # onclick CALISIR cunku bu gercek DOM elementi, st.markdown degil
    stc.html(f"""
    <script>
    (function() {{
        var pd = window.parent.document;
        var old = pd.getElementById('detail-panel');
        if (old) old.remove();
        var wrapper = pd.createElement('div');
        wrapper.innerHTML = '{html_escaped}';
        var panel = wrapper.firstChild;
        pd.body.appendChild(panel);
        // Close button
        var closeBtn = panel.querySelector('#detail-close-btn');
        if (closeBtn) {{
            closeBtn.onclick = function() {{
                panel.remove();
            }};
            closeBtn.onmouseenter = function() {{
                this.style.background = 'rgba(255,255,255,0.15)';
                this.style.color = '#fff';
            }};
            closeBtn.onmouseleave = function() {{
                this.style.background = 'rgba(255,255,255,0.06)';
                this.style.color = '#888';
            }};
        }}
    }})();
    </script>
    """, height=0)

