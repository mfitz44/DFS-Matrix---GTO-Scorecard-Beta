# streamlit run app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="DFS Matrix — GTO Scorecard", layout="wide")

import io

st.subheader("Get input templates")

# Header-only template
header_cols = ["Player","PlayerID","Salary","Event","Slate","FieldSize","Timestamp",
               "Vegas_WinProb","Vegas_Top20Prob","Vegas_LineMove","Handle_SharpRatio",
               "Social_Sentiment","Social_Volume","Social_RecencyScore",
               "Tout_ProjPts","Tout_Heat","Tout_ProjOwn",
               "SG_Long","SG_Recent","Course_Fit","Volatility_Index","CompField_Adjust",
               "Value_PPD_Baseline","Salary_Tier",
               "Hist_Own_TierAvg","Hist_Event_Result","Hist_CutRate"]

tmpl = io.StringIO()
tmpl.write(",".join(header_cols) + "\n")
st.download_button("Download input header CSV", tmpl.getvalue(),
                   file_name="gto_scorecard_input_template.csv", mime="text/csv")

# 3-row sample
sample_csv = """Player,PlayerID,Salary,Event,Slate,FieldSize,Timestamp,Vegas_WinProb,Vegas_Top20Prob,Vegas_LineMove,Handle_SharpRatio,Social_Sentiment,Social_Volume,Social_RecencyScore,Tout_ProjPts,Tout_Heat,Tout_ProjOwn,SG_Long,SG_Recent,Course_Fit,Volatility_Index,CompField_Adjust,Value_PPD_Baseline,Salary_Tier,Hist_Own_TierAvg,Hist_Event_Result,Hist_CutRate
Scottie Scheffler,1001,12000,TOUR Championship 2025,DK Classic,30,2025-08-19T12:00:00Z,0.22,0.85,0.004,1.30,0.30,380,0.85,95.0,0.80,0.28,1.35,0.90,0.45,0.80,0.20,8.0,Stud,0.26,0.65,1.00
Rory McIlroy,1002,11500,TOUR Championship 2025,DK Classic,30,2025-08-19T12:00:00Z,0.20,0.82,0.003,1.18,0.25,310,0.80,93.0,0.72,0.24,1.10,0.88,0.35,0.78,0.12,7.6,Stud,0.22,0.58,0.98
Wyndham Clark,1003,9300,TOUR Championship 2025,DK Classic,30,2025-08-19T12:00:00Z,0.08,0.58,0.002,1.05,0.10,140,0.70,82.0,0.35,0.10,0.35,0.55,-0.05,0.92,-0.10,7.2,Upper,0.12,0.35,0.94
"""
st.download_button("Download sample input CSV", sample_csv,
                   file_name="gto_scorecard_input_sample.csv", mime="text/csv")


# ---------------- Config ----------------
@dataclass
class Weights:
    # Market cluster (bubbles 1–3)
    w_betting: float = 0.50
    w_social: float = 0.20
    w_tout: float = 0.30
    alpha_po: float  = 0.75  # PO: S_Market^alpha * S_Perf^beta
    beta_po: float   = 0.25

    # Performance cluster (bubbles 4–6)
    w_adv: float     = 0.55
    w_price: float   = 0.25
    w_hist: float    = 0.20
    gamma_gto: float = 0.85  # GTO: S_Perf^gamma * S_Market^delta
    delta_gto: float = 0.15

DEFAULT_W = Weights()

# ---------------- Utils ----------------
def _safe_minmax(x: pd.Series) -> pd.Series:
    if x.isnull().all(): return pd.Series(0.5, index=x.index)
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(hi, lo): 
        return pd.Series(0.5, index=x.index)
    return (x.fillna(lo) - lo) / (hi - lo)

def _zscore(x: pd.Series) -> pd.Series:
    mu, sd = x.mean(skipna=True), x.std(ddof=0, skipna=True)
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd

def _geom_blend(parts, weights):
    parts = list(parts)
    w = np.array(list(weights), dtype=float)
    if w.sum() == 0: w = np.ones_like(w)
    w = w / w.sum()
    logsum = None
    for s, wi in zip(parts, w):
        s_clipped = s.clip(1e-6, 1.0)
        term = np.log(s_clipped) * wi
        logsum = term if logsum is None else (logsum + term)
    y = np.exp(logsum)
    return _safe_minmax(y)

# ---------------- Bubbles ----------------
def bubble_betting(df):
    win   = _safe_minmax(df.get('Vegas_WinProb', 0))
    top20 = _safe_minmax(df.get('Vegas_Top20Prob', 0))
    linem = _safe_minmax(df.get('Vegas_LineMove', 0))
    sharp = _safe_minmax(df.get('Handle_SharpRatio', 1))
    raw = 0.45*win + 0.25*top20 + 0.15*linem + 0.15*sharp
    return _safe_minmax(raw)

def bubble_social(df):
    sent = _safe_minmax((df.get('Social_Sentiment', 0) + 1) / 2.0)
    vol  = _safe_minmax(np.log1p(df.get('Social_Volume', 0)))
    rec  = _safe_minmax(df.get('Social_RecencyScore', 0))
    raw = 0.5*sent + 0.3*vol + 0.2*rec
    return _safe_minmax(raw)

def bubble_tout(df):
    proj = _safe_minmax(df.get('Tout_ProjPts', np.nan).fillna(0))
    heat = _safe_minmax(df.get('Tout_Heat', 0))
    pown = _safe_minmax(df.get('Tout_ProjOwn', 0))
    raw = 0.5*proj + 0.3*heat + 0.2*pown
    return _safe_minmax(raw)

def bubble_advanced(df):
    sgl = _safe_minmax(df.get('SG_Long', 0))
    sgr = _safe_minmax(df.get('SG_Recent', 0))
    fit = _safe_minmax((df.get('Course_Fit', 0) + 1) / 2.0)
    vol = 1 - _safe_minmax(df.get('Volatility_Index', 0))  # prefer stability by default
    comp= _safe_minmax((df.get('CompField_Adjust', 0) + 1) / 2.0)
    raw = 0.35*sgl + 0.30*sgr + 0.20*fit + 0.10*vol + 0.05*comp
    return _safe_minmax(raw)

def bubble_price(df):
    val = _safe_minmax(df.get('Value_PPD_Baseline', 0))
    tier_map = {'Stud':0.50,'Upper':0.60,'Mid':0.70,'Value':0.80,'Punt':0.75}
    tier = df.get('Salary_Tier', 'Mid').map(tier_map) if 'Salary_Tier' in df else pd.Series(0.7, index=df.index)
    raw = 0.8*val + 0.2*tier
    return _safe_minmax(raw)

def bubble_history(df):
    own = _safe_minmax(df.get('Hist_Own_TierAvg', 0))
    res = _safe_minmax((df.get('Hist_Event_Result', 0) + 1) / 2.0)
    cut = _safe_minmax(df.get('Hist_CutRate', 0))
    raw = 0.4*own + 0.4*res + 0.2*cut
    return _safe_minmax(raw)

# ---------------- Core calcs ----------------
def compute_clusters(df, w: Weights):
    df = df.copy()
    df['B1_Betting']  = bubble_betting(df)
    df['B2_Social']   = bubble_social(df)
    df['B3_Tout']     = bubble_tout(df)
    df['B4_Adv']      = bubble_advanced(df)
    df['B5_Price']    = bubble_price(df)
    df['B6_History']  = bubble_history(df)

    df['S_Market'] = _geom_blend([df['B1_Betting'], df['B2_Social'], df['B3_Tout']],
                                 [w.w_betting,      w.w_social,      w.w_tout])
    df['S_Perf']   = _geom_blend([df['B4_Adv'], df['B5_Price'], df['B6_History']],
                                 [w.w_adv,      w.w_price,      w.w_hist])
    return df

def compute_projections(df):
    df = df.copy()
    # Placeholder projection model — swap with your model later
    df['Proj_Mean'] = 20 * df['S_Perf'] + 2.5 * df['B5_Price']
    df['Proj_SD']   = ((6.0 * (1 - df['S_Perf'])) + 4.0 * (1 - (1 - df.get('Volatility_Index', 0)).clip(0,1))).clip(lower=3.0)

    mu, sd = df['Proj_Mean'], df['Proj_SD']
    df['Proj_1SD_Low']  = mu - sd
    df['Proj_1SD_High'] = mu + sd
    df['Proj_2SD_Low']  = mu - 2*sd
    df['Proj_2SD_High'] = mu + 2*sd
    df['Proj_P10']      = mu - 1.2816*sd
    df['Proj_P90']      = mu + 1.2816*sd
    return df

def compute_projected_ownership(df, w: Weights):
    raw = (df['S_Market'] ** w.alpha_po) * (df['S_Perf'] ** w.beta_po)
    tier_prior = df.get('Salary_Tier', 'Mid').map({'Stud':1.08,'Upper':1.05,'Mid':1.00,'Value':0.95,'Punt':0.92}) \
                 if 'Salary_Tier' in df else pd.Series(1.0, index=df.index)
    raw *= tier_prior
    total = raw.sum()
    return (raw / total * 100.0) if total > 0 else pd.Series(100.0/len(raw), index=raw.index)

def compute_gto_ownership(df, w: Weights):
    Q = (df['S_Perf'] ** w.gamma_gto) * (df['S_Market'] ** w.delta_gto)
    Q += 0.05 * df['B5_Price'] + 0.02 * _safe_minmax(df['Proj_SD'])
    Q = Q.clip(lower=1e-6)
    total = Q.sum()
    slots_total = 600.0
    gto = (Q / total * slots_total) if total > 0 else pd.Series(slots_total/len(Q), index=Q.index)
    return gto  # already in "percent of slots" units

def compute_diagnostics(df, po, gto):
    df = df.copy()
    df['PO_%']  = po
    df['GTO_%'] = gto
    df['HypeGap_Z'] = _zscore(df['S_Market']) - _zscore(df['S_Perf'])
    df['Leverage_%'] = df['GTO_%'] - df['PO_%']
    df['Tier_Floor_%'] = 0.5
    df['Cap_Flag'] = False
    df['TargetSlots']   = np.rint(df['GTO_%'] * 1.5).astype(int)
    df['MaxExposure_%'] = 26.5
    df['MinInclude']    = df['GTO_%'] > 0
    return df

def build_scorecard(raw_df: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    df = compute_clusters(raw_df, weights)
    df = compute_projections(df)
    po = compute_projected_ownership(df, weights)
    gto = compute_gto_ownership(df, weights)
    df = compute_diagnostics(df, po, gto)

    # Preferred order (only keep columns that exist)
    preferred = [
        'Player','PlayerID','Salary','Event','Slate','FieldSize','Timestamp',
        'Vegas_WinProb','Vegas_Top20Prob','Vegas_LineMove','Handle_SharpRatio',
        'Social_Sentiment','Social_Volume','Social_RecencyScore',
        'Tout_ProjPts','Tout_Heat','Tout_ProjOwn',
        'SG_Long','SG_Recent','Course_Fit','Volatility_Index','CompField_Adjust',
        'Value_PPD_Baseline','Salary_Tier',
        'Hist_Own_TierAvg','Hist_Event_Result','Hist_CutRate',
        'B1_Betting','B2_Social','B3_Tout','B4_Adv','B5_Price','B6_History',
        'S_Market','S_Perf',
        'Proj_Mean','Proj_SD','Proj_P10','Proj_P90','Proj_1SD_Low','Proj_1SD_High','Proj_2SD_Low','Proj_2SD_High',
        'PO_%','GTO_%','HypeGap_Z','Leverage_%','Tier_Floor_%','Cap_Flag',
        'TargetSlots','MaxExposure_%','MinInclude'
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]

# ---------------- UI ----------------
st.title("DFS Matrix — GTO Scorecard (Beta)")

with st.sidebar:
    st.header("Weights & Settings")

    w = Weights(
        w_betting = st.slider("Market: Betting weight", 0.0, 1.0, DEFAULT_W.w_betting, 0.01),
        w_social  = st.slider("Market: Social weight",  0.0, 1.0, DEFAULT_W.w_social, 0.01),
        w_tout    = st.slider("Market: Tout weight",    0.0, 1.0, DEFAULT_W.w_tout, 0.01),
        alpha_po  = st.slider("PO α (Market influence)", 0.0, 1.5, DEFAULT_W.alpha_po, 0.01),
        beta_po   = st.slider("PO β (Perf influence)",   0.0, 1.0, DEFAULT_W.beta_po, 0.01),

        w_adv     = st.slider("Perf: Advanced weight",  0.0, 1.0, DEFAULT_W.w_adv, 0.01),
        w_price   = st.slider("Perf: Price weight",     0.0, 1.0, DEFAULT_W.w_price, 0.01),
        w_hist    = st.slider("Perf: History weight",   0.0, 1.0, DEFAULT_W.w_hist, 0.01),
        gamma_gto = st.slider("GTO γ (Perf influence)",  0.0, 1.5, DEFAULT_W.gamma_gto, 0.01),
        delta_gto = st.slider("GTO δ (Market influence)",0.0, 1.0, DEFAULT_W.delta_gto, 0.01),
    )

    st.caption("Tip: Market cluster mostly drives Projected Ownership. Performance cluster mostly drives GTO Ownership and Projections.")

uploaded = st.file_uploader("Upload raw CSV (with the bubble spec columns)", type=["csv"])

if uploaded is None:
    st.info("Upload your weekly raw CSV to compute PO%, GTO%, and projections.\n\nNeed a header? Ask me for the spec or paste from our last message.")
else:
    raw = pd.read_csv(uploaded)
    st.subheader("Raw Input (preview)")
    st.dataframe(raw.head(20), use_container_width=True)

    try:
        scorecard = build_scorecard(raw, w)
        # Quick validity checks
        po_sum = scorecard['PO_%'].sum() if 'PO_%' in scorecard else np.nan
        gto_sum = scorecard['GTO_%'].sum() if 'GTO_%' in scorecard else np.nan

        st.subheader("Scorecard Outputs")
        st.write(f"**PO_% sum:** {po_sum:.2f} (should be ~100.0)")
        st.write(f"**GTO_% sum:** {gto_sum:.2f} (should be ~600.0 for 150×6)")

        # Main views
        cols = st.columns([3,2])
        with cols[0]:
            st.markdown("**Top by GTO_%**")
            st.dataframe(scorecard.sort_values('GTO_%', ascending=False).head(25), use_container_width=True)
        with cols[1]:
            st.markdown("**Top Leverage (GTO_% − PO_%)**")
            st.dataframe(scorecard.sort_values('Leverage_%', ascending=False).head(25)[
                ['Player','Salary','PO_%','GTO_%','Leverage_%','HypeGap_Z']
            ], use_container_width=True)

        # Download
        out = io.StringIO()
        scorecard.to_csv(out, index=False)
        st.download_button("Download GTO Scorecard CSV", out.getvalue(), file_name="gto_scorecard.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Processing error: {e}")
        st.stop()

