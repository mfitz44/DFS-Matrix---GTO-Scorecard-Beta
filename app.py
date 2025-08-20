# app.py — DFS Matrix: GTO Scorecard (Beta)
from __future__ import annotations
import io, re, unicodedata
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DFS Matrix — GTO Scorecard", layout="wide")

# =========================
# Config / weights
# =========================
@dataclass
class Weights:
    # Market cluster (bubbles 1–3)
    w_betting: float = 0.50
    w_social:  float = 0.20
    w_tout:    float = 0.30
    alpha_po:  float = 0.75  # PO: S_Market^alpha * S_Perf^beta
    beta_po:   float = 0.25

    # Performance cluster (bubbles 4–6)
    w_adv:     float = 0.55
    w_price:   float = 0.25
    w_hist:    float = 0.20
    gamma_gto: float = 0.85  # GTO: S_Perf^gamma * S_Market^delta
    delta_gto: float = 0.15

DEFAULT_W = Weights()

# =========================
# Utility helpers
# =========================
def normalize_name(s: str) -> str:
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.strip()
    if "," in s:  # flip "Last, First" → "First Last"
        last, first = [t.strip() for t in s.split(",", 1)]
        s = f"{first} {last}"
    s = re.sub(r"\s+", " ", s)
    return s

def _safe_minmax(x: pd.Series) -> pd.Series:
    if x is None or len(x) == 0 or x.isnull().all():
        return pd.Series(0.5, index=x.index if isinstance(x, pd.Series) else None)
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(hi, lo):
        return pd.Series(0.5, index=x.index)
    return (x.fillna(lo) - lo) / (hi - lo)

def _zscore(x: pd.Series) -> pd.Series:
    mu, sd = x.mean(skipna=True), x.std(ddof=0, skipna=True)
    if sd == 0 or pd.isna(sd): return pd.Series(0.0, index=x.index)
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

def add_salary_tier(df: pd.DataFrame) -> pd.DataFrame:
    def tier(sal):
        sal = float(sal)
        if sal >= 10200: return "Stud"
        if sal >= 9100:  return "Upper"
        if sal >= 7800:  return "Mid"
        return "Value"
    df["Salary_Tier"] = df["Salary"].astype(float).apply(tier)
    return df

# =========================
# ETL loaders (your exact files)
# =========================
def load_dk_pool(file) -> pd.DataFrame:
    dk = pd.read_csv(file)
    # Expect: Name, ID, Salary, Game Info
    dk = dk.rename(columns={"Name":"Player","ID":"PlayerID","Salary":"Salary","Game Info":"Event"})
    dk["Player"] = dk["Player"].map(normalize_name)
    dk["Event"]  = dk.get("Event","TOUR Championship 2025")
    dk["Slate"]  = "DK Classic"
    dk["FieldSize"] = len(dk)
    dk["Timestamp"] = pd.Timestamp.utcnow().isoformat()
    dk = dk[["Player","PlayerID","Salary","Event","Slate","FieldSize","Timestamp"]]
    dk["PlayerID"] = dk["PlayerID"].astype(str)
    dk["Salary"]   = dk["Salary"].astype(int)
    dk = add_salary_tier(dk)
    return dk

def load_dg_pre_odds(file) -> pd.DataFrame:
    dg = pd.read_csv(file)
    # Expect: player_name, win, top_5, top_10, top_20
    dg["Player"] = dg["player_name"].map(normalize_name)
    keep = ["Player","win","top_5","top_10","top_20"]
    for c in keep:
        if c not in dg: dg[c] = np.nan
    return dg[keep]

def load_rg_proj(file) -> pd.DataFrame:
    rg = pd.read_csv(file)
    # Expect: name, fpts, proj_own, ceil, floor, salary
    rg["Player"] = rg["name"].map(normalize_name)
    rg = rg.rename(columns={
        "fpts":"RG_fpts","proj_own":"RG_proj_own","ceil":"RG_ceil","floor":"RG_floor","salary":"RG_salary"
    })
    keep = ["Player","RG_fpts","RG_proj_own","RG_ceil","RG_floor","RG_salary"]
    for c in keep:
        if c not in rg: rg[c] = np.nan
    return rg[keep]

# ---- Vertical Excel parsers (engine='openpyxl') ----

def load_dg_course_fit_vertical(file) -> pd.DataFrame:
    """
    Course-fit / adjusted SG Excel is one column stacked.
    We parse into Player + (SG_Baseline, CH_Adj, CF_Adj). Extra labels ignored.
    """
    df = pd.read_excel(file, sheet_name=0, header=None, engine="openpyxl")
    s = df.iloc[:, 0].astype(str)

    rows = []
    current = None
    rec = {}
    for raw in s:
        val = raw.strip()
        if not val:
            continue
        # Heuristic: a player header (contains comma or looks like a name)
        if ("," in val) or (re.search(r"[A-Za-z]+\s+[A-Za-z]+", val) and not re.search(r"[_/\\]", val)):
            if current and rec:
                rows.append(rec)
            current = normalize_name(val)
            rec = {"Player": current, "SG_Baseline": np.nan, "CH_Adj": np.nan, "CF_Adj": np.nan}
            continue
        key = val.lower().replace(" ", "").replace("\u00a0","")
        if key.startswith("baseline") or key.startswith("ch_adj") or key.startswith("cf_adj"):
            # labels — wait for the next numeric
            continue
        # numeric lines fill first empty of baseline → ch_adj → cf_adj
        if re.match(r"^-?\d+(\.\d+)?$", val):
            for fld in ["SG_Baseline","CH_Adj","CF_Adj"]:
                if pd.isna(rec.get(fld, np.nan)):
                    rec[fld] = float(val)
                    break
        else:
            # ignore other labels (tee_time, decay, etc.)
            pass
    if current and rec:
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "Player" in out:
        out["Player"] = out["Player"].map(normalize_name)
    return out

def load_dg_sg_history_vertical(file) -> pd.DataFrame:
    """
    DG SG history vertical: 1 column with category labels and numbers.
    We extract simple aggregates SG_Long / SG_Recent (same for now).
    """
    df = pd.read_excel(file, sheet_name=0, header=None, engine="openpyxl")
    s = df.iloc[:, 0].astype(str)

    rows = []
    current = None
    rec = {}
    last_key = None
    for raw in s:
        val = raw.strip()
        if not val:
            continue
        if ("," in val) or (re.search(r"[A-Za-z]+\s+[A-Za-z]+", val) and not re.search(r"[_/\\]", val)):
            if current and rec:
                rows.append(rec)
            current = normalize_name(val)
            rec = {"Player": current, "SG_Putt": np.nan, "SG_ARG": np.nan, "SG_APP": np.nan, "SG_OTT": np.nan}
            last_key = None
            continue
        low = val.lower()
        if low.startswith("putt"): last_key = "SG_Putt";  continue
        if low.startswith("arg"):  last_key = "SG_ARG";   continue
        if low.startswith("app"):  last_key = "SG_APP";   continue
        if low.startswith("ott"):  last_key = "SG_OTT";   continue

        if re.match(r"^-?\d+(\.\d+)?$", val) and last_key:
            if pd.isna(rec.get(last_key, np.nan)):
                rec[last_key] = float(val)

    if current and rec:
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "Player" in out:
        out["Player"] = out["Player"].map(normalize_name)
    for fld in ["SG_Putt","SG_ARG","SG_APP","SG_OTT"]:
        if fld not in out: out[fld] = np.nan
    out["SG_Long"]   = out[["SG_Putt","SG_ARG","SG_APP","SG_OTT"]].mean(axis=1, skipna=True)
    out["SG_Recent"] = out["SG_Long"]  # refine later if you split windows
    return out[["Player","SG_Long","SG_Recent"]].drop_duplicates(subset=["Player"])

# =========================
# Bubble computations
# =========================
def bubble_betting(df: pd.DataFrame) -> pd.Series:
    win   = _safe_minmax(df.get("win", 0))
    top20 = _safe_minmax(df.get("top_20", 0))
    top10 = _safe_minmax(df.get("top_10", 0))
    top5  = _safe_minmax(df.get("top_5", 0))
    raw = 0.45*win + 0.25*top20 + 0.20*top10 + 0.10*top5
    return _safe_minmax(raw)

def bubble_social(df: pd.DataFrame) -> pd.Series:
    sent = _safe_minmax((df.get("Social_Sentiment", 0) + 1) / 2.0)
    vol  = _safe_minmax(np.log1p(df.get("Social_Volume", 0)))
    rec  = _safe_minmax(df.get("Social_RecencyScore", 0))
    raw = 0.5*sent + 0.3*vol + 0.2*rec
    return _safe_minmax(raw)

def bubble_tout(df: pd.DataFrame) -> pd.Series:
    # Use RG mean points and RG ownership as "industry" signal
    proj = _safe_minmax(df.get("RG_fpts", 0))
    pown = _safe_minmax(df.get("RG_proj_own", 0))
    heat = _safe_minmax(df.get("Tout_Heat", 0))
    raw = 0.5*proj + 0.3*heat + 0.2*pown
    return _safe_minmax(raw)

def bubble_advanced(df: pd.DataFrame) -> pd.Series:
    comp = _safe_minmax((df.get("CompField_Adjust", 0) + 1) / 2.0)
    fit  = _safe_minmax((df.get("Course_Fit", 0) + 1) / 2.0)
    sgl  = _safe_minmax(df.get("SG_Baseline", df.get("SG_Long", 0)))
    sgr  = _safe_minmax(df.get("SG_Recent", 0))
    vol  = 1 - _safe_minmax(df.get("Volatility_Index", 0))  # prefer stability by default
    raw = 0.35*sgl + 0.30*sgr + 0.20*fit + 0.10*vol + 0.05*comp
    return _safe_minmax(raw)

def bubble_price(df: pd.DataFrame) -> pd.Series:
    val = _safe_minmax(df.get("Value_PPD_Baseline", 0))
    tier_map = {"Stud":0.50,"Upper":0.60,"Mid":0.70,"Value":0.80}
    if "Salary_Tier" in df:
        tier = df["Salary_Tier"].map(tier_map).fillna(0.7)
    else:
        tier = pd.Series(0.7, index=df.index)
    raw = 0.8*val + 0.2*tier
    return _safe_minmax(raw)

def bubble_history(df: pd.DataFrame) -> pd.Series:
    own = _safe_minmax(df.get("Hist_Own_TierAvg", 0))
    res = _safe_minmax((df.get("Hist_Event_Result", 0) + 1) / 2.0)
    cut = _safe_minmax(df.get("Hist_CutRate", 0))
    raw = 0.4*own + 0.4*res + 0.2*cut
    return _safe_minmax(raw)

def compute_clusters(df: pd.DataFrame, w: Weights) -> pd.DataFrame:
    df = df.copy()
    df["B1_Betting"]  = bubble_betting(df)
    df["B2_Social"]   = bubble_social(df)
    df["B3_Tout"]     = bubble_tout(df)
    df["B4_Adv"]      = bubble_advanced(df)
    df["B5_Price"]    = bubble_price(df)
    df["B6_History"]  = bubble_history(df)

    df["S_Market"]    = _geom_blend([df["B1_Betting"], df["B2_Social"], df["B3_Tout"]],
                                    [w.w_betting,      w.w_social,      w.w_tout])
    df["S_Perf"]      = _geom_blend([df["B4_Adv"], df["B5_Price"], df["B6_History"]],
                                    [w.w_adv,          w.w_price,        w.w_hist])
    return df

# =========================
# Projections / Ownerships
# =========================
def compute_projections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Placeholder model — replace with your regression later
    df["Proj_Mean"] = 20 * df["S_Perf"] + 2.5 * df["B5_Price"]
    df["Proj_SD"]   = ((6.0 * (1 - df["S_Perf"])) + 4.0 * (1 - (1 - df.get("Volatility_Index", 0)).clip(0,1))).clip(lower=3.0)

    mu, sd = df["Proj_Mean"], df["Proj_SD"]
    df["Proj_1SD_Low"]  = mu - sd
    df["Proj_1SD_High"] = mu + sd
    df["Proj_2SD_Low"]  = mu - 2*sd
    df["Proj_2SD_High"] = mu + 2*sd
    df["Proj_P10"]      = mu - 1.2816*sd
    df["Proj_P90"]      = mu + 1.2816*sd
    return df

def compute_projected_ownership(df: pd.DataFrame, w: Weights) -> pd.Series:
    raw = (df["S_Market"] ** w.alpha_po) * (df["S_Perf"] ** w.beta_po)
    if "Salary_Tier" in df:
        tier_prior = df["Salary_Tier"].map({"Stud":1.08,"Upper":1.05,"Mid":1.00,"Value":0.95}).fillna(1.0)
        raw *= tier_prior
    total = raw.sum()
    return (raw / total * 100.0) if total > 0 else pd.Series(100.0/len(raw), index=raw.index)

def compute_gto_ownership(df: pd.DataFrame, w: Weights) -> pd.Series:
    Q = (df["S_Perf"] ** w.gamma_gto) * (df["S_Market"] ** w.delta_gto)
    Q += 0.05 * df["B5_Price"] + 0.02 * _safe_minmax(df["Proj_SD"])  # mild ceiling nudge
    Q = Q.clip(lower=1e-6)
    total = Q.sum()
    slots_total = 600.0  # 150×6
    return (Q / total * slots_total) if total > 0 else pd.Series(slots_total/len(Q), index=Q.index)

def compute_diagnostics(df: pd.DataFrame, po: pd.Series, gto: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["PO_%"]  = po
    df["GTO_%"] = gto
    df["HypeGap_Z"] = _zscore(df["S_Market"]) - _zscore(df["S_Perf"])
    df["Leverage_%"] = df["GTO_%"] - df["PO_%"]
    df["Tier_Floor_%"] = 0.5
    df["Cap_Flag"] = False
    df["TargetSlots"]   = np.rint(df["GTO_%"] * 1.5).astype(int)
    df["MaxExposure_%"] = 26.5
    df["MinInclude"]    = df["GTO_%"] > 0
    return df

def build_scorecard(raw_df: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    df = compute_clusters(raw_df, weights)
    df = compute_projections(df)
    po  = compute_projected_ownership(df, weights)
    gto = compute_gto_ownership(df, weights)
    df  = compute_diagnostics(df, po, gto)

    preferred = [
        "Player","PlayerID","Salary","Event","Slate","FieldSize","Timestamp",
        "win","top_5","top_10","top_20",
        "RG_fpts","RG_proj_own","RG_ceil","RG_floor","RG_salary",
        "SG_Baseline","CH_Adj","CF_Adj","SG_Long","SG_Recent","Course_Fit","Volatility_Index","CompField_Adjust",
        "Value_PPD_Baseline","Salary_Tier",
        "Hist_Own_TierAvg","Hist_Event_Result","Hist_CutRate",
        "B1_Betting","B2_Social","B3_Tout","B4_Adv","B5_Price","B6_History","S_Market","S_Perf",
        "Proj_Mean","Proj_SD","Proj_P10","Proj_P90","Proj_1SD_Low","Proj_1SD_High","Proj_2SD_Low","Proj_2SD_High",
        "PO_%","GTO_%","HypeGap_Z","Leverage_%","Tier_Floor_%","Cap_Flag",
        "TargetSlots","MaxExposure_%","MinInclude"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]

def preflight_feasibility(df: pd.DataFrame, L=150, R=6, cap_pct=26.5, gto_floor=0.0) -> dict:
    eligible = df[df["GTO_%"] > gto_floor].copy()
    E = len(eligible)
    S = R * L
    cap_slots = int(np.floor(L * cap_pct / 100.0))
    feasible = (E * cap_slots) >= S
    min_cap_pct = 100.0 * (np.ceil(S / max(E,1)) / L) if E > 0 else 100.0
    return {"E":E, "S":S, "cap_pct":cap_pct, "cap_slots":cap_slots, "feasible":feasible, "min_cap_pct":float(min_cap_pct)}

# =========================
# UI: Weights & file uploads
# =========================
st.title("DFS Matrix — GTO Scorecard (Beta)")

with st.sidebar:
    st.header("Weights & Settings")
    w = Weights(
        w_betting = st.slider("Market: Betting weight", 0.0, 1.0, DEFAULT_W.w_betting, 0.01),
        w_social  = st.slider("Market: Social weight",  0.0, 1.0, DEFAULT_W.w_social, 0.01),
        w_tout    = st.slider("Market: Tout weight",    0.0, 1.0, DEFAULT_W.w_tout, 0.01),
        alpha_po  = st.slider("PO α (Market influence)", 0.0, 1.5, DEFAULT_W.alpha_po, 0.01),
        beta_po   = st.slider("PO β (Perf influence)",   0.0, 1.0,  DEFAULT_W.beta_po,  0.01),

        w_adv     = st.slider("Perf: Advanced weight",  0.0, 1.0, DEFAULT_W.w_adv, 0.01),
        w_price   = st.slider("Perf: Price weight",     0.0, 1.0, DEFAULT_W.w_price, 0.01),
        w_hist    = st.slider("Perf: History weight",   0.0, 1.0, DEFAULT_W.w_hist, 0.01),
        gamma_gto = st.slider("GTO γ (Perf influence)",  0.0, 1.5, DEFAULT_W.gamma_gto, 0.01),
        delta_gto = st.slider("GTO δ (Market influence)",0.0, 1.0, DEFAULT_W.delta_gto, 0.01),
    )

st.subheader("Upload weekly data")
c1, c2 = st.columns(2)
with c1:
    dk_pool_file = st.file_uploader("DK Player Pool CSV", type=["csv"], key="dk_pool")
    rg_file      = st.file_uploader("Rotogrinders Projections CSV", type=["csv"], key="rg")
    dg_odds_file = st.file_uploader("DataGolf Pre-Tournament Odds CSV", type=["csv"], key="dg_odds")
with c2:
    dg_fit_file  = st.file_uploader("DataGolf Course Fit / Adjusted SG (Excel)", type=["xlsx","xls"], key="dg_fit")
    dg_2024_file = st.file_uploader("DG SG History 2024 (Excel)", type=["xlsx","xls"], key="dg24")
    dg_2025_file = st.file_uploader("DG SG History 2025 (Excel)", type=["xlsx","xls"], key="dg25")

run_etl = st.button("Run ETL & Build Scorecard")

if run_etl:
    if not dk_pool_file:
        st.error("Please upload the DK Player Pool CSV.")
        st.stop()

    st.info("Loading DK pool…")
    dk = load_dk_pool(dk_pool_file)
    merged = dk.copy()

    if dg_odds_file:
        st.info("Merging DataGolf pre-tournament odds…")
        merged = merged.merge(load_dg_pre_odds(dg_odds_file), on="Player", how="left")

    if rg_file:
        st.info("Merging Rotogrinders projections…")
        merged = merged.merge(load_rg_proj(rg_file), on="Player", how="left")

    if dg_fit_file:
        st.info("Parsing DG course fit / adjusted SG (vertical)…")
        merged = merged.merge(load_dg_course_fit_vertical(dg_fit_file), on="Player", how="left")

    sg_hist = None
    if dg_2024_file:
        st.info("Parsing DG SG history 2024 (vertical)…")
        sg_hist = load_dg_sg_history_vertical(dg_2024_file)
    if dg_2025_file:
        st.info("Parsing DG SG history 2025 (vertical)…")
        sg25 = load_dg_sg_history_vertical(dg_2025_file)
        sg_hist = sg25 if sg_hist is None else pd.concat([sg_hist, sg25]).groupby("Player", as_index=False).mean(numeric_only=True)
    if sg_hist is not None:
        merged = merged.merge(sg_hist, on="Player", how="left")

    # Defaults for optional fields
    for col, default in [
        ("Course_Fit", 0.0), ("Volatility_Index", 0.9), ("CompField_Adjust", 0.0),
        ("Value_PPD_Baseline", np.nan), ("Hist_Own_TierAvg", np.nan),
        ("Hist_Event_Result", np.nan), ("Hist_CutRate", 0.95),
    ]:
        if col not in merged: merged[col] = default

    # If no PPD, compute from RG_fpts
    if "Value_PPD_Baseline" in merged and merged["Value_PPD_Baseline"].isna().all() and "RG_fpts" in merged:
        with np.errstate(invalid="ignore", divide="ignore"):
            merged["Value_PPD_Baseline"] = (merged["RG_fpts"] / (merged["Salary"]/1000)).replace([np.inf,-np.inf], np.nan)

    st.success("ETL merge complete. Preview below.")
    st.dataframe(merged.head(20), use_container_width=True)

    # Build scorecard
    scorecard = build_scorecard(merged, w)

    # Pre-flight feasibility
    try:
        info = preflight_feasibility(scorecard, L=150, R=6, cap_pct=26.5, gto_floor=0.0)
        st.subheader("Pre-flight (lineup feasibility)")
        st.write(f"Eligible golfers (E): **{info['E']}**")
        st.write(f"Slots needed (S = 6×150): **{info['S']}**")
        st.write(f"Cap **{info['cap_pct']}%** → cap slots **{info['cap_slots']}**")
        if info["feasible"]:
            st.success("Feasible under current cap/exposure.")
        else:
            st.error("Not feasible — raise cap% or increase eligible golfers.")
        st.caption(f"Suggested minimum cap%: **≥ {info['min_cap_pct']:.1f}%**")
    except Exception:
        pass

    st.subheader("Scorecard Outputs")
    st.write(f"PO_% sum: **{scorecard['PO_%'].sum():.2f}** (≈100)")
    st.write(f"GTO_% sum: **{scorecard['GTO_%'].sum():.2f}** (≈600)")

    cA, cB = st.columns([3,2])
    with cA:
        st.markdown("**Top by GTO_%**")
        st.dataframe(scorecard.sort_values("GTO_%", ascending=False).head(25), use_container_width=True)
    with cB:
        st.markdown("**Top Leverage (GTO_% − PO_%)**")
        st.dataframe(scorecard.sort_values("Leverage_%", ascending=False).head(25)[
            ["Player","Salary","PO_%","GTO_%","Leverage_%","HypeGap_Z"]
        ], use_container_width=True)

    # Download scorecard
    out = io.StringIO()
    scorecard.to_csv(out, index=False)
    st.download_button("Download GTO Scorecard CSV", out.getvalue(),
                       file_name="gto_scorecard.csv", mime="text/csv")

else:
    st.info("Upload your files on the left, then click **Run ETL & Build Scorecard** to generate PO%, GTO%, and projections.")




