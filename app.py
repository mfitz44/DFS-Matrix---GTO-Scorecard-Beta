import io, re, unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DFS Matrix — Scalar Resonance GTO", layout="wide")

# -------------------------
# Helpers
# -------------------------
def normalize_name(s: str) -> str:
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s.strip())
    if "," in s:
        last, first = [t.strip() for t in s.split(",", 1)]
        s = f"{first} {last}"
    return s

def require_cols(df: pd.DataFrame, need: list, src: str):
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.error(f"{src}: missing columns {missing}. Check your file headers.")
        st.stop()

def safe_minmax(x: pd.Series) -> pd.Series:
    if x is None or x.isnull().all(): return pd.Series(0.5, index=x.index)
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(hi, lo): return pd.Series(0.5, index=x.index)
    return (x - lo) / (hi - lo)

# -------------------------
# Loaders (robust renames)
# -------------------------
def load_dk_pool(file) -> pd.DataFrame:
    dk = pd.read_csv(file)
    # DK typically: Name, ID, Salary
    # Accept common variants too:
    cand_name = next((c for c in dk.columns if c.lower() in ["name","player","player_name"]), None)
    cand_id   = next((c for c in dk.columns if c.lower() in ["id","playerid","player_id"]), None)
    cand_sal  = next((c for c in dk.columns if c.lower() in ["salary"]), None)
    if not cand_name or not cand_id or not cand_sal:
        require_cols(dk, ["Name","ID","Salary"], "DK Pool")
    name_col = cand_name or "Name"
    id_col   = cand_id   or "ID"
    sal_col  = cand_sal  or "Salary"

    dk["Player"]   = dk[name_col].map(normalize_name)
    dk["PlayerID"] = dk[id_col].astype(str)
    dk["Salary"]   = dk[sal_col].astype(int)
    return dk[["Player","PlayerID","Salary"]]

def load_dg_odds(file) -> pd.DataFrame:
    dg = pd.read_csv(file)
    # Expect DG: player_name, win, top_5, top_10, top_20 (0–1)
    require_cols(dg, ["player_name","win","top_5","top_10","top_20"], "DG Odds")
    dg["Player"] = dg["player_name"].map(normalize_name)
    keep = ["Player","win","top_5","top_10","top_20"]
    return dg[keep]

def load_rg_proj(file) -> pd.DataFrame:
    rg = pd.read_csv(file)
    # Expect RG: name, fpts, proj_own, ceil, floor, salary
    require_cols(rg, ["name","fpts","proj_own","ceil","floor","salary"], "RG Projections")
    rg["Player"] = rg["name"].map(normalize_name)
    rg = rg.rename(columns={
        "fpts":"RG_fpts","proj_own":"RG_proj_own","ceil":"RG_ceil","floor":"RG_floor","salary":"RG_salary"
    })
    rg["RG_salary"] = rg["RG_salary"].astype(float)
    keep = ["Player","RG_fpts","RG_ceil","RG_floor","RG_proj_own","RG_salary"]
    return rg[keep]

# -------------------------
# Scalar Resonance GTO
# -------------------------
def build_scorecard(dk, dg, rg, lambda_psi=0.30):
    # Merge on normalized Player
    m = dk.merge(dg, on="Player", how="left").merge(rg, on="Player", how="left")

    # Sanity for odds/proj: fill minimal neutral values if missing
    for c, v in {"win":0.0,"top_5":0.0,"top_10":0.0,"top_20":0.0,
                 "RG_fpts":0.0,"RG_ceil":0.0,"RG_floor":0.0,"RG_proj_own":0.0}.items():
        if c not in m: m[c] = v
        m[c] = m[c].fillna(v)

    # Bubbles (normalize)
    win   = safe_minmax(m["win"])
    top20 = safe_minmax(m["top_20"])
    fpts  = safe_minmax(m["RG_fpts"])
    ceil  = safe_minmax(m["RG_ceil"])
    floor = safe_minmax(m["RG_floor"])
    psi   = safe_minmax(m["RG_proj_own"])

    # Salary boundary: efficiency + tier
    eff = safe_minmax( (m["RG_fpts"].replace(0, np.nan) / m["Salary"].replace(0, np.nan)).fillna(0.0) )
    m["Tier"] = pd.qcut(m["Salary"], 4, labels=["Value","Mid","Upper","Stud"])
    tier_map  = {"Stud":0.8,"Upper":0.7,"Mid":0.6,"Value":0.5}
    tier = m["Tier"].map(tier_map).astype(float).fillna(0.6)

    # Dynamic weights = variance (numeric only)
    bubbles = {"Win":win, "Top20":top20, "Fpts":fpts, "Ceil":ceil, "Floor":floor, "Eff":eff, "Tier":tier}
    variances = {k: v.var() for k,v in bubbles.items()}  # all numeric by construction here
    total_var = sum(variances.values()) or 1.0
    weights = {k: variances[k]/total_var for k in variances}

    # RealScore (variance-weighted) with PSI as coherence amplifier
    real = sum(bubbles[k]*weights[k] for k in bubbles)
    real = real * (1 + lambda_psi * psi)

    m["RealScore"] = real

    # Salary-weighted GTO normalization → 600
    q = (real / m["Salary"].replace(0, np.nan)).fillna(0.0).clip(lower=1e-6)
    m["GTO_%"] = q / q.sum() * 600.0

    # Projected Ownership → 600 (for apples-to-apples)
    po_raw = m["RG_proj_own"].clip(lower=0.0)
    m["PO_%"] = po_raw / (po_raw.sum() if po_raw.sum() > 0 else 1.0) * 600.0

    # Leverage + adjusted GTO (gradient drift)
    m["Leverage_%"] = m["GTO_%"] - m["PO_%"]
    m["Adj_GTO_%"]  = 0.7*m["GTO_%"] + 0.3*(m["GTO_%"] + m["Leverage_%"])

    # Final tidy
    out = m[[
        "Player","PlayerID","Salary",
        "win","top_5","top_10","top_20",
        "RG_fpts","RG_ceil","RG_floor","RG_proj_own",
        "RealScore","PO_%","GTO_%","Leverage_%","Adj_GTO_%"
    ]].sort_values("Adj_GTO_%", ascending=False)
    return out

# -------------------------
# UI
# -------------------------
st.title("DFS Matrix — Scalar Resonance GTO Scorecard")

c1, c2, c3 = st.columns(3)
with c1:
    dk_file = st.file_uploader("DraftKings Player Pool (CSV)", type=["csv"])
with c2:
    dg_file = st.file_uploader("DataGolf Odds (CSV)", type=["csv"])
with c3:
    rg_file = st.file_uploader("RotoGrinders Projections (CSV)", type=["csv"])

if dk_file and dg_file and rg_file:
    dk = load_dk_pool(dk_file)
    dg = load_dg_odds(dg_file)
    rg = load_rg_proj(rg_file)

    scorecard = build_scorecard(dk, dg, rg)

    st.success("✅ GTO Scorecard built successfully!")
    st.write(f"**Total GTO_% = {scorecard['GTO_%'].sum():.2f} (should be 600)**")
    st.write(f"**Total PO_% = {scorecard['PO_%'].sum():.2f} (should be 600)**")
    st.write(f"**Total Adj_GTO_% = {scorecard['Adj_GTO_%'].sum():.2f} (≈ 600)**")

    st.dataframe(scorecard, use_container_width=True)

    # Download CSV (and a builder-friendly copy mapping Adj_GTO_% → GTO_Ownership%)
    out = io.StringIO()
    scorecard.to_csv(out, index=False)
    st.download_button("📥 Download Resonant GTO Scorecard", out.getvalue(),
                       file_name="gto_scorecard_resonant.csv", mime="text/csv")

    builder = scorecard.rename(columns={"Adj_GTO_%":"GTO_Ownership%"})
    out2 = io.StringIO()
    builder.to_csv(out2, index=False)
    st.download_button("📥 Download Builder Scorecard (Adj→GTO_Ownership%)", out2.getvalue(),
                       file_name="gto_scorecard_for_builder.csv", mime="text/csv")
else:
    st.info("Upload all three files (DK pool, DG odds, RG projections) to build the scorecard.")








