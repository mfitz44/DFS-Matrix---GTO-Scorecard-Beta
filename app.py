import io, re, unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DFS Matrix — Scalar Resonance GTO", layout="wide")

# ========== Helpers ==========
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
    if x.isnull().all():
        return pd.Series(0.5, index=x.index)
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if np.isclose(lo, hi):
        return pd.Series(0.5, index=x.index)
    return (x - lo) / (hi - lo)

# ========== Loaders ==========
def load_dk_pool(file):
    dk = pd.read_csv(file)
    dk = dk.rename(columns={"Name":"Player","ID":"PlayerID","Salary":"Salary"})
    dk["Player"] = dk["Player"].map(normalize_name)
    dk["Salary"] = dk["Salary"].astype(int)
    return dk[["Player","PlayerID","Salary"]]

def load_dg_odds(file):
    dg = pd.read_csv(file)
    dg["Player"] = dg["player_name"].map(normalize_name)
    keep = ["Player","win","top_5","top_10","top_20"]
    return dg[keep]

def load_rg_proj(file):
    rg = pd.read_csv(file)
    rg["Player"] = rg["name"].map(normalize_name)
    rg = rg.rename(columns={
        "fpts":"RG_fpts","proj_own":"RG_proj_own","ceil":"RG_ceil","floor":"RG_floor","salary":"RG_salary"
    })
    keep = ["Player","RG_fpts","RG_ceil","RG_floor","RG_proj_own","RG_salary"]
    return rg[keep]

# ========== Scalar Resonance GTO Process ==========
def build_scorecard(dk, dg, rg, lambda_psi=0.3):
    merged = dk.merge(dg, on="Player", how="left").merge(rg, on="Player", how="left")

    # --- Normalize bubbles ---
    win   = _safe_minmax(merged["win"])
    top20 = _safe_minmax(merged["top_20"])
    fpts  = _safe_minmax(merged["RG_fpts"])
    ceil  = _safe_minmax(merged["RG_ceil"])
    floor = _safe_minmax(merged["RG_floor"])
    psi   = _safe_minmax(merged["RG_proj_own"])

    # Salary bubbles
    eff   = _safe_minmax(merged["RG_fpts"] / merged["Salary"])
    merged["Tier"] = pd.qcut(merged["Salary"], 4, labels=["Value","Mid","Upper","Stud"])
    tier_map = {"Stud":0.8,"Upper":0.7,"Mid":0.6,"Value":0.5}
    tier  = merged["Tier"].map(tier_map)

    # --- Dynamic weighting (variance-driven) ---
    bubbles = {
        "Win": win, "Top20": top20, "Fpts": fpts,
        "Ceil": ceil, "Floor": floor, "Eff": eff, "Tier": tier
    }
    variances = {k: v.var() for k,v in bubbles.items()}
    total_var = sum(variances.values()) or 1
    weights = {k: variances[k]/total_var for k in variances}

    # --- RealScore (variance-weighted sum) ---
    real = sum(bubbles[k] * weights[k] for k in bubbles)

    # --- PSI as coherence amplifier ---
    real = real * (1 + lambda_psi * psi)

    merged["RealScore"] = real

    # --- Salary-weighted GTO normalization ---
    q = (real / merged["Salary"]).clip(lower=1e-6)
    gto = q / q.sum() * 600
    merged["GTO_%"] = gto

    # --- Projected Ownership scaled to 600 ---
    if "RG_proj_own" in merged and not merged["RG_proj_own"].isna().all():
        po_raw = merged["RG_proj_own"].clip(lower=0.0)
        merged["PO_%"] = po_raw / po_raw.sum() * 600.0
    else:
        merged["PO_%"] = np.nan

    # --- Leverage as gradient driver ---
    merged["Leverage_%"] = merged["GTO_%"] - merged["PO_%"]
    merged["Adj_GTO_%"] = (0.7*merged["GTO_%"] + 0.3*(merged["GTO_%"] + merged["Leverage_%"]))

    return merged

# ========== Streamlit UI ==========
st.title("DFS Matrix — Scalar Resonance GTO Scorecard")

c1, c2, c3 = st.columns(3)
with c1:
    dk_file = st.file_uploader("DK Player Pool CSV", type=["csv"])
with c2:
    dg_file = st.file_uploader("DataGolf Odds CSV", type=["csv"])
with c3:
    rg_file = st.file_uploader("Rotogrinders Projections CSV", type=["csv"])

if dk_file and dg_file and rg_file:
    dk = load_dk_pool(dk_file)
    dg = load_dg_odds(dg_file)
    rg = load_rg_proj(rg_file)

    scorecard = build_scorecard(dk, dg, rg)

    st.success("Scorecard built successfully!")
    st.write(f"**Total GTO_% = {scorecard['GTO_%'].sum():.2f} (should be 600)**")
    st.write(f"**Total PO_% = {scorecard['PO_%'].sum():.2f} (should be 600)**")
    st.write(f"**Total Adj_GTO_% = {scorecard['Adj_GTO_%'].sum():.2f} (≈ 600)**")

    st.dataframe(scorecard[[
        "Player","Salary","PlayerID",
        "RG_fpts","RG_ceil","RG_floor","RG_proj_own",
        "win","top_5","top_10","top_20",
        "RealScore","PO_%","GTO_%","Leverage_%","Adj_GTO_%"
    ]], use_container_width=True)

    out = io.StringIO()
    scorecard.to_csv(out, index=False)
    st.download_button("📥 Download GTO Scorecard CSV", out.getvalue(),
                       file_name="gto_scorecard.csv", mime="text/csv")

else:
    st.info("Upload all three files to generate the GTO Scorecard.")

    st.info("Upload all three files to generate the GTO Scorecard.")




