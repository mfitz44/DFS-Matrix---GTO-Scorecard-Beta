import io, re, unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DFS Matrix — Simplified GTO Scorecard", layout="wide")

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

# ========== Scorecard Logic ==========
def build_scorecard(dk, dg, rg):
    merged = dk.merge(dg, on="Player", how="left").merge(rg, on="Player", how="left")

    # Normalize inputs
    win   = _safe_minmax(merged["win"])
    top20 = _safe_minmax(merged["top_20"])
    fpts  = _safe_minmax(merged["RG_fpts"])
    ceil  = _safe_minmax(merged["RG_ceil"])
    floor = _safe_minmax(merged["RG_floor"])
    sal   = _safe_minmax(-merged["Salary"])  # lower salary = better
    psi   = _safe_minmax(merged["RG_proj_own"])

    # Weighted RealScore
    merged["RealScore"] = (
        0.30*win +
        0.20*top20 +
        0.15*fpts +
        0.10*ceil +
        0.05*floor +
        0.10*sal +
        0.10*psi
    )

    # GTO Ownership (sum = 600%)
    edge_pos = (merged["RealScore"] - merged["RealScore"].min()) + 1e-6
    merged["GTO_%"] = edge_pos / edge_pos.sum() * 600.0

    # Projected Ownership (RG proj_own → sum = 600%)
    if "RG_proj_own" in merged and not merged["RG_proj_own"].isna().all():
        po_raw = merged["RG_proj_own"].clip(lower=0.0)
        merged["PO_%"] = po_raw / po_raw.sum() * 600.0
    else:
        merged["PO_%"] = np.nan

    # Leverage (GTO – PO)
    merged["Leverage_%"] = merged["GTO_%"] - merged["PO_%"]

    return merged

# ========== Streamlit UI ==========
st.title("DFS Matrix — Simplified GTO Scorecard")

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

    st.dataframe(scorecard[[
        "Player","Salary",
        "RG_fpts","RG_ceil","RG_floor","RG_proj_own",
        "win","top_5","top_10","top_20",
        "RealScore","PO_%","GTO_%","Leverage_%"
    ]], use_container_width=True)

    out = io.StringIO()
    scorecard.to_csv(out, index=False)
    st.download_button("Download GTO Scorecard CSV", out.getvalue(),
                       file_name="gto_scorecard.csv", mime="text/csv")

else:
    st.info("Upload all three files to generate the GTO Scorecard.")



