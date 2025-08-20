import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DFS Matrix — GTO Scorecard (Resonant)", layout="wide")

# ------------------------
# Loaders
# ------------------------
def load_dk_pool(file):
    return pd.read_csv(file)

def load_dg_odds(file):
    return pd.read_csv(file)

def load_rg_projections(file):
    return pd.read_csv(file)

# ------------------------
# Build Scorecard
# ------------------------
def build_scorecard(dk, dg, rg):
    # Merge data
    merged = dk.merge(dg, on="Player", how="left")
    merged = merged.merge(rg, on="Player", how="left")

    # Define bubbles (raw inputs)
    bubbles = {
        "Salary": merged["Salary"],
        "WinOdds": merged["Win%"],
        "Top20Odds": merged["Top20%"],
        "Proj": merged["Projection"],
        "Ceil": merged["Ceiling"],
        "Floor": merged["Floor"],
        "PSI": merged["ProjOwn"]  # public sentiment
    }

    # Variances only for numeric bubbles
    variances = {
        k: v.var()
        for k, v in bubbles.items()
        if pd.api.types.is_numeric_dtype(v)
    }

    # Normalize each bubble
    normed = {}
    for k, v in bubbles.items():
        if pd.api.types.is_numeric_dtype(v):
            var = variances.get(k, 1)
            normed[k] = (v - v.mean()) / np.sqrt(var) if var > 0 else v - v.mean()
        else:
            normed[k] = v

    # Weights (scalar resonance distribution)
    weights = {
        "Salary": 0.30,
        "WinOdds": 0.25,
        "Top20Odds": 0.15,
        "Proj": 0.15,
        "Ceil": 0.05,
        "Floor": 0.05,
        "PSI": 0.05
    }

    # Score
    merged["GTO_Score"] = sum(
        weights.get(k, 0) * normed[k] for k in normed if k in weights
    )

    # Convert to %
    exp_scores = np.exp(merged["GTO_Score"] - merged["GTO_Score"].max())
    merged["GTO%"] = 600 * exp_scores / exp_scores.sum()

    return merged[[
        "Player", "Salary", "Win%", "Top20%", "Projection", "Ceiling", "Floor",
        "ProjOwn", "GTO_Score", "GTO%"
    ]]

# ------------------------
# Streamlit UI
# ------------------------
st.title("DFS Matrix — GTO Scorecard (Resonant Beta)")

st.sidebar.header("Upload Data Files")
dk_file = st.sidebar.file_uploader("DraftKings Player Pool (CSV)", type=["csv"])
dg_file = st.sidebar.file_uploader("DataGolf Odds (CSV)", type=["csv"])
rg_file = st.sidebar.file_uploader("RotoGrinders Projections (CSV)", type=["csv"])

if dk_file and dg_file and rg_file:
    dk = load_dk_pool(dk_file)
    dg = load_dg_odds(dg_file)
    rg = load_rg_projections(rg_file)

    scorecard = build_scorecard(dk, dg, rg)
    st.success("✅ GTO Scorecard built successfully!")

    st.dataframe(scorecard, use_container_width=True)

    # Download
    csv = scorecard.to_csv(index=False).encode("utf-8")
    st.download_button("Download GTO Scorecard", csv, "gto_scorecard.csv", "text/csv")

else:
    st.info("Please upload all three files: DK pool, DG odds, RG projections.")






