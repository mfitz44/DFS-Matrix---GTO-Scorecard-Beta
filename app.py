import io, re, unicodedata, math, random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fairway Theory — GTO v2 (Lineup-Feasible)", layout="wide")

# =========================================================
# Utilities
# =========================================================
def normalize_name(s: str) -> str:
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s.strip())
    if "," in s:
        last, first = [t.strip() for t in s.split(",", 1)]
        s = f"{first} {last}"
    return s

def safe_minmax(x: pd.Series) -> pd.Series:
    if x is None or x.isna().all(): return pd.Series(0.5, index=x.index)
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(hi, lo): return pd.Series(0.5, index=x.index)
    return (x - lo) / (hi - lo)

def ensure_600(s: pd.Series) -> pd.Series:
    """If someone uploads 100-scale ownerships, promote ×6 to 600; otherwise return as-is."""
    if s is None or s.isna().all(): return s
    tot = s.fillna(0).sum()
    if 80 <= tot <= 120:  # looks like 100-scale
        return s * 6.0
    return s

def tier_from_salary(sal: pd.Series) -> pd.Series:
    # 4 quantiles: Value, Mid, Upper, Stud
    try:
        q = pd.qcut(sal, 4, labels=["Value","Mid","Upper","Stud"])
    except Exception:
        # fallback if not enough unique values
        bins = pd.cut(sal, 4, labels=["Value","Mid","Upper","Stud"])
        q = bins
    return q

# =========================================================
# Loaders (robust to header variants)
# =========================================================
def load_dk(file) -> pd.DataFrame:
    dk = pd.read_csv(file)
    name_col  = next((c for c in dk.columns if c.lower() in ["name","player","player_name"]), None)
    id_col    = next((c for c in dk.columns if c.lower() in ["id","playerid","player_id"]), None)
    sal_col   = next((c for c in dk.columns if c.lower() == "salary"), None)
    if not name_col or not id_col or not sal_col:
        st.error("DK Pool must contain Name/Player, ID, Salary.")
        st.stop()
    out = pd.DataFrame({
        "Player": dk[name_col].map(normalize_name),
        "PlayerID": dk[id_col].astype(str),
        "Salary": pd.to_numeric(dk[sal_col], errors="coerce")
    }).dropna(subset=["Player","Salary"])
    return out

def load_dg(file) -> pd.DataFrame:
    dg = pd.read_csv(file)
    need = ["player_name","win","top_5","top_10","top_20"]
    if not all(c in dg.columns for c in need):
        st.error(f"DG Odds must contain {need}.")
        st.stop()
    out = pd.DataFrame({
        "Player": dg["player_name"].map(normalize_name),
        "DG_win": pd.to_numeric(dg["win"], errors="coerce").fillna(0.0),
        "DG_top5": pd.to_numeric(dg["top_5"], errors="coerce").fillna(0.0),
        "DG_top10": pd.to_numeric(dg["top_10"], errors="coerce").fillna(0.0),
        "DG_top20": pd.to_numeric(dg["top_20"], errors="coerce").fillna(0.0),
    })
    return out

def load_rg(file) -> pd.DataFrame:
    rg = pd.read_csv(file)
    need = ["name","fpts","proj_own","ceil","floor","salary"]
    if not all(c in rg.columns for c in need):
        st.error(f"RG Projections must contain {need}.")
        st.stop()
    out = pd.DataFrame({
        "Player": rg["name"].map(normalize_name),
        "RG_fpts": pd.to_numeric(rg["fpts"], errors="coerce").fillna(0.0),
        "RG_ceil": pd.to_numeric(rg["ceil"], errors="coerce").fillna(0.0),
        "RG_floor": pd.to_numeric(rg["floor"], errors="coerce").fillna(0.0),
        "PSI_raw": pd.to_numeric(rg["proj_own"], errors="coerce").fillna(0.0),  # public sentiment
        "RG_salary": pd.to_numeric(rg["salary"], errors="coerce").fillna(np.nan),
    })
    return out

# =========================================================
# Resonance scoring (variance-driven + PSI coherence)
# =========================================================
def resonance_scores(df: pd.DataFrame, lambda_psi: float = 0.30) -> pd.DataFrame:
    # Normalize features (0..1)
    f_win   = safe_minmax(df["DG_win"])
    f_top20 = safe_minmax(df["DG_top20"])
    f_fpts  = safe_minmax(df["RG_fpts"])
    f_ceil  = safe_minmax(df["RG_ceil"])
    f_floor = safe_minmax(df["RG_floor"])
    # Efficiency bubble (favor flexible salary fit)
    eff     = safe_minmax((df["RG_fpts"].replace(0, np.nan) / df["Salary"].replace(0, np.nan)).fillna(0.0))
    # Tier prior (soft structure)
    df["Tier"] = tier_from_salary(df["Salary"])
    tier_map = {"Stud":0.80, "Upper":0.70, "Mid":0.60, "Value":0.50}
    f_tier   = df["Tier"].map(tier_map).astype(float).fillna(0.60)
    # PSI coherence (small influence)
    psi = safe_minmax(df["PSI_raw"])

    # Variance-driven weights across bubbles
    parts = {
        "win": f_win, "top20": f_top20, "fpts": f_fpts,
        "ceil": f_ceil, "floor": f_floor, "eff": eff, "tier": f_tier
    }
    variances = {k: v.var() for k,v in parts.items()}
    tot_var = sum(variances.values()) or 1.0
    weights = {k: variances[k]/tot_var for k in parts}

    base = sum(parts[k]*weights[k] for k in parts)
    # PSI as coherence amplifier (not a driver)
    real = base * (1.0 + lambda_psi * psi)

    df["RealScore"] = real
    df["PSI"] = psi
    return df

# =========================================================
# Build a bank of feasible lineups (salary-valid, unique)
# =========================================================
def build_lineup_bank(players: List[str],
                      salaries: np.ndarray,
                      realscore: np.ndarray,
                      rg_fpts: np.ndarray,
                      salary_min: int,
                      salary_max: int,
                      bank_size: int = 50000,
                      beta_sample: float = 6.0,
                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int,...]]]:
    """
    Returns:
      A  -> incidence matrix (M x N, uint8) where A[k,i] = 1 if player i in lineup k
      sal -> lineup salaries (M,)
      w0  -> baseline weights (M,) from lineup scores
      tuples -> list of sorted player index tuples for each lineup
    """
    rng = np.random.default_rng(seed)
    N = len(players)
    # sampling probs ∝ exp(beta_sample * RealScore)
    rs = np.asarray(realscore, dtype=float)
    rs = (rs - rs.mean()) / (rs.std(ddof=0) + 1e-8)
    p = np.exp(beta_sample * rs)
    p = p / p.sum()

    bank = []
    seen = set()
    attempts = 0
    max_attempts = bank_size * 200  # generous to find uniques
    while len(bank) < bank_size and attempts < max_attempts:
        attempts += 1
        cand_idx = rng.choice(N, size=6, replace=False, p=p)
        sal_sum = int(salaries[cand_idx].sum())
        if not (salary_min <= sal_sum <= salary_max):
            continue
        tup = tuple(sorted(int(i) for i in cand_idx))
        if tup in seen:  # uniqueness
            continue
        seen.add(tup)
        bank.append((tup, sal_sum))
    if len(bank) == 0:
        return np.zeros((0, N), dtype=np.uint8), np.zeros(0), np.zeros(0), []

    M = len(bank)
    A = np.zeros((M, N), dtype=np.uint8)
    sal = np.zeros(M, dtype=np.int32)
    # lineup score (quality)
    # LineupScore = sum(RealScore) + γ*(sum RG_fpts)/100 − δ*(leftover/100)
    gamma = 0.10
    delta = 0.02
    lineup_score = np.zeros(M, dtype=float)
    for k, (tup, s) in enumerate(bank):
        A[k, list(tup)] = 1
        sal[k] = s
        lineup_score[k] = realscore[list(tup)].sum() + gamma*(rg_fpts[list(tup)].sum()/100.0) - delta*((50000 - s)/100.0)

    # baseline weights w0 ∝ exp(lineup_score)
    ls = lineup_score - lineup_score.max()
    w0 = np.exp(ls)
    w0 = w0 / w0.sum()
    tuples = [t for (t, _) in bank]
    return A, sal, w0, tuples

# =========================================================
# Exposure targets & bands (from resonance)
# =========================================================
def compute_resonance_targets(df: pd.DataFrame, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns GTO targets (600-scale), lower/upper counts (size N each).
    """
    # provisional slot weights: salary-aware soft weight
    # use RealScore / Salary softmax → normalize to 600 slots
    q = (df["RealScore"] / df["Salary"]).clip(lower=1e-9)
    gto600 = 600.0 * (q / q.sum()).to_numpy()

    # Convert to % of lineups (not 600) for bands
    t_pct = gto600 / 6.0
    T = np.rint(L * t_pct / 100.0).astype(int)
    return gto600, t_pct, T

def compute_bands_from_T(T: np.ndarray, L: int, abs_pp: float, rel_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    A = int(round(L * abs_pp / 100.0))  # absolute band in counts (e.g., 2.0 p.p. → 3 counts at L=150)
    rel = np.ceil((rel_pct/100.0) * T).astype(int)
    band = np.maximum(A, np.maximum(rel, 1))
    lower = np.maximum(0, T - band)
    upper = T + band
    return lower, upper

# =========================================================
# Entropy solver (mirror descent over bank)
# =========================================================
def solve_exposures(A: np.ndarray,
                    w0: np.ndarray,
                    lower: np.ndarray,
                    upper: np.ndarray,
                    L: int,
                    max_iter: int = 2000,
                    eta: float = 0.05,
                    tol_counts: float = 0.5,
                    backoff_rounds: int = 2) -> Tuple[np.ndarray, Dict]:
    """
    Minimize KL(p || w0) s.t. lower/L ≤ A^T p ≤ upper/L, sum p = 1, p ≥ 0
    Mirror descent: p ← Normalize(p * exp( ±η · A[:,i] )) for violated constraints.
    """
    M, N = A.shape
    if M == 0:
        return np.zeros(0), {"converged": False, "iters": 0, "backoff": 0}

    p = w0.copy()
    backoff = 0
    for rd in range(backoff_rounds + 1):
        for it in range(max_iter):
            e = A.T @ p  # exposures as fraction of lineups (since p sums to 1)
            e_counts = e * L
            # Compute violations
            over = e_counts - upper
            under = lower - e_counts
            max_violation = max(float(over.max(initial=-1e9)), float(under.max(initial=-1e9)))
            if max_violation <= tol_counts:
                return p, {"converged": True, "iters": it + rd*max_iter, "backoff": backoff}

            # Multiplicative updates
            # For players over upper: push down p where A[:,i]=1
            # For players under lower: push up p where A[:,i]=1
            grad = np.zeros(M, dtype=float)
            if over.max() > tol_counts:
                idx = np.where(over > tol_counts)[0]
                if len(idx) > 0:
                    grad -= (A[:, idx] @ np.ones(len(idx)))  # subtract on lineups containing those players
            if under.max() > tol_counts:
                idx = np.where(under > tol_counts)[0]
                if len(idx) > 0:
                    grad += (A[:, idx] @ np.ones(len(idx)))

            p = p * np.exp(-eta * grad)
            s = p.sum()
            if s <= 0:
                p = np.ones(M)/M
            else:
                p = p / s

        # widen bands by +1 count globally and retry
        lower = np.maximum(0, lower - 0)  # keep lower
        upper = upper + 1
        backoff += 1

    return p, {"converged": False, "iters": max_iter*(backoff_rounds+1), "backoff": backoff}

# =========================================================
# UI — Controls
# =========================================================
st.sidebar.header("Inputs")
dk_file = st.sidebar.file_uploader("DraftKings Player Pool (CSV)", type=["csv"])
dg_file = st.sidebar.file_uploader("DataGolf Odds (CSV)", type=["csv"])
rg_file = st.sidebar.file_uploader("RotoGrinders Projections (CSV)", type=["csv"])

st.sidebar.header("Resonance & Feasibility")
lambda_psi = st.sidebar.slider("PSI Coherence (λ)", 0.0, 0.75, 0.30, step=0.05)
salary_min, salary_max = st.sidebar.slider("Salary Band", 49000, 50000, (49700, 50000), step=100)
bank_size = st.sidebar.select_slider("Lineup Bank Size", options=[20000, 30000, 50000, 75000, 100000], value=50000)
beta_sample = st.sidebar.slider("Sampling Temperature (β)", 2.0, 10.0, 6.0, step=0.5)
seed = st.sidebar.number_input("Random Seed", 0, 10_000_000, 42)

st.sidebar.header("Exposure Bands")
L = st.sidebar.slider("Number of Lineups (L)", 1, 150, 150)
abs_pp = st.sidebar.slider("Absolute Tolerance (p.p.)", 0.0, 5.0, 2.0, step=0.5)
rel_pct = st.sidebar.slider("Relative Tolerance (% of target)", 0.0, 30.0, 10.0, step=1.0)
backoff_rounds = st.sidebar.slider("Backoff Rounds (+1 to bands)", 0, 3, 2)

run = st.button("Run GTO v2 (Lineup-Feasible)")

# =========================================================
# Main — Execute
# =========================================================
if run:
    if not (dk_file and dg_file and rg_file):
        st.error("Upload all three files to run.")
        st.stop()

    st.info("Loading & merging data…")
    dk = load_dk(dk_file)
    dg = load_dg(dg_file)
    rg = load_rg(rg_file)

    df = dk.merge(dg, on="Player", how="left").merge(rg, on="Player", how="left")
    # Fill small gaps
    for c in ["DG_win","DG_top5","DG_top10","DG_top20","RG_fpts","RG_ceil","RG_floor","PSI_raw"]:
        if c not in df: df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    # If RG salary exists, use DK salary as master; RG salary only for QA.
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)

    st.info("Resonance scoring (variance-driven + PSI coherence)…")
    df = resonance_scores(df, lambda_psi=lambda_psi)

    st.info("Generating lineup bank (feasible, unique)…")
    players = df["Player"].tolist()
    salaries = df["Salary"].to_numpy(dtype=np.int32)
    real = df["RealScore"].to_numpy(dtype=float)
    rg_fpts = df["RG_fpts"].to_numpy(dtype=float)

    A, bank_sal, w0, tuples = build_lineup_bank(
        players, salaries, real, rg_fpts,
        salary_min=salary_min, salary_max=salary_max,
        bank_size=bank_size, beta_sample=beta_sample, seed=seed
    )

    if A.shape[0] == 0:
        st.error("Could not generate any feasible unique lineups in the specified salary band. Loosen salary band or check inputs.")
        st.stop()

    M, N = A.shape
    st.success(f"Lineup bank: {M} lineups • {N} players • Avg salary ${bank_sal.mean():,.0f} • ≥$49.9k ratio {(bank_sal>=49900).mean():.0%}")

    st.info("Computing resonance targets & exposure bands…")
    gto600_prov, t_pct, T = compute_resonance_targets(df, L)
    lower, upper = compute_bands_from_T(T, L, abs_pp, rel_pct)

    # Entropy solver
    st.info("Solving exposures over the bank (entropy-preserving)…")
    p, meta = solve_exposures(
        A=A, w0=w0, lower=lower, upper=upper,
        L=L, max_iter=2000, eta=0.05, tol_counts=0.5, backoff_rounds=backoff_rounds
    )
    if not meta.get("converged", False):
        st.warning("Solver reached iteration/ backoff limit; exposures are as close as possible to bands.")

    # Final GTO exposures from p (feasible, lineup-aware)
    frac_lineups = (A.T @ p)  # fraction of lineups per player
    gto600 = frac_lineups * L * (600.0/100.0) * (100.0/L)  # simplifies to frac * 600
    # Sanity normalize to sum ~600
    gto600 = gto600 / gto600.sum() * 600.0

    # Public ownership on 600 scale for comparison if provided
    po_raw = ensure_600(df.get("PSI_raw", pd.Series(np.nan, index=df.index)))
    if po_raw is not None and not po_raw.isna().all():
        po600 = po_raw / po_raw.sum() * 600.0
    else:
        po600 = np.full(N, np.nan)

    leverage = gto600 - po600

    # Assemble output
    out = df[["Player","PlayerID","Salary","DG_win","DG_top5","DG_top10","DG_top20","RG_fpts","RG_ceil","RG_floor","PSI","RealScore"]].copy()
    out["GTO_%"] = gto600
    out["PO_%"]  = po600
    out["Leverage_%"] = leverage

    st.subheader("GTO v2 — Lineup-Feasible Scorecard")
    st.write(f"∑ GTO_% = {out['GTO_%'].sum():.2f} (≈ 600)")
    st.dataframe(out.sort_values("GTO_%", ascending=False), use_container_width=True)

    # Downloads
    full_csv = io.StringIO()
    out.to_csv(full_csv, index=False)
    st.download_button("📥 Download Private Scorecard (full)", full_csv.getvalue(), file_name="gto_scorecard_resonant_v2.csv", mime="text/csv")

    builder_out = out.copy()
    builder_out["GTO_Ownership%"] = builder_out["GTO_%"]
    builder_csv = io.StringIO()
    builder_out.to_csv(builder_csv, index=False)
    st.download_button("📥 Download Builder Scorecard (GTO_Ownership%=GTO_% 600)", builder_csv.getvalue(), file_name="gto_scorecard_for_builder.csv", mime="text/csv")

    # Diagnostics
    st.subheader("Diagnostics")
    st.write(f"Solver meta: {meta}")
    st.write(f"KL baseline drift (approx): {np.sum(p * (np.log((p+1e-12)/(w0+1e-12)))):.4f}")
    st.write(f"Average bank salary: ${bank_sal.mean():,.0f} • Near cap ≥$49.9k: {(bank_sal>=49900).mean():.1%}")
    # Target RMSE (vs provisional gto600_prov) — optional but indicative
    rmse = math.sqrt(np.mean((gto600 - gto600_prov)**2))
    st.write(f"RMSE vs provisional resonance targets (600): {rmse:.3f}")

else:
    st.info("Upload DK Pool + DG Odds + RG Projections, set parameters, then click **Run GTO v2**.")










