import io, re, unicodedata, time, math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fairway Theory — GTO v2 (Adaptive & Lineup-Feasible)", layout="wide")

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
    if 80 <= tot <= 120:  # looks like 100-scale totals
        return s * 6.0
    return s

def tier_from_salary(sal: pd.Series) -> pd.Series:
    try:
        return pd.qcut(sal, 4, labels=["Value","Mid","Upper","Stud"])
    except Exception:
        return pd.cut(sal, 4, labels=["Value","Mid","Upper","Stud"])

# =========================================================
# Loaders
# =========================================================
def load_dk(file) -> pd.DataFrame:
    dk = pd.read_csv(file)
    name_col  = next((c for c in dk.columns if c.lower() in ["name","player","player_name"]), None)
    id_col    = next((c for c in dk.columns if c.lower() in ["id","playerid","player_id"]), None)
    sal_col   = next((c for c in dk.columns if c.lower() == "salary"), None)
    if not name_col or not id_col or not sal_col:
        st.error("DK Pool must contain Name/Player, ID, and Salary.")
        st.stop()
    out = pd.DataFrame({
        "Player":  dk[name_col].map(normalize_name),
        "PlayerID": dk[id_col].astype(str),
        "Salary":  pd.to_numeric(dk[sal_col], errors="coerce")
    }).dropna(subset=["Player","Salary"])
    out["Salary"] = out["Salary"].astype(int)
    return out

def load_dg(file) -> pd.DataFrame:
    dg = pd.read_csv(file)
    need = ["player_name","win","top_5","top_10","top_20"]
    if not all(c in dg.columns for c in need):
        st.error(f"DG Odds must contain {need}.")
        st.stop()
    out = pd.DataFrame({
        "Player":   dg["player_name"].map(normalize_name),
        "DG_win":   pd.to_numeric(dg["win"], errors="coerce").fillna(0.0),
        "DG_top5":  pd.to_numeric(dg["top_5"], errors="coerce").fillna(0.0),
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
        "Player":  rg["name"].map(normalize_name),
        "RG_fpts": pd.to_numeric(rg["fpts"], errors="coerce").fillna(0.0),
        "RG_ceil": pd.to_numeric(rg["ceil"], errors="coerce").fillna(0.0),
        "RG_floor":pd.to_numeric(rg["floor"], errors="coerce").fillna(0.0),
        "PSI_raw": pd.to_numeric(rg["proj_own"], errors="coerce").fillna(0.0),  # public sentiment
        "RG_salary": pd.to_numeric(rg["salary"], errors="coerce").fillna(np.nan)
    })
    return out

# =========================================================
# Resonance scoring (variance-driven + PSI coherence)
# =========================================================
def resonance_scores(df: pd.DataFrame, lambda_psi: float = 0.30) -> pd.DataFrame:
    f_win   = safe_minmax(df["DG_win"])
    f_top20 = safe_minmax(df["DG_top20"])
    f_fpts  = safe_minmax(df["RG_fpts"])
    f_ceil  = safe_minmax(df["RG_ceil"])
    f_floor = safe_minmax(df["RG_floor"])
    eff     = safe_minmax((df["RG_fpts"].replace(0, np.nan) / df["Salary"].replace(0, np.nan)).fillna(0.0))
    df["Tier"] = tier_from_salary(df["Salary"])
    tier_map = {"Stud":0.80,"Upper":0.70,"Mid":0.60,"Value":0.50}
    f_tier   = df["Tier"].map(tier_map).astype(float).fillna(0.60)
    psi      = safe_minmax(df["PSI_raw"])

    parts = {
        "win": f_win, "top20": f_top20, "fpts": f_fpts,
        "ceil": f_ceil, "floor": f_floor, "eff": eff, "tier": f_tier
    }
    variances = {k: v.var() for k, v in parts.items()}
    tot_var = sum(variances.values()) or 1.0
    weights = {k: variances[k]/tot_var for k in parts}

    base = sum(parts[k]*weights[k] for k in parts)
    real = base * (1.0 + lambda_psi * psi)  # coherence amplifier
    df["RealScore"] = real
    df["PSI"] = psi
    return df

# =========================================================
# Build lineup bank (salary-valid, unique)
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
    rng = np.random.default_rng(seed)
    N = len(players)
    rs = np.asarray(realscore, dtype=float)
    rs = (rs - rs.mean()) / (rs.std(ddof=0) + 1e-8)
    p = np.exp(beta_sample * rs); p = p / p.sum()

    bank = []
    seen = set()
    attempts = 0
    max_attempts = bank_size * 200

    while len(bank) < bank_size and attempts < max_attempts:
        attempts += 1
        cand_idx = rng.choice(N, size=6, replace=False, p=p)
        sal_sum = int(salaries[cand_idx].sum())
        if not (salary_min <= sal_sum <= salary_max):
            continue
        tup = tuple(sorted(int(i) for i in cand_idx))
        if tup in seen:
            continue
        seen.add(tup)
        bank.append((tup, sal_sum))

    if len(bank) == 0:
        return np.zeros((0, N), dtype=np.uint8), np.zeros(0), np.zeros(0), []

    M = len(bank)
    A = np.zeros((M, N), dtype=np.uint8)
    sal = np.zeros(M, dtype=np.int32)
    gamma, delta = 0.10, 0.02  # projection bump, leftover penalty
    lineup_score = np.zeros(M, dtype=float)

    for k, (tup, s) in enumerate(bank):
        A[k, list(tup)] = 1
        sal[k] = s
        lineup_score[k] = realscore[list(tup)].sum() + gamma*(rg_fpts[list(tup)].sum()/100.0) - delta*((50000 - s)/100.0)

    w0 = np.exp(lineup_score - lineup_score.max())
    w0 = w0 / w0.sum()
    tuples = [t for (t, _) in bank]
    return A, sal, w0, tuples

# =========================================================
# Targets & Bands
# =========================================================
def compute_resonance_targets(df: pd.DataFrame, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = (df["RealScore"] / df["Salary"]).clip(lower=1e-9)
    gto600 = 600.0 * (q / q.sum()).to_numpy()
    t_pct = gto600 / 6.0
    T = np.rint(L * t_pct / 100.0).astype(int)
    return gto600, t_pct, T

def compute_bands_from_T(T: np.ndarray, L: int, abs_pp: float, rel_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    A = int(round(L * abs_pp / 100.0))
    rel = np.ceil((rel_pct/100.0) * T).astype(int)
    band = np.maximum(A, np.maximum(rel, 1))
    lower = np.maximum(0, T - band)
    upper = T + band
    return lower, upper

# =========================================================
# Adaptive entropy solver over the bank
# =========================================================
def solve_exposures_adaptive(A: np.ndarray,
                             w0: np.ndarray,
                             lower: np.ndarray,
                             upper: np.ndarray,
                             L: int,
                             max_seconds: float = 12.0,
                             max_iter: int = 2000,
                             eta: float = 0.05,
                             tol_counts: float = 0.5,
                             patience: int = 60,
                             global_backoff: int = 2,
                             status_cb=None) -> Tuple[np.ndarray, Dict]:
    """
    Minimize KL(p || w0) over lineups p s.t. lower/L ≤ A^T p ≤ upper/L.
    Adaptive features:
      - per-player widening when violations stall (patience)
      - time guard: returns best-so-far before timeout
      - final global backoff as last resort
    """
    start = time.time()
    M, N = A.shape
    if M == 0:
        return np.zeros(0), {"converged": False, "iters": 0, "backoff": 0}

    p = w0.copy()
    best_p, best_violation = p.copy(), float("inf")
    patience_ctr = 0
    backoff = 0

    def report(msg):
        if status_cb: status_cb(msg)

    # Simple pre-check: if a player never appears, clamp lower=0 for that player
    appear = A.sum(axis=0)  # how many lineups in bank include player i
    lower = np.where(appear == 0, 0, lower)

    while True:
        # inner mirror descent loop
        for it in range(max_iter):
            e = A.T @ p                # fraction of lineups
            e_counts = e * L           # expected lineup counts
            over = e_counts - upper
            under = lower - e_counts
            max_over  = float(over.max(initial=-1e9))
            max_under = float(under.max(initial=-1e9))
            max_violation = max(max_over, max_under)

            # track best
            if max_violation < best_violation:
                best_violation = max_violation
                best_p = p.copy()
                patience_ctr = 0
            else:
                patience_ctr += 1

            # done?
            if max_violation <= tol_counts:
                report(f"Converged: max violation ≤ {tol_counts:.2f} in {it + backoff*max_iter} iters, backoff={backoff}")
                return p, {"converged": True, "iters": it + backoff*max_iter, "backoff": backoff, "best_violation":best_violation}

            # adaptive band widening if stalling
            if patience_ctr >= patience:
                patience_ctr = 0
                # widen only the offenders by +1 on the side they violate
                widened = 0
                idx_over  = np.where(over  > tol_counts)[0]
                idx_under = np.where(under > tol_counts)[0]
                if idx_over.size:
                    upper[idx_over] += 1; widened += idx_over.size
                if idx_under.size:
                    lower[idx_under] = np.maximum(0, lower[idx_under] - 1); widened += idx_under.size
                report(f"Adapt: widened {widened} player bands (+1 on offending side).")
                # continue with same p

            # multiplicative update
            grad = np.zeros(M, dtype=float)
            if max_over  > tol_counts:
                idx = np.where(over  > tol_counts)[0]
                grad -= (A[:, idx] @ np.ones(len(idx)))
            if max_under > tol_counts:
                idx = np.where(under > tol_counts)[0]
                grad += (A[:, idx] @ np.ones(len(idx)))

            p = p * np.exp(-eta * grad)
            s = p.sum()
            if s <= 0: p = np.ones(M)/M
            else:      p = p / s

            # time guard
            if time.time() - start > max_seconds:
                report(f"Time guard hit ({max_seconds}s). Returning best-so-far (max violation {best_violation:.2f}).")
                return best_p, {"converged": best_violation<=tol_counts, "iters": it + backoff*max_iter, "backoff": backoff, "best_violation":best_violation}

        # global backoff (last resort)
        if backoff >= global_backoff:
            report(f"Global backoff limit reached. Returning best-so-far (max violation {best_violation:.2f}).")
            return best_p, {"converged": best_violation<=tol_counts, "iters": max_iter*(backoff+1), "backoff": backoff, "best_violation":best_violation}
        upper = upper + 1
        backoff += 1
        report(f"Global backoff +1 (round {backoff}/{global_backoff}).")

# =========================================================
# UI — Controls
# =========================================================
st.sidebar.header("Inputs")
dk_file = st.sidebar.file_uploader("DraftKings Player Pool (CSV)", type=["csv"])
dg_file = st.sidebar.file_uploader("DataGolf Odds (CSV)", type=["csv"])
rg_file = st.sidebar.file_uploader("RotoGrinders Projections (CSV)", type=["csv"])

st.sidebar.header("Resonance & Feasibility")
lambda_psi = st.sidebar.slider("PSI Coherence (λ)", 0.0, 0.75, 0.30, step=0.05)
salary_min, salary_max = st.sidebar.slider("Salary Band ($)", 49000, 50000, (49700, 50000), step=100)
bank_size   = st.sidebar.select_slider("Lineup Bank Size", options=[20000, 30000, 50000, 75000, 100000], value=50000)
beta_sample = st.sidebar.slider("Sampling Temperature (β)", 2.0, 10.0, 6.0, step=0.5)
seed        = st.sidebar.number_input("Random Seed", 0, 10_000_000, 42)

st.sidebar.header("Exposure Bands")
L = st.sidebar.slider("Number of Lineups (L)", 1, 150, 150)
abs_pp   = st.sidebar.slider("Absolute Tolerance (p.p.)", 0.0, 5.0, 2.0, step=0.5)
rel_pct  = st.sidebar.slider("Relative Tolerance (% of target)", 0.0, 30.0, 10.0, step=1.0)
patience = st.sidebar.slider("Adaptive Patience (iters)", 20, 200, 60, step=5)
max_seconds = st.sidebar.slider("Solver Time Guard (sec)", 5, 60, 12, step=1)
global_backoff = st.sidebar.slider("Global Backoff (rounds)", 0, 3, 2)

run = st.button("Run GTO v2 (Adaptive & Lineup-Feasible)")

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

    st.info("Resonance scoring (variance-driven + PSI coherence)…")
    df = resonance_scores(df, lambda_psi=lambda_psi)

    st.info("Generating lineup bank (feasible, unique)…")
    players  = df["Player"].tolist()
    salaries = df["Salary"].to_numpy(dtype=np.int32)
    real     = df["RealScore"].to_numpy(dtype=float)
    rg_fpts  = df["RG_fpts"].to_numpy(dtype=float)

    A, bank_sal, w0, tuples = build_lineup_bank(
        players, salaries, real, rg_fpts,
        salary_min=salary_min, salary_max=salary_max,
        bank_size=bank_size, beta_sample=beta_sample, seed=seed
    )
    if A.shape[0] == 0:
        st.error("No feasible unique lineups in the chosen salary band. Loosen the band or check inputs.")
        st.stop()

    M, N = A.shape
    st.success(f"Lineup bank: {M} lineups • {N} players • Avg salary ${bank_sal.mean():,.0f} • ≥$49.9k: {(bank_sal>=49900).mean():.0%}")

    st.info("Computing resonance targets & exposure bands…")
    gto600_prov, t_pct, T = compute_resonance_targets(df, L)
    lower, upper = compute_bands_from_T(T, L, abs_pp, rel_pct)

    # Solver status callback
    status_box = st.empty()
    def status(msg): status_box.info(msg)

    st.info("Solving exposures over the bank (adaptive entropy)…")
    p, meta = solve_exposures_adaptive(
        A=A, w0=w0, lower=lower, upper=upper, L=L,
        max_seconds=max_seconds, max_iter=2000, eta=0.05, tol_counts=0.5,
        patience=patience, global_backoff=global_backoff, status_cb=status
    )

    # Final GTO exposures (lineup-feasible)
    e_frac   = (A.T @ p)                 # fraction of lineups per player
    gto600   = e_frac * 600.0            # fraction * 600 (since 6 slots per lineup total 600)
    gto600   = gto600 / gto600.sum() * 600.0  # ensure exact sum 600

    # Public on 600 scale for leverage (optional)
    po_raw = ensure_600(df.get("PSI_raw", pd.Series(np.nan, index=df.index)))
    po600  = np.full(N, np.nan)
    if po_raw is not None and not po_raw.isna().all():
        s = po_raw.sum()
        if s > 0: po600 = po_raw / s * 600.0

    leverage = gto600 - po600

    # Assemble output
    out = df[["Player","PlayerID","Salary","DG_win","DG_top5","DG_top10","DG_top20","RG_fpts","RG_ceil","RG_floor","PSI","RealScore"]].copy()
    out["GTO_%"]      = gto600
    out["PO_%"]       = po600
    out["Leverage_%"] = leverage

    st.subheader("GTO v2 — Adaptive, Lineup-Feasible Scorecard")
    st.write(f"∑ GTO_% = {out['GTO_%'].sum():.2f} (≈ 600)")
    st.dataframe(out.sort_values("GTO_%", ascending=False), use_container_width=True)

    # Downloads
    full_csv = io.StringIO(); out.to_csv(full_csv, index=False)
    st.download_button("📥 Download Private Scorecard (full)", full_csv.getvalue(),
                       file_name="gto_scorecard_resonant_v2.csv", mime="text/csv")

    builder_out = out.copy()
    builder_out["GTO_Ownership%"] = builder_out["GTO_%"]
    builder_csv = io.StringIO(); builder_out.to_csv(builder_csv, index=False)
    st.download_button("📥 Download Builder Scorecard (GTO_Ownership%=GTO_% 600)", builder_csv.getvalue(),
                       file_name="gto_scorecard_for_builder.csv", mime="text/csv")

    # Diagnostics
    st.subheader("Diagnostics")
    st.write(f"Solver meta: {meta}")
    # KL drift (quality vs baseline)
    kl = float(np.sum(p * (np.log((p+1e-12)/(w0+1e-12)))))
    st.write(f"KL(p || baseline) ≈ {kl:.4f}")
    st.write(f"Average bank salary: ${bank_sal.mean():,.0f} • Near cap ≥$49.9k: {(bank_sal>=49900).mean():.1%}")
    rmse = math.sqrt(np.mean((gto600 - gto600_prov)**2))
    st.write(f"RMSE vs provisional resonance (600): {rmse:.3f}")

else:
    st.info("Upload DK Pool + DG Odds + RG Projections, set parameters, then click **Run GTO v2**.")












