"""Statistical comparison: synthetic vs MIMIC-IV-ED demo, and the effect of calibration.

Reads:
  ml/data/generated/synthetic_original.csv
  ml/data/generated/synthetic_mimic_calibrated.csv
  mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/{triage,edstays}.csv.gz

Writes: statistical_tests.json
"""
from __future__ import annotations
import json, re, sys
import numpy as np, pandas as pd
from scipy import stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
NUM = ["heart_rate", "respiratory_rate", "systolic_bp", "diastolic_bp",
       "oxygen_saturation", "temperature", "pain_level", "shock_index", "pulse_pressure"]
RANGES = {"age": (0, 120), "heart_rate": (10, 300), "respiratory_rate": (2, 80),
          "systolic_bp": (40, 300), "diastolic_bp": (20, 200), "oxygen_saturation": (50, 100),
          "temperature": (30.0, 45.0), "pain_level": (0, 10)}
SEED = 42


def derive(df):
    df = df.copy()
    for c, (lo, hi) in RANGES.items():
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").where(
                lambda s: (s >= lo) & (s <= hi))
    df["shock_index"] = df.heart_rate / df.systolic_bp.replace(0, np.nan)
    df["pulse_pressure"] = df.systolic_bp - df.diastolic_bp
    return df


def load_mimic():
    d = f"{ROOT}/mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed/"
    t = pd.read_csv(d + "triage.csv.gz")
    e = pd.read_csv(d + "edstays.csv.gz")
    m = t.merge(e[["stay_id", "gender", "arrival_transport"]], on="stay_id", how="left")

    def pain(v):
        if pd.isna(v):
            return np.nan
        g = re.search(r"\d+(\.\d+)?", str(v))
        return min(10, max(0, float(g.group()))) if g else np.nan

    out = pd.DataFrame({
        "age": np.nan, "sex": m.gender.str.upper(),
        "arrival_transport": m.arrival_transport.str.lower().str.replace(" ", "_"),
        "heart_rate": m.heartrate, "respiratory_rate": m.resprate,
        "systolic_bp": m.sbp, "diastolic_bp": m.dbp, "oxygen_saturation": m.o2sat,
        "temperature": (m.temperature - 32) * 5 / 9, "pain_level": m.pain.map(pain),
        "chief_complaint": m.chiefcomplaint, "esi": m.acuity})
    return out[out.esi.isin([1, 2, 3, 4, 5])].reset_index(drop=True)


def holm_bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    holm = np.empty(n)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (n - i) * p[idx]))
        holm[idx] = run
    bh = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, min(1.0, p[idx] * n / k))
        bh[idx] = prev
    return holm.tolist(), bh.tolist()


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp if sp else np.nan
    g = d * (1 - 3 / (4 * (na + nb) - 9))
    return float(d), float(g)


def cliffs_delta(a, b, cap=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    a = rng.choice(a, min(cap, len(a)), replace=False)
    b = rng.choice(b, min(cap, len(b)), replace=False)
    a = np.sort(a)
    gt = np.searchsorted(a, b, "left").sum()
    ge = np.searchsorted(a, b, "right").sum()
    n = len(a) * len(b)
    return float((ge + gt) / n - 1)


def welch_ci(a, b, conf=0.95):
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(1 - (1 - conf) / 2, df)
    return float(ma - mb), [float(ma - mb - t * se), float(ma - mb + t * se)], float(df)


def two_sample(syn, mim, label):
    rows, ps = [], []
    for f in NUM:
        a = syn[f].dropna().to_numpy()
        b = mim[f].dropna().to_numpy()
        if len(b) < 5:
            continue
        tt = stats.ttest_ind(a, b, equal_var=False)
        st = stats.ttest_ind(a, b, equal_var=True)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        ks = stats.ks_2samp(a, b)
        lev = stats.levene(a, b, center="median")
        d, g = cohens_d(a, b)
        diff, ci, df = welch_ci(a, b)
        rows.append({"feature": f, "n_syn": len(a), "n_mimic": len(b),
                     "mean_syn": float(a.mean()), "mean_mimic": float(b.mean()),
                     "sd_syn": float(a.std(ddof=1)), "sd_mimic": float(b.std(ddof=1)),
                     "mean_diff": diff, "ci95": ci, "welch_t": float(tt.statistic),
                     "welch_df": df, "welch_p": float(tt.pvalue),
                     "student_t": float(st.statistic), "student_p": float(st.pvalue),
                     "mannwhitney_u": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
                     "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                     "levene_p": float(lev.pvalue), "cohens_d": d, "hedges_g": g,
                     "cliffs_delta": cliffs_delta(a, b)})
        ps.append(float(tt.pvalue))
    holm, bh = holm_bh(ps)
    for r, h, b_ in zip(rows, holm, bh):
        r["welch_p_holm"] = h
        r["welch_p_bh"] = b_
    return {"label": label, "features": rows}


def quantile_paired(syn, mim, label, seed=SEED):
    """Pair MIMIC values with synthetic values at the same quantile rank."""
    rows, ps = [], []
    for f in NUM:
        b = mim[f].dropna().to_numpy()
        a_all = syn[f].dropna().to_numpy()
        if len(b) < 5:
            continue
        q = (np.arange(1, len(b) + 1) - 0.5) / len(b)
        a = np.quantile(a_all, q)
        b_sorted = np.sort(b)
        dif = a - b_sorted
        tt = stats.ttest_rel(a, b_sorted)
        w = stats.wilcoxon(a, b_sorted) if np.any(dif != 0) else None
        sd = dif.std(ddof=1)
        dz = float(dif.mean() / sd) if sd else np.nan
        se = sd / np.sqrt(len(dif))
        tcrit = stats.t.ppf(0.975, len(dif) - 1)
        rows.append({"feature": f, "n_pairs": len(dif), "mean_diff": float(dif.mean()),
                     "ci95": [float(dif.mean() - tcrit * se), float(dif.mean() + tcrit * se)],
                     "t": float(tt.statistic), "df": len(dif) - 1, "p": float(tt.pvalue),
                     "cohens_dz": dz,
                     "wilcoxon_p": float(w.pvalue) if w else None})
        ps.append(float(tt.pvalue))
    holm, bh = holm_bh(ps)
    for r, h, b_ in zip(rows, holm, bh):
        r["p_holm"] = h
        r["p_bh"] = b_
    return {"label": label, "features": rows}


def stratum_paired(syn, mim):
    """Pair synthetic and MIMIC class means within each usable ESI stratum."""
    classes = [1, 2, 3]
    pairs, detail = [], []
    for f in NUM:
        for c in classes:
            a = syn.loc[syn.esi == c, f].dropna()
            b = mim.loc[mim.esi == c, f].dropna()
            if len(b) < 5:
                continue
            sd_p = b.std(ddof=1)
            z = float((a.mean() - b.mean()) / sd_p) if sd_p else np.nan
            detail.append({"feature": f, "esi": c, "n_mimic": int(len(b)),
                           "mean_syn": float(a.mean()), "mean_mimic": float(b.mean()),
                           "diff": float(a.mean() - b.mean()), "std_diff": z})
            pairs.append(a.mean() - b.mean())
    pairs = np.array(pairs, float)
    tt = stats.ttest_1samp(pairs, 0)
    w = stats.wilcoxon(pairs)
    per_class = {}
    for c in classes:
        d = np.array([r["std_diff"] for r in detail if r["esi"] == c], float)
        t2 = stats.ttest_1samp(d, 0)
        per_class[str(c)] = {"n_features": len(d), "mean_std_diff": float(d.mean()),
                             "t": float(t2.statistic), "p": float(t2.pvalue)}
    return {"n_pairs": len(pairs), "mean_diff": float(pairs.mean()),
            "t": float(tt.statistic), "p": float(tt.pvalue),
            "wilcoxon_p": float(w.pvalue), "per_esi_standardised": per_class,
            "detail": detail}


def calibration_paired(syn, cal, mim):
    """Paired over features: distance to MIMIC before vs after calibration."""
    out = {}
    for metric in ["abs_mean_diff", "ks", "wasserstein", "cohens_d_abs"]:
        before, after, feats = [], [], []
        for f in NUM:
            b = mim[f].dropna().to_numpy()
            if len(b) < 5:
                continue
            s = syn[f].dropna().to_numpy()
            c = cal[f].dropna().to_numpy()
            if metric == "abs_mean_diff":
                v1, v2 = abs(s.mean() - b.mean()), abs(c.mean() - b.mean())
            elif metric == "ks":
                v1, v2 = stats.ks_2samp(s, b).statistic, stats.ks_2samp(c, b).statistic
            elif metric == "wasserstein":
                v1, v2 = stats.wasserstein_distance(s, b), stats.wasserstein_distance(c, b)
            else:
                v1, v2 = abs(cohens_d(s, b)[0]), abs(cohens_d(c, b)[0])
            before.append(float(v1)); after.append(float(v2)); feats.append(f)
        bef, aft = np.array(before), np.array(after)
        dif = aft - bef
        tt = stats.ttest_rel(aft, bef)
        w = stats.wilcoxon(aft, bef)
        sgn = stats.binomtest(int((dif < 0).sum()), len(dif), 0.5)
        sd = dif.std(ddof=1)
        out[metric] = {"features": feats, "before": before, "after": after,
                       "mean_before": float(bef.mean()), "mean_after": float(aft.mean()),
                       "mean_change": float(dif.mean()),
                       "t": float(tt.statistic), "df": len(dif) - 1, "p": float(tt.pvalue),
                       "cohens_dz": float(dif.mean() / sd) if sd else None,
                       "wilcoxon_p": float(w.pvalue),
                       "n_improved": int((dif < 0).sum()), "n_total": len(dif),
                       "sign_test_p": float(sgn.pvalue)}
    return out


def tost(syn, mim, margin_sd=0.2):
    """Equivalence test: is |mean difference| smaller than margin x MIMIC SD?"""
    rows = []
    for f in NUM:
        a = syn[f].dropna().to_numpy()
        b = mim[f].dropna().to_numpy()
        if len(b) < 5:
            continue
        m = margin_sd * b.std(ddof=1)
        diff, ci, df = welch_ci(a, b, 0.90)
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        p1 = stats.t.sf((diff + m) / se, df)
        p2 = stats.t.cdf((diff - m) / se, df)
        rows.append({"feature": f, "margin": float(m), "mean_diff": diff,
                     "ci90": ci, "p_tost": float(max(p1, p2)),
                     "equivalent": bool(max(p1, p2) < 0.05)})
    return {"margin_sd": margin_sd, "features": rows}


def categorical(syn, mim):
    out = []
    for col in ["sex", "arrival_transport", "esi"]:
        obs_s = syn[col].value_counts()
        obs_m = mim[col].value_counts()
        cats = sorted(set(obs_s.index) | set(obs_m.index), key=str)
        o = np.array([obs_m.get(c, 0) for c in cats], float)
        p = np.array([obs_s.get(c, 0) for c in cats], float)
        p = p / p.sum()
        exp = p * o.sum()
        keep = exp > 0
        o_k, exp_k = o[keep], exp[keep]
        exp_k = exp_k * (o_k.sum() / exp_k.sum()) if exp_k.sum() else exp_k
        chi = stats.chisquare(o_k, exp_k)
        dfree = keep.sum() - 1
        v = float(np.sqrt(chi.statistic / (o.sum() * dfree))) if dfree else np.nan
        tbl = np.vstack([[obs_s.get(c, 0) for c in cats], [obs_m.get(c, 0) for c in cats]])
        tbl = tbl[:, tbl.sum(0) > 0]
        ind = stats.chi2_contingency(tbl)
        q = o / o.sum()
        pp = p[keep] / p[keep].sum() if keep.all() else p
        jsd = float(np.sqrt(max(0.0, stats.entropy(q, (q + p) / 2, base=2) / 2
                                + stats.entropy(p, (q + p) / 2, base=2) / 2)))
        out.append({"column": col, "categories": [str(c) for c in cats],
                    "observed_mimic": o.tolist(), "expected_from_synth": exp.tolist(),
                    "chi2_gof": float(chi.statistic), "df": int(dfree), "excluded_categories": [str(c) for c,k in zip(cats,keep) if not k],
                    "p_gof": float(chi.pvalue), "cramers_v": v,
                    "chi2_independence_p": float(ind.pvalue),
                    "tvd": float(0.5 * np.abs(q - p).sum()), "js_distance": jsd})
    return out


def correlations(syn, cal, mim):
    pairs = [("heart_rate", "systolic_bp"), ("heart_rate", "respiratory_rate"),
             ("oxygen_saturation", "respiratory_rate"), ("systolic_bp", "diastolic_bp"),
             ("heart_rate", "esi"), ("oxygen_saturation", "esi"),
             ("respiratory_rate", "esi"), ("pain_level", "esi")]
    rows = []
    for x, y in pairs:
        def r(df, sp):
            d = df[[x, y]].dropna()
            f = stats.spearmanr if sp else stats.pearsonr
            res = f(d[x], d[y])
            return float(res[0]), len(d)
        sp = y == "esi"
        rs, ns = r(syn, sp); rc, nc = r(cal, sp); rm, nm = r(mim, sp)
        za = np.arctanh(np.clip(rs, -0.999, 0.999)); zb = np.arctanh(np.clip(rm, -0.999, 0.999))
        zc = np.arctanh(np.clip(rc, -0.999, 0.999))
        se = np.sqrt(1 / (ns - 3) + 1 / (nm - 3))
        z = (za - zb) / se
        se2 = np.sqrt(1 / (nc - 3) + 1 / (nm - 3))
        z2 = (zc - zb) / se2
        rows.append({"pair": f"{x} ~ {y}", "method": "spearman" if sp else "pearson",
                     "r_synth": rs, "r_calibrated": rc, "r_mimic": rm, "n_mimic": nm,
                     "fisher_z_synth_vs_mimic": float(z),
                     "p_synth_vs_mimic": float(2 * stats.norm.sf(abs(z))),
                     "fisher_z_calibrated_vs_mimic": float(z2),
                     "p_calibrated_vs_mimic": float(2 * stats.norm.sf(abs(z2)))})
    ps = [r["p_synth_vs_mimic"] for r in rows]
    holm, bh = holm_bh(ps)
    for r, h, b in zip(rows, holm, bh):
        r["p_holm"] = h; r["p_bh"] = b
    d_before = np.array([abs(r["r_synth"] - r["r_mimic"]) for r in rows])
    d_after = np.array([abs(r["r_calibrated"] - r["r_mimic"]) for r in rows])
    tt = stats.ttest_rel(d_after, d_before)
    w = stats.wilcoxon(d_after, d_before)
    return {"pairs": rows,
            "paired_abs_gap": {"mean_before": float(d_before.mean()),
                               "mean_after": float(d_after.mean()),
                               "t": float(tt.statistic), "df": len(rows) - 1,
                               "p": float(tt.pvalue), "wilcoxon_p": float(w.pvalue)}}


def normality(syn, mim):
    rows = []
    rng = np.random.default_rng(SEED)
    for f in NUM:
        b = mim[f].dropna().to_numpy()
        a = syn[f].dropna().to_numpy()
        if len(b) < 5:
            continue
        a_s = rng.choice(a, min(4000, len(a)), replace=False)
        rows.append({"feature": f,
                     "shapiro_p_mimic": float(stats.shapiro(b).pvalue) if len(b) <= 5000 else None,
                     "shapiro_p_synth_sample": float(stats.shapiro(a_s).pvalue),
                     "skew_mimic": float(stats.skew(b)), "kurtosis_mimic": float(stats.kurtosis(b))})
    return rows


def power(mim):
    # smallest standardised effect detectable at 80% power, alpha .05, n2=207 vs n1=12000
    n1, n2 = 12000, len(mim)
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    d = (za + zb) * np.sqrt(1 / n1 + 1 / n2)
    return {"n_synthetic": n1, "n_mimic": int(n2), "alpha": 0.05, "power": 0.80,
            "min_detectable_cohens_d": float(d)}


def main():
    syn = derive(pd.read_csv(f"{ROOT}/ml/data/generated/synthetic_original.csv"))
    cal = derive(pd.read_csv(f"{ROOT}/ml/data/generated/synthetic_mimic_calibrated.csv"))
    mim_raw = load_mimic()
    mim = derive(mim_raw)
    res = {
        "n": {"synthetic_original": len(syn), "synthetic_calibrated": len(cal), "mimic": len(mim)},
        "cleaning": {"note": "values outside ml/config.py FEATURE_RANGES set to NaN before testing",
                     "mimic_dropped_values": {c: int(pd.to_numeric(mim_raw[c], errors='coerce').notna().sum()
                                                    - mim[c].notna().sum()) for c in RANGES if c in mim_raw}},
        "two_sample_original_vs_mimic": two_sample(syn, mim, "synthetic_original vs mimic"),
        "two_sample_calibrated_vs_mimic": two_sample(cal, mim, "synthetic_calibrated vs mimic"),
        "quantile_paired_original_vs_mimic": quantile_paired(syn, mim, "quantile-matched pairs, original"),
        "quantile_paired_calibrated_vs_mimic": quantile_paired(cal, mim, "quantile-matched pairs, calibrated"),
        "stratum_paired_original": stratum_paired(syn, mim),
        "stratum_paired_calibrated": stratum_paired(cal, mim),
        "calibration_paired_over_features": calibration_paired(syn, cal, mim),
        "equivalence_tost_calibrated": tost(cal, mim),
        "equivalence_tost_original": tost(syn, mim),
        "categorical_original": categorical(syn, mim),
        "categorical_calibrated": categorical(cal, mim),
        "correlations": correlations(syn, cal, mim),
        "normality": normality(syn, mim),
        "power": power(mim),
    }
    with open("statistical_tests.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("written statistical_tests.json")


if __name__ == "__main__":
    main()
