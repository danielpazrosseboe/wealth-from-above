from __future__ import annotations
import os, re, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

class CONFIG:

    DHS_DIR      = str(PACKAGE_ROOT / "data" / "metadata")
    CLUSTERS_CSV = str(PACKAGE_ROOT / "data" / "metadata" / "clusters_wealth_index.csv")

    PREDS_IC   = str(PACKAGE_ROOT / "data" / "predictions" / "full_incountry_predictions.csv")
    PREDS_OOC  = str(PACKAGE_ROOT / "data" / "predictions" / "full_outofcountry_predictions.csv")
    PREDS_WA   = str(PACKAGE_ROOT / "data" / "predictions" / "west_africa_singlepatch_predictions.csv")
    PREDS_MIL  = str(PACKAGE_ROOT / "data" / "predictions" / "west_africa_mil_predictions.csv")

    META_CSV       = str(PACKAGE_ROOT / "data" / "metadata" / "dhs_combined_metadata.csv")
    ADM2_GPKG      = str(PACKAGE_ROOT / "data" / "boundaries" / "adm2_geoboundaries_combined.gpkg")
    COUNTRY_R2_WA  = str(PACKAGE_ROOT / "data" / "predictions" / "west_africa_country_r2_foldavg.csv")

    OUT_DIR = str(PACKAGE_ROOT / "outputs" / "analysis")

CC2NAME = {"AO":"Angola","BF":"Burkina Faso","BJ":"Benin","CD":"DR Congo",
           "CI":"Côte d'Ivoire","CM":"Cameroon","GA":"Gabon","GH":"Ghana",
           "GM":"Gambia","GN":"Guinea","KE":"Kenya","LB":"Liberia","LS":"Lesotho",
           "MD":"Madagascar","ML":"Mali","MR":"Mauritania","MZ":"Mozambique",
           "NG":"Nigeria","SL":"Sierra Leone","SN":"Senegal","TZ":"Tanzania","ZM":"Zambia"}
CC2ISO3 = {"AO":"AGO","BF":"BFA","BJ":"BEN","CD":"COD","CI":"CIV","CM":"CMR","GA":"GAB",
           "GH":"GHA","GM":"GMB","GN":"GIN","KE":"KEN","LB":"LBR","LS":"LSO","MD":"MDG",
           "ML":"MLI","MR":"MRT","MZ":"MOZ","NG":"NGA","SL":"SLE","SN":"SEN","TZ":"TZA","ZM":"ZMB"}
WA = {"BF","BJ","CI","GH","GM","GN","ML","MR","NG","SL","SN"}

def r2(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    m = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[m], yp[m]
    ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

def r2w(yt, yp, w):
    yt, yp, w = np.asarray(yt, float), np.asarray(yp, float), np.asarray(w, float)
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(w) & (w > 0)
    yt, yp, w = yt[m], yp[m], w[m]
    if len(yt) < 2: return np.nan
    ybar = (w * yt).sum() / w.sum()
    ss_res = (w * (yt - yp) ** 2).sum(); ss_tot = (w * (yt - ybar) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

def _col(df, *cands):
    for c in cands:
        if df is not None and c in df.columns: return c
    return None

def _load(path):
    if not os.path.exists(path):
        print(f"[WARN] missing: {path}"); return None
    return pd.read_csv(path)

def load_predictions() -> dict:
    d = {"ic": _load(CONFIG.PREDS_IC), "ooc": _load(CONFIG.PREDS_OOC),
         "wa": _load(CONFIG.PREDS_WA), "mil_raw": _load(CONFIG.PREDS_MIL)}
    mil = d["mil_raw"]
    if mil is not None and "mode" in mil.columns:
        d["mil_f"] = mil[mil["mode"] == "fused"]
        d["mil_m"] = mil[mil["mode"] == "ms"]
        d["mil_n"] = mil[mil["mode"] == "nl"]
    return d

def r2_summary(d: dict) -> dict:
    ic, ooc, wa = d["ic"], d["ooc"], d["wa"]
    out = {"pooled": {}, "country": {}}

    def by_cc(df, ycol):
        return {cc: r2(g["y_true"], g[ycol]) for cc, g in df.groupby("country")}

    for name, df in [("IC", ic), ("OOC", ooc)]:
        out["pooled"][name] = {"ms": r2(df["y_true"], df["yhat_ms"]),
                               "nl": r2(df["y_true"], df["yhat_nl"]),
                               "fused": r2(df["y_true"], df["yhat_fused"])}
        out["country"][name] = {"ms": by_cc(df, "yhat_ms"), "nl": by_cc(df, "yhat_nl"),
                                "fused": by_cc(df, "yhat_fused")}

    if {"yhat_ms", "yhat_nl"} <= set(wa.columns):
        wa_ms = by_cc(wa, "yhat_ms"); wa_nl = by_cc(wa, "yhat_nl")
        p_ms, p_nl = r2(wa["y_true"], wa["yhat_ms"]), r2(wa["y_true"], wa["yhat_nl"])
    else:
        wa2 = _load(CONFIG.COUNTRY_R2_WA); f1 = wa2[wa2["frac_train"] == 1.0] if wa2 is not None else None
        look = f1.groupby(["mode","country"])["test_r2_mean_across_folds"].mean() if f1 is not None else None
        get = lambda mode, cc: float(look.loc[(mode, cc)]) if look is not None and (mode, cc) in look.index else np.nan
        wa_ms = {cc: get("ms", cc) for cc in WA}; wa_nl = {cc: get("nl", cc) for cc in WA}
        p_ms = float(f1[f1["mode"]=="ms"]["test_r2_mean_across_folds"].mean()) if f1 is not None else np.nan
        p_nl = float(f1[f1["mode"]=="nl"]["test_r2_mean_across_folds"].mean()) if f1 is not None else np.nan
    wa_f = by_cc(wa, "y_pred")
    out["pooled"]["WA"] = {"ms": p_ms, "nl": p_nl, "fused": r2(wa["y_true"], wa["y_pred"])}
    out["country"]["WA"] = {"ms": wa_ms, "nl": wa_nl, "fused": wa_f}

    if "mil_f" in d:
        mf, mm, mn = d["mil_f"], d["mil_m"], d["mil_n"]
        out["pooled"]["MIL"] = {"ms": r2(mm["y_true"], mm["y_pred"]),
                                "nl": r2(mn["y_true"], mn["y_pred"]),
                                "fused": r2(mf["y_true"], mf["y_pred"])}
        out["country"]["MIL"] = {"ms": by_cc(mm, "y_pred"), "nl": by_cc(mn, "y_pred"),
                                 "fused": by_cc(mf, "y_pred")}
    return out

def per_fold_spread(d: dict) -> pd.DataFrame:
    rows = []
    def add(name, df, fused):
        if df is None: return
        yt = _col(df, "y_true", "wealthpooled"); fk = _col(df, "fold")
        pooled = r2(df[yt], df[fused])
        if fk is None:
            rows.append({"config": name, "pooled_R2": pooled, "n_folds": np.nan}); return
        per = np.array([r2(g[yt], g[fused]) for _, g in df.groupby(fk)], float)
        rows.append({"config": name, "pooled_R2": pooled, "n_folds": len(per),
                     "fold_min": per.min(), "fold_max": per.max(),
                     "fold_range": per.max()-per.min(),
                     "fold_SD": per.std(ddof=1) if len(per) > 1 else np.nan})
    add("Full-IC", d["ic"], "yhat_fused"); add("Full-OOC", d["ooc"], "yhat_fused")
    add("WA-SP", d["wa"], "y_pred")
    if "mil_f" in d: add("WA-MIL", d["mil_f"], "y_pred")
    return pd.DataFrame(rows)

def _attach_urban(df, clusters, fused):
    df = df.copy()
    if clusters is None or "urban" not in clusters.columns:
        df["urban"] = np.nan; return df
    if "key" in df.columns and {"country", "year", "cluster"} <= set(clusters.columns):
        ref = clusters.copy()
        ref["key"] = ref["country"].astype(str) + "|" + ref["year"].astype(str) + "|" + ref["cluster"].astype(str)
        df["urban"] = df["key"].map(ref.drop_duplicates("key").set_index("key")["urban"])
        return df
    yt = _col(df, "y_true", "wealthpooled"); urban = np.full(len(df), np.nan)
    for cc, g in df.groupby("country"):
        ref = clusters[clusters["country"] == cc]
        if ref.empty: continue
        rw, ru = ref["wealthpooled"].values, ref["urban"].values
        for i, v in zip(g.index, g[yt].values):
            urban[df.index.get_loc(i)] = ru[int(np.argmin(np.abs(rw - v)))]
    df["urban"] = urban; return df

def decomposition(d: dict, clusters) -> pd.DataFrame:
    out = []
    def row(name, scope, n, ms_r, nl_r, fu_r):
        return {"config": name, "scope": scope, "n": n,
                "MS_only": ms_r, "NL_only": nl_r, "fused": fu_r,
                "NTL_gain": (fu_r - ms_r) if (ms_r is not None and not np.isnan(ms_r)) else np.nan}
    def block(name, df, ms, nl, fused, scope, sub):
        if len(sub) <= 3: return row(name, scope, len(sub), np.nan, np.nan, np.nan)
        yt = _col(sub, "y_true", "wealthpooled")
        return row(name, scope, len(sub),
                   r2(sub[yt], sub[ms]) if ms else np.nan,
                   r2(sub[yt], sub[nl]) if nl else np.nan,
                   r2(sub[yt], sub[fused]))
    for name, df in [("Full-IC", d["ic"]), ("Full-OOC", d["ooc"])]:
        if df is None: continue
        du = _attach_urban(df, clusters, "yhat_fused")
        out.append(block(name, du, "yhat_ms", "yhat_nl", "yhat_fused", "Pooled", du))
        if du["urban"].notna().any():
            out.append(block(name, du, "yhat_ms", "yhat_nl", "yhat_fused", "Urban", du[du["urban"] == 1]))
            out.append(block(name, du, "yhat_ms", "yhat_nl", "yhat_fused", "Rural", du[du["urban"] == 0]))
    if d.get("wa") is not None:
        out.append(block("WA-SP", d["wa"], _col(d["wa"], "yhat_ms"), _col(d["wa"], "yhat_nl"), "y_pred", "Pooled", d["wa"]))
    if "mil_f" in d:
        mf, mm, mn = d["mil_f"], d["mil_m"], d["mil_n"]
        ms_r = r2(mm[_col(mm, "y_true", "wealthpooled")], mm["y_pred"])
        nl_r = r2(mn[_col(mn, "y_true", "wealthpooled")], mn["y_pred"])
        fu_r = r2(mf[_col(mf, "y_true", "wealthpooled")], mf["y_pred"])
        out.append(row("WA-MIL", "Pooled", len(mf), ms_r, nl_r, fu_r))
    return pd.DataFrame(out)

def bias_analysis(d: dict):
    rows, quint = [], {}
    for name, df, fused in [("Full-IC", d["ic"], "yhat_fused"), ("Full-OOC", d["ooc"], "yhat_fused"),
                            ("WA-SP", d["wa"], "y_pred"),
                            ("WA-MIL", d.get("mil_f"), "y_pred")]:
        if df is None: continue
        yt = _col(df, "y_true", "wealthpooled")
        y, p = df[yt].values.astype(float), df[fused].values.astype(float)
        slope, intercept = np.polyfit(y, p, 1); resid = p - y
        q = pd.qcut(y, 5, labels=[f"Q{i+1}" for i in range(5)])
        by_q = pd.DataFrame({"q": q, "r": resid}).groupby("q")["r"].mean()
        rows.append({"config": name, "slope_pred_on_true": round(float(slope),4),
                     "intercept": round(float(intercept),4),
                     "mean_resid_Q1": round(float(by_q.iloc[0]),4),
                     "mean_resid_Q5": round(float(by_q.iloc[-1]),4),
                     "compresses_extremes": bool(slope < 0.95),
                     "resid_SD": round(float(resid.std(ddof=1)),4)})
        quint[name] = by_q
    return pd.DataFrame(rows), quint

def _nearest_latlon(df_pred, meta, yt_col="y_true"):
    parts = []
    for cc, g in df_pred.groupby("country"):
        m = meta[meta["country"] == cc][["lat","lon","wealthpooled"]].reset_index(drop=True)
        g = g.copy().reset_index(drop=True)
        idx = [int(np.argmin(np.abs(m["wealthpooled"].values - v))) for v in g[yt_col].values]
        g["lat"] = m.iloc[idx]["lat"].values; g["lon"] = m.iloc[idx]["lon"].values
        parts.append(g)
    return pd.concat(parts, ignore_index=True)

def district_aggregation(d: dict):
    import geopandas as gpd
    meta = _load(CONFIG.META_CSV)
    if meta is None: raise FileNotFoundError(CONFIG.META_CSV)
    if "key" in d["ic"].columns and {"country", "year", "cluster"} <= set(meta.columns):
        m = meta.copy(); m["key"] = m["country"].astype(str) + "|" + m["year"].astype(str) + "|" + m["cluster"].astype(str)
        ic_c  = d["ic"].merge(m[["key","lat","lon"]], on="key", how="left").rename(columns={"yhat_fused":"y_pred"})
        ooc_c = d["ooc"].merge(m[["key","lat","lon"]], on="key", how="left").rename(columns={"yhat_fused":"y_pred"})
    elif "key" in meta.columns and "key" in d["ic"].columns:
        ic_c  = d["ic"].merge(meta[["key","lat","lon"]], on="key", how="left").rename(columns={"yhat_fused":"y_pred"})
        ooc_c = d["ooc"].merge(meta[["key","lat","lon"]], on="key", how="left").rename(columns={"yhat_fused":"y_pred"})
    else:
        ic_c  = _nearest_latlon(d["ic"].rename(columns={"yhat_fused":"y_pred"}), meta)
        ooc_c = _nearest_latlon(d["ooc"].rename(columns={"yhat_fused":"y_pred"}), meta)
    wa_c  = _nearest_latlon(d["wa"], meta)
    mil_c = _nearest_latlon(d["mil_f"], meta) if "mil_f" in d else None

    adm2 = gpd.read_file(CONFIG.ADM2_GPKG).to_crs("EPSG:4326")
    if "shapeID" in adm2.columns: adm2 = adm2.rename(columns={"shapeID": "adm2_id"})
    adm2 = adm2[[c for c in ["iso3","adm2_id","geometry"] if c in adm2.columns]]

    def join(df):
        df = df.copy(); df["country"] = df["country"].str.upper(); df["iso3"] = df["country"].map(CC2ISO3)
        df = df.dropna(subset=["lat","lon","iso3"])
        gpts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
        parts = []
        for iso3, pts in gpts.groupby("iso3"):
            poly = adm2[adm2["iso3"] == iso3]
            if poly.empty: continue
            j = gpd.sjoin(pts, poly, how="left", predicate="within")
            parts.append(pd.DataFrame(j.drop(columns=["geometry"], errors="ignore")))
        out = pd.concat(parts, ignore_index=True).dropna(subset=["adm2_id"])
        out["adm2_id"] = out["adm2_id"].astype(str)
        return out.groupby(["country","adm2_id"]).agg(n=("y_true","count"),
                yt=("y_true","mean"), yp=("y_pred","mean")).reset_index()

    cluster, district = {}, {}
    for k, df in [("IC", ic_c), ("OOC", ooc_c), ("WA", wa_c)] + ([("MIL", mil_c)] if mil_c is not None else []):
        agg = join(df)
        cluster[k]  = r2(df["y_true"], df["y_pred"])
        district[k] = r2w(agg.yt, agg.yp, agg.n)
    return cluster, district

def district_by_country_wa(d: dict):
    import geopandas as gpd
    meta = _load(CONFIG.META_CSV)
    if meta is None: raise FileNotFoundError(CONFIG.META_CSV)
    wa_c  = _nearest_latlon(d["wa"], meta)
    mil_c = _nearest_latlon(d["mil_f"], meta) if "mil_f" in d else None

    adm2 = gpd.read_file(CONFIG.ADM2_GPKG).to_crs("EPSG:4326")
    if "shapeID" in adm2.columns: adm2 = adm2.rename(columns={"shapeID": "adm2_id"})
    adm2 = adm2[[c for c in ["iso3", "adm2_id", "geometry"] if c in adm2.columns]]

    def agg_join(df):
        df = df.copy(); df["country"] = df["country"].str.upper(); df["iso3"] = df["country"].map(CC2ISO3)
        df = df.dropna(subset=["lat", "lon", "iso3"])
        gpts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
        parts = []
        for iso3, pts in gpts.groupby("iso3"):
            poly = adm2[adm2["iso3"] == iso3]
            if poly.empty: continue
            j = gpd.sjoin(pts, poly, how="left", predicate="within")
            parts.append(pd.DataFrame(j.drop(columns=["geometry"], errors="ignore")))
        out = pd.concat(parts, ignore_index=True).dropna(subset=["adm2_id"])
        out["adm2_id"] = out["adm2_id"].astype(str)
        return out.groupby(["country", "adm2_id"]).agg(n=("y_true", "count"),
                yt=("y_true", "mean"), yp=("y_pred", "mean")).reset_index()

    def by_cc(agg):
        per = {cc: r2w(g.yt, g.yp, g.n) for cc, g in agg.groupby("country")}
        return per, r2w(agg.yt, agg.yp, agg.n)

    res = {}
    per, pooled = by_cc(agg_join(wa_c)); res["WA"] = {"per": per, "pooled": pooled}
    if mil_c is not None:
        per, pooled = by_cc(agg_join(mil_c)); res["MIL"] = {"per": per, "pooled": pooled}
    return res

def headline_r2_cis(d: dict, B=10_000, seed=20260530) -> pd.DataFrame:
    def boot(df, fused):
        yt, yp = df["y_true"].to_numpy(float), df[fused].to_numpy(float)
        rng = np.random.default_rng(seed); n = len(yt); vals = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, n, n); vals[i] = r2(yt[idx], yp[idx])
        return r2(yt, yp), *np.percentile(vals, [2.5, 97.5])
    rows = []
    for name, df, fused in [("in-country", d["ic"], "yhat_fused"), ("out-of-country", d["ooc"], "yhat_fused")]:
        pt, lo, hi = boot(df, fused)
        rows.append({"scheme": name, "R2": round(pt,4), "ci_lo": round(lo,3), "ci_hi": round(hi,3)})
    return pd.DataFrame(rows)

def wa_mil_comparison(d: dict, B=10_000, seed=20260530):
    try:
        from scipy.stats import binomtest
    except Exception:
        import math
        class _BinomResult:
            def __init__(self, pvalue): self.pvalue = pvalue
        def binomtest(k, n, p=0.5, alternative="two-sided"):
            probs = [math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
            obs = probs[k]
            return _BinomResult(sum(v for v in probs if v <= obs + 1e-15))
    wa, mil = d["wa"], d.get("mil_f")
    if mil is None: return None
    sp_r2  = r2(wa["y_true"], wa["y_pred"]); mil_r2 = r2(mil["y_true"], mil["y_pred"])
    sp_cc  = {cc: r2(g["y_true"], g["y_pred"]) for cc, g in wa.groupby("country")}
    mil_cc = {cc: r2(g["y_true"], g["y_pred"]) for cc, g in mil.groupby("country")}
    common = sorted(set(sp_cc) & set(mil_cc))
    ahead  = sum(mil_cc[c] > sp_cc[c] for c in common)
    p = binomtest(ahead, len(common), 0.5, alternative="two-sided").pvalue
    rng = np.random.default_rng(seed)
    st, sp_ = wa["y_true"].to_numpy(float), wa["y_pred"].to_numpy(float)
    mt, mp_ = mil["y_true"].to_numpy(float), mil["y_pred"].to_numpy(float)
    diffs = np.empty(B)
    for i in range(B):
        diffs[i] = (r2(mt[rng.integers(0, len(mt), len(mt))], mp_[rng.integers(0, len(mp_), len(mp_))])
                    - r2(st[rng.integers(0, len(st), len(st))], sp_[rng.integers(0, len(sp_), len(sp_))]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"sp_pooled_R2": round(sp_r2,3), "mil_pooled_R2": round(mil_r2,3),
            "n_countries": len(common), "mil_ahead": ahead, "sign_test_p": round(float(p),2),
            "diff_ci": (round(float(lo),3), round(float(hi),3))}

def main():
    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
    clusters = _load(CONFIG.CLUSTERS_CSV)
    d = load_predictions()

    print("\n===== (2) POOLED R^2 SUMMARY =====")
    s = r2_summary(d)
    for k, v in s["pooled"].items():
        print(f"  {k:4s}  MS={v['ms']:.4f}  NL={v['nl']:.4f}  Fused={v['fused']:.4f}")

    print("\n===== (3A) PER-FOLD R^2 SPREAD =====")
    a = per_fold_spread(d); print(a.to_string(index=False))
    a.to_csv(os.path.join(CONFIG.OUT_DIR, "A_per_fold_R2.csv"), index=False)

    print("\n===== (3B) MS/NL/FUSED DECOMPOSITION =====")
    b = decomposition(d, clusters); print(b.to_string(index=False))
    b.to_csv(os.path.join(CONFIG.OUT_DIR, "B_ms_nl_decomposition.csv"), index=False)

    print("\n===== (3C) RESIDUAL / BIAS =====")
    c, quint = bias_analysis(d); print(c.to_string(index=False))
    c.to_csv(os.path.join(CONFIG.OUT_DIR, "C_bias_summary.csv"), index=False)
    pd.DataFrame(quint).to_csv(os.path.join(CONFIG.OUT_DIR, "C_residual_by_quintile.csv"))

    print("\n===== (5) HEADLINE R^2 BOOTSTRAP CIs =====")
    print(headline_r2_cis(d).to_string(index=False))

    print("\n===== (5) WEST AFRICA MIL vs SINGLE-PATCH =====")
    w = wa_mil_comparison(d)
    if w: print("  " + "  ".join(f"{k}={v}" for k, v in w.items()))

    try:
        print("\n===== (4) DISTRICT AGGREGATION (ADM2) =====")
        cl, di = district_aggregation(d)
        for k in cl: print(f"  {k:4s}  cluster={cl[k]:.4f}  district={di[k]:.4f}  d={di[k]-cl[k]:+.4f}")
    except Exception as e:
        print(f"  [skipped: needs {CONFIG.META_CSV} + {CONFIG.ADM2_GPKG}] {e}")

if __name__ == "__main__":
    main()
