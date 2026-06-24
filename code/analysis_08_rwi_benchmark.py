from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr
import analysis_results as A

RWI_DIR = str(A.PACKAGE_ROOT / "data" / "rwi")
ISO2CC = {v: k for k, v in A.CC2ISO3.items()}


def _rwi_path(cc):
    iso = A.CC2ISO3[cc].lower()
    cand = [os.path.join(RWI_DIR, f"{iso}_relative_wealth_index.csv"),
            f"{iso}_relative_wealth_index.csv"]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"RWI file for {cc} ({iso}) not found in {RWI_DIR}/")


def build_trees():
    trees = {}
    for cc in A.CC2ISO3:
        try:
            r = pd.read_csv(_rwi_path(cc))
        except FileNotFoundError:
            continue
        cl = np.cos(np.deg2rad(r["latitude"].mean()))
        trees[cc] = (cKDTree(np.c_[r["longitude"].values * cl, r["latitude"].values]),
                     r["rwi"].values, cl)
    return trees


def attach_rwi_by_coords(df, trees):
    df = df.reset_index(drop=True).copy()
    rwi = np.full(len(df), np.nan)
    dist = np.full(len(df), np.nan)
    for cc, g in df.groupby("country"):
        if cc not in trees:
            continue
        tree, vals, cl = trees[cc]
        d, idx = tree.query(np.c_[g["lon"].values * cl, g["lat"].values], k=1)
        rwi[g.index] = vals[idx]
        dist[g.index] = d * 111.0
    df["rwi"] = rwi
    df["rwi_match_km"] = dist
    return df


def attach_rwi_by_truth(pred, clusters_with_rwi):
    pred = pred.reset_index(drop=True).copy()
    rwi = np.full(len(pred), np.nan)
    for cc, gp in pred.groupby("country"):
        pool = defaultdict(deque)
        sub = clusters_with_rwi[clusters_with_rwi["country"] == cc]
        for w, rv in zip(sub["wealthpooled"].round(5), sub["rwi"]):
            pool[w].append(rv)
        for i, yt in zip(gp.index, gp["y_true"].round(5)):
            if pool[yt]:
                rwi[i] = pool[yt].popleft()
            else:
                ks = [k for k in pool if pool[k]]
                if ks:
                    k = ks[int(np.argmin(np.abs(np.array(ks) - yt)))]
                    rwi[i] = pool[k].popleft()
    pred["rwi"] = rwi
    return pred


def per_country(df, truth, model):
    rows = []
    for cc, g in df.groupby("country"):
        g = g.dropna(subset=[truth, model, "rwi"])
        if len(g) < 10:
            continue
        rows.append(dict(
            country=cc, name=A.CC2NAME.get(cc, cc), n=len(g),
            m_r2=pearsonr(g[truth], g[model])[0] ** 2,
            m_rho=spearmanr(g[truth], g[model])[0],
            r_r2=pearsonr(g[truth], g["rwi"])[0] ** 2,
            r_rho=spearmanr(g[truth], g["rwi"])[0]))
    return pd.DataFrame(rows)


def aggregate(t):
    w = t["n"].values
    return {k: float(np.average(t[k], weights=w)) for k in ["m_r2", "m_rho", "r_r2", "r_rho"]}


def load_keyed_with_coords(preds_csv, clusters):
    p = pd.read_csv(preds_csv)
    parts = p["key"].str.split("|", expand=True)
    p["year"] = parts[1].astype(int)
    p["cluster"] = parts[2].astype(int)
    return p.merge(clusters[["country", "year", "cluster", "lat", "lon"]],
                   on=["country", "year", "cluster"], how="inner")


def run():
    clusters = pd.read_csv(A.CONFIG.CLUSTERS_CSV)
    trees = build_trees()

    ic = attach_rwi_by_coords(load_keyed_with_coords(A.CONFIG.PREDS_IC, clusters), trees)
    ooc = attach_rwi_by_coords(load_keyed_with_coords(A.CONFIG.PREDS_OOC, clusters), trees)

    wac = clusters[clusters["country"].isin(A.WA)].reset_index(drop=True).copy()
    wac = attach_rwi_by_coords(wac.assign(lat=wac["lat"], lon=wac["lon"]), trees)

    sp = pd.read_csv(A.CONFIG.PREDS_WA)[["country", "y_true", "y_pred"]]
    sp = attach_rwi_by_truth(sp, wac).rename(columns={"y_pred": "yhat"})

    mil = pd.read_csv(A.CONFIG.PREDS_MIL)
    mil = mil[mil["mode"] == "fused"][["country", "y_true", "y_pred"]]
    mil = attach_rwi_by_truth(mil, wac).rename(columns={"y_pred": "yhat"})

    per = {
        "ic": per_country(ic, "y_true", "yhat_fused"),
        "ooc": per_country(ooc, "y_true", "yhat_fused"),
        "sp": per_country(sp, "y_true", "yhat"),
        "mil": per_country(mil, "y_true", "yhat"),
    }
    agg = {k: aggregate(v) for k, v in per.items()}
    raw = {"ic": ic, "ooc": ooc, "sp": sp, "mil": mil}
    return per, agg, raw


if __name__ == "__main__":
    per, agg, raw = run()
    labels = {"ic": "Full-IC", "ooc": "Full-OOC", "sp": "WA-SP", "mil": "WA-MIL"}
    print(f"{'config':<10}{'n':>8}{'rho_model':>11}{'rho_rwi':>9}{'R2_model':>10}{'R2_rwi':>9}")
    for k in ["ic", "ooc", "sp", "mil"]:
        a = agg[k]
        n = int(per[k]["n"].sum())
        print(f"{labels[k]:<10}{n:>8}{a['m_rho']:>11.3f}{a['r_rho']:>9.3f}{a['m_r2']:>10.3f}{a['r_r2']:>9.3f}")
    med = float(np.nanmedian(raw["ic"]["rwi_match_km"]))
    print(f"\nin-country nearest-RWI match: median {med:.2f} km")
    wins = int((per["ic"]["m_r2"] > per["ic"]["r_r2"]).sum())
    print(f"in-country: model R2 > RWI R2 in {wins}/{len(per['ic'])} countries")
