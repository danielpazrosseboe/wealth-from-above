import os, re, glob, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pandas.errors import PerformanceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display

warnings.simplefilter("ignore", PerformanceWarning)

BASE     = "/content/drive/My Drive/Master_Thesis/Surveys"
OUT_PATH = os.path.join(BASE, "clusters_yeh_spec.csv")
MAP_PATH = os.path.join(BASE, "survey_round_shapefile_map.csv")

ASSET_VARS = ["water","toilet","floor","electricity","radio","tv","fridge","motorbike","car","phone","rooms_per_person"]
URBAN_CANDS = ["hv025","URBAN_RURA","URBAN_RUR","urban","Urban","URBAN"]

def pick_largest(paths): return max(paths, key=os.path.getsize) if paths else None
def to_num(s): return pd.to_numeric(s, errors="coerce")
def mode_numeric(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.mode().iloc[0] if s.notna().any() else np.nan
def coerce_urban(s, idx):
    if s is None: return pd.Series(np.nan, index=idx, dtype="float32")
    st = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=idx, dtype="float32")
    out[st.isin(["urban","1"])] = 1.0
    out[st.isin(["rural","2","0"])] = 0.0
    return out
def recode_binary_01(s):
    v = pd.to_numeric(s, errors="coerce")
    if v.dropna().isin([0,1]).all(): return v
    if v.dropna().isin([1,2]).all(): return v.map({1: 1, 2: 0})
    return v
def recode_floor(x):
    s = pd.to_numeric(x, errors="coerce"); o = pd.Series(np.nan, index=s.index, dtype="float32")
    o[s.isin([10,11,12,13])] = 1; o[s.isin([20,21,22])] = 2; o[s.isin([30,31,32,33,34,35])] = 3
    return o
def recode_water(x):
    s = pd.to_numeric(x, errors="coerce"); o = pd.Series(np.nan, index=s.index, dtype="float32")
    o[s.isin([40,43])] = 1; o[s.isin([22,42,61,96])] = 2; o[s.isin([21,41,51])] = 3
    o[s.isin([13,20])] = 4; o[s.isin([10,11,12,71])] = 5
    return o
def recode_toilet(hv205, hv225=None):
    s = pd.to_numeric(hv205, errors="coerce"); o = pd.Series(np.nan, index=s.index, dtype="float32")
    o[s.isin([30,31])] = 1
    o[s.isin([14,15,23,42,43,96])] = 2
    improved = [11,12,13,21,22,41]
    if hv225 is not None:
        sh = pd.to_numeric(hv225, errors="coerce")
        o[s.isin(improved) & (sh == 1)] = 3
        o[s.isin(improved) & (sh == 0)] = 4
    else:
        o[s.isin(improved)] = 4
    return o

GEO_CACHE = {}
def load_geo(shp_path):
    if shp_path in GEO_CACHE: return GEO_CACHE[shp_path]
    try:
        g = gpd.read_file(shp_path)
    except Exception:
        GEO_CACHE[shp_path] = None
        return None
    if "DHSCLUST" not in g.columns:
        GEO_CACHE[shp_path] = None
        return None
    g = g.copy()
    g["_cl_"] = to_num(g["DHSCLUST"])
    g = g[g["_cl_"].notna()]
    if len(g) == 0:
        GEO_CACHE[shp_path] = None
        return None

    try:
        if g.crs is not None and getattr(g.crs, "is_geographic", False) is False:
            g = g.to_crs(4326)
    except Exception:
        pass
    cent = g.geometry if g.geometry.geom_type.eq("Point").all() else g.geometry.centroid
    g["lat_final"] = cent.y
    g["lon_final"] = cent.x
    df = g[["_cl_","lat_final","lon_final"]].drop_duplicates("_cl_")
    clset = set(to_num(df["_cl_"]).dropna().astype(int).tolist())
    GEO_CACHE[shp_path] = (clset, df)
    return GEO_CACHE[shp_path]

def choose_best_shp(cc, hr_clusters):
    shps = sorted(glob.glob(os.path.join(BASE, f"{cc}GE*FL", "*.shp")))
    scored, approx = [], []
    for shp in shps:
        res = load_geo(shp)
        if res is None:
            continue
        ge_clusters, _ = res
        ov = len(hr_clusters & ge_clusters)
        union = len(hr_clusters | ge_clusters) or 1
        jac = ov / union
        approx.append((ov/len(hr_clusters), jac, len(ge_clusters), shp))
        if ov == len(hr_clusters):
            extra = len(ge_clusters - hr_clusters)
            scored.append((jac, -extra, -abs(len(ge_clusters) - len(hr_clusters)), os.path.getsize(shp), shp))
    if not scored:
        top = sorted(approx, reverse=True)[:5]
        msg = "\n".join([f"  cov={cov:.1%} jacc={jac:.3f} geN={geN} :: {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
                         for cov,jac,geN,p in top]) or "  (no DHSCLUST shapefiles found)"
        raise RuntimeError(f"[{cc}] No shapefile fully covers HR clusters (need 100% coverage). Top candidates:\n{msg}")
    return max(scored)[-1]

household_chunks, map_rows = [], []
dt_dirs = sorted([p for p in glob.glob(os.path.join(BASE, "*HR*DT")) if os.path.isdir(p)])
print(f"Found {len(dt_dirs)} HR folders in {BASE}")

for dt_dir in dt_dirs:
    hr_folder = os.path.basename(dt_dir)
    m = re.match(r"^([A-Z]{2})HR(.+?)DT$", hr_folder, re.IGNORECASE)
    if not m:
        print(f"Skip {hr_folder}: name mismatch");
        continue
    cc = m.group(1).upper()
    dta_path = pick_largest(glob.glob(os.path.join(dt_dir, "*.dta")) + glob.glob(os.path.join(dt_dir, "*.DTA")))
    if not dta_path:
        print(f"Skip {hr_folder}: no .dta");
        continue

    try:
        hr = pd.read_stata(dta_path, convert_categoricals=False)
    except Exception as e:
        print(f"Skip {hr_folder}: read error {e}");
        continue

    hr["_cl_"] = to_num(hr.get("hv001"))
    if hr["_cl_"].isna().all():
        print(f"Skip {hr_folder}: hv001 missing");
        continue
    hr_clusters = set(hr["_cl_"].dropna().astype(int).unique().tolist())

    best_shp = choose_best_shp(cc, hr_clusters)
    ge_clusters, geo_df = load_geo(best_shp)

    yrs = sorted(to_num(hr.get("hv007")).dropna().astype(int).unique().tolist())
    for y in (yrs or [np.nan]):
        map_rows.append({"country": cc, "year": int(y) if pd.notna(y) else np.nan, "hr_folder": hr_folder, "shp_path": best_shp,
                         "hr_n_clusters": len(hr_clusters), "ge_n_clusters": len(ge_clusters), "jaccard": len(hr_clusters & ge_clusters)/len(hr_clusters | ge_clusters)})

    merged = hr.merge(geo_df, on="_cl_", how="left")
    merged["hv001"] = to_num(merged.get("hv001"))
    merged["hv007"] = to_num(merged.get("hv007"))
    merged["country"] = (merged["hv000"].astype(str).str.extract(r"^([A-Za-z]{2})", expand=False).str.upper()
                         if "hv000" in merged.columns else cc)
    merged["lat"] = to_num(merged["lat_final"])
    merged["lon"] = to_num(merged["lon_final"])

    ur_col = next((c for c in URBAN_CANDS if c in merged.columns), None)
    merged["urban"] = coerce_urban(merged[ur_col] if ur_col else None, merged.index)

    rooms  = to_num(merged.get("hv216"))
    hhsize = to_num(merged.get("hv009"))
    rooms_pp = np.where(hhsize.gt(0), rooms / hhsize, np.nan)

    chunk = pd.DataFrame({
        "country": merged["country"],
        "hv007": merged["hv007"],
        "hv001": merged["hv001"],
        "lat": merged["lat"],
        "lon": merged["lon"],
        "urban": merged["urban"],
        "water": recode_water(merged.get("hv201")),
        "toilet": recode_toilet(merged.get("hv205", merged.get("hv204")), merged.get("hv225") if "hv225" in merged.columns else None),
        "floor": recode_floor(merged.get("hv213", merged.get("hv213a"))),
        "electricity": recode_binary_01(merged.get("hv206")),
        "radio": recode_binary_01(merged.get("hv207")),
        "tv": recode_binary_01(merged.get("hv208")),
        "fridge": recode_binary_01(merged.get("hv209")),
        "motorbike": recode_binary_01(merged.get("hv211")),
        "car": recode_binary_01(merged.get("hv212")),
        "phone": recode_binary_01(merged.get("hv221", merged.get("hv243a"))),
        "rooms_per_person": rooms_pp,
    })

    chunk = chunk[chunk["hv001"].notna() & chunk["hv007"].notna() & chunk["country"].notna()]
    if len(chunk): household_chunks.append(chunk)

map_df = pd.DataFrame(map_rows).drop_duplicates(subset=["country","year","hr_folder","shp_path"])
map_df.to_csv(MAP_PATH, index=False)
print(f"\nSaved shapefile mapping to {MAP_PATH}")

if not household_chunks:
    raise RuntimeError("NO DATA PROCESSED. Check BASE path and file names.")

hh = pd.concat(household_chunks, ignore_index=True)
X = hh[ASSET_VARS].copy()
X = X.fillna(X.median(axis=0)).to_numpy(dtype=np.float32)
Xs = StandardScaler().fit_transform(X)
w  = PCA(n_components=1, random_state=0).fit_transform(Xs).ravel()
hh["wealth_index_pca"] = ((w - w.mean()) / w.std(ddof=0)).astype("float32")

cl = (hh.groupby(["country","hv007","hv001"], as_index=False)
        .agg(wealthpooled=("wealth_index_pca","mean"),
             lat=("lat","mean"), lon=("lon","mean"),
             n_households=("wealth_index_pca","size"),
             urban=("urban", mode_numeric))
        .rename(columns={"hv007":"year","hv001":"cluster"}))

cl = cl.loc[~((cl["lat"].abs() <= 0.01) & (cl["lon"].abs() <= 0.01))]
cl.to_csv(OUT_PATH, index=False)
print(f"\nSUCCESS: Saved {len(cl)} clusters to {OUT_PATH}")
display(cl.head())
