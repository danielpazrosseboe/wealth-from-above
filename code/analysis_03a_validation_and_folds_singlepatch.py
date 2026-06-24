FORCE_REBUILD = False
DISABLE_GPU   = False
MAKE_MAPS     = True
MAKE_PANELS   = True
K_PANELS      = 5

REFRESH_LOCAL_TFRECORDS = False
REFRESH_LOCAL_SURVEYS   = False
EXCLUDE_GLOBS = ["*.tfrecord.gz"]

import os, sys
from pathlib import Path

def _detect_base_master() -> Path:

    target_path = Path.home() / "Documents" / "Master"

    candidates = [
        target_path,

        Path.cwd(),
        Path.cwd() / "Master",
    ]

    for c in candidates:

        if c.exists():
            return c.resolve()

    return Path.cwd().resolve()

DRIVE_ROOT = _detect_base_master()

LOCAL_PROJECT_ROOT = DRIVE_ROOT
LOCAL_TF_ROOT      = (LOCAL_PROJECT_ROOT / "TFRecords").resolve()

LOCAL_SURVEYS_DIR  = (LOCAL_PROJECT_ROOT / "Master_Thesis" / "Surveys").resolve()

SRC_TF       = LOCAL_TF_ROOT
SRC_CLUSTERS = LOCAL_SURVEYS_DIR / "clusters_yeh_spec.csv"

print(f"--- LOCAL RUNTIME CONFIGURATION ---")
print(f"  System Platform      : {sys.platform}")
print(f"  Current Working Dir  : {Path.cwd()}")
print(f"  Detected DRIVE_ROOT  : {DRIVE_ROOT}")
print(f"  LOCAL_TF_ROOT        : {LOCAL_TF_ROOT}")
print(f"  LOCAL_SURVEYS_DIR    : {LOCAL_SURVEYS_DIR}")
print(f"  TFRecords Found      : {SRC_TF.exists()}")
print(f"  Clusters CSV Found   : {SRC_CLUSTERS.exists()}")

missing = []
if not SRC_TF.exists():
    missing.append(f"Missing TFRecords folder at: {SRC_TF}")
if not SRC_CLUSTERS.exists():
    missing.append(f"Missing clusters file at: {SRC_CLUSTERS}")

if missing:
    msg = "\n".join(missing) + (
        "\n\nTROUBLESHOOTING:\n"
        f"1. The script looked for data in: {DRIVE_ROOT}\n"
        "2. Ensure you have 'TFRecords' and 'Master_Thesis/Surveys' inside your Documents/Master folder."
    )
    raise FileNotFoundError(msg)
else:
    print("\nSUCCESS: Local data found. Pipeline ready.")

import os, sys, math, glob, re, io, gc, json, time, shutil, pickle, zipfile
from pathlib import Path
from collections import defaultdict

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import geopandas as gpd
import requests
import scipy.spatial
from sklearn.cluster import dbscan as _dbscan
from matplotlib.lines import Line2D

try:
    from IPython.display import display as nb_display
    from IPython.display import Image as _NBImage, display as _nbdisplay
except Exception:
    nb_display = print
    _NBImage = None
    _nbdisplay = print

try:
    PROJECT_ROOT = LOCAL_PROJECT_ROOT
    TF_ROOT      = LOCAL_TF_ROOT
    SURVEYS_DIR  = LOCAL_SURVEYS_DIR
except NameError:
    raise RuntimeError("Critical variables missing. Please run CELL 0 first.")

RAW_DIR        = TF_ROOT
PROCESSED_DIR  = TF_ROOT / "processed"
CLUSTERS_CSV   = SURVEYS_DIR / "clusters_yeh_spec.csv"

VALIDATION_DIR = PROJECT_ROOT / "Data Validation"
BAND_OUT_DIR   = VALIDATION_DIR / "Band Analysis"
DD_DIR         = PROJECT_ROOT / "Data Distribution"

NE_CACHE_DIR   = PROJECT_ROOT / "_cache" / "natural_earth"

for p in [PROCESSED_DIR, BAND_OUT_DIR, DD_DIR, NE_CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT  :", PROJECT_ROOT)
print("RAW_DIR       :", RAW_DIR)
print("PROCESSED_DIR :", PROCESSED_DIR)
print("SURVEYS_DIR   :", SURVEYS_DIR)
print("BAND_OUT_DIR  :", BAND_OUT_DIR)
print("DD_DIR        :", DD_DIR)
print("NE_CACHE_DIR  :", NE_CACHE_DIR)

if DISABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

plt.rcParams["figure.dpi"] = 120
print("TF version:", tf.__version__)

PATCH = 255
CROP  = 224
SCALE_M = 30
PIXELS_PATCH = PATCH * PATCH
PIXELS_CROP  = CROP * CROP

VIIRS_YEAR_MIN = 2012

BANDS_MS   = ["RED", "GREEN", "BLUE", "SWIR1", "SWIR2", "TEMP1", "NIR"]
BANDS_ALL  = BANDS_MS + ["NIGHTLIGHTS"]
BAND_ORDER = BANDS_ALL[:]

TEMP_SCALE  = 0.00341802
TEMP_OFFSET = 149.0

RADIUS_EARTH = 6356.7523

def savefig_dd(name_or_path, dpi=200):
    p = Path(name_or_path)
    if not p.is_absolute():
        p = DD_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    print("Saved figure:", p)

def _as_int(f, k):
    if k in f and f[k].int64_list.value:
        return int(f[k].int64_list.value[0])
    if k in f and f[k].float_list.value:
        return int(f[k].float_list.value[0])
    return None

def _as_float(f, k):
    if k in f and f[k].float_list.value:
        return float(f[k].float_list.value[0])
    if k in f and f[k].int64_list.value:
        return float(f[k].int64_list.value[0])
    return None

def _as_str(f, k):
    if k in f and f[k].bytes_list.value:
        return f[k].bytes_list.value[0].decode("utf-8")
    return None

def _as_arr(f, k):
    return np.asarray(f[k].float_list.value, dtype=np.float32) if k in f else None

def _center_crop(arr, patch=PATCH, crop=CROP):
    if arr is None or arr.size != patch * patch:
        return None
    z = arr.reshape(patch, patch)
    s = (patch - crop) // 2
    return z[s:s+crop, s:s+crop].ravel()

def _nl_center_mean(nl_arr):
    if nl_arr is None or nl_arr.size != PIXELS_PATCH:
        return np.nan, np.nan
    z = nl_arr.reshape(PATCH, PATCH)
    center_val = float(z[PATCH//2, PATCH//2])
    s = (PATCH - CROP) // 2
    zc = z[s:s+CROP, s:s+CROP]
    return center_val, float(zc.mean())

def list_raw_tfrecords_uncompressed():

    return sorted(str(p) for p in RAW_DIR.rglob("*.tfrecord"))

def list_raw_tfrecords_gz():
    return sorted(str(p) for p in RAW_DIR.glob("*.tfrecord.gz"))

def list_processed_per_example(root_dir):
    root_dir = Path(root_dir)
    return sorted(str(p) for p in root_dir.glob("*/*.tfrecord"))

def good_mask(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return np.isfinite(arr) & (arr > 0)

def init_band_stats(bands):
    return {
        b: {
            "n_pixels": 0, "n_good": 0,
            "sum": 0.0, "sum_sq": 0.0,
            "min": float("inf"), "max": float("-inf")
        }
        for b in bands
    }

def update_band_stats(stats, band, arr):
    if arr is None:
        return
    if band == "TEMP1":
        arr = arr.astype(np.float32) * TEMP_SCALE + TEMP_OFFSET
    m = good_mask(arr)
    s = stats[band]
    s["n_pixels"] += int(arr.size)
    good = arr[m]
    if good.size == 0:
        return
    s["n_good"] += int(good.size)
    s["sum"]    += float(good.sum())
    s["sum_sq"] += float((good**2).sum())
    s["min"]     = min(s["min"], float(good.min()))
    s["max"]     = max(s["max"], float(good.max()))

import re, json, time
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

PROCESSED_DIR = Path(r"C:\Users\d-rosseboe\Documents\Master\TFRecords\processed")
SURVEYS_DIR   = Path(r"C:\Users\d-rosseboe\Documents\Master\Master_Thesis\Surveys")
CLUSTERS_CSV  = SURVEYS_DIR / "clusters_yeh_spec.csv"

GOOD_LIST_CSV = "good_clusters.csv"
BAD_LIST_CSV  = "bad_clusters.csv"
DONE_MARKER   = "_SUCCESS.json"

CHECK_COORDS = True
LON_LAT_TOL  = 0.0

def as_int(fm, k):
    if k in fm and fm[k].int64_list.value:
        return int(fm[k].int64_list.value[0])
    if k in fm and fm[k].float_list.value:
        return int(fm[k].float_list.value[0])
    return None

def as_f32(fm, k):
    if k in fm and fm[k].float_list.value:
        return np.float32(fm[k].float_list.value[0])
    if k in fm and fm[k].int64_list.value:
        return np.float32(fm[k].int64_list.value[0])
    return None

def write_lists(out_dir: Path, cc: str, yr: int, good_set: set, bad_df: pd.DataFrame, extra_meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"cluster_index": sorted(map(int, good_set))}).to_csv(out_dir / GOOD_LIST_CSV, index=False)

    if bad_df is None or bad_df.empty:
        bad_df = pd.DataFrame(columns=["cluster_index","reason","n_bad_examples"])
    else:
        bad_df = bad_df.copy()
        bad_df["cluster_index"] = pd.to_numeric(bad_df["cluster_index"], errors="coerce").astype("Int64")
        bad_df = bad_df.dropna(subset=["cluster_index"]).copy()
        bad_df["cluster_index"] = bad_df["cluster_index"].astype(int)
        if "reason" not in bad_df.columns:
            bad_df["reason"] = "missing_after_processing:1"
        if "n_bad_examples" not in bad_df.columns:
            bad_df["n_bad_examples"] = 1
        bad_df = bad_df.sort_values(["reason","cluster_index"])

    bad_df.to_csv(out_dir / BAD_LIST_CSV, index=False)

    meta = {
        "country": cc,
        "year": int(yr),
        "n_good_clusters": int(len(set(good_set))),
        "n_bad_clusters": int(bad_df["cluster_index"].nunique() if len(bad_df) else 0),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra_meta or {}),
    }
    (out_dir / DONE_MARKER).write_text(json.dumps(meta, indent=2))

def scan_processed_folder(out_dir: Path, expected_latlon=None):
    good = set()
    coord_mismatch = {}
    year_counts = {}

    tf_files = sorted(out_dir.glob("*.tfrecord"))
    for p in tf_files:
        try:
            raw = next(iter(tf.data.TFRecordDataset(str(p)).take(1).as_numpy_iterator()))
            ex = tf.train.Example.FromString(raw)
            fm = ex.features.feature
        except Exception:
            continue

        ci = as_int(fm, "cluster_index")
        y  = as_int(fm, "year")
        if ci is not None:
            good.add(int(ci))
        if y is not None:
            year_counts[int(y)] = year_counts.get(int(y), 0) + 1

        if expected_latlon is not None and ci is not None:
            try:
                lon_ex = as_f32(fm, "lon")
                lat_ex = as_f32(fm, "lat") if "lat" in fm else None
                lon_csv = np.float32(expected_latlon.loc[int(ci), "lon"])
                lat_csv = np.float32(expected_latlon.loc[int(ci), "lat"])
                if (lon_ex is None) or (lat_ex is None):
                    coord_mismatch[int(ci)] = coord_mismatch.get(int(ci), 0) + 1
                else:
                    if LON_LAT_TOL == 0.0:
                        bad = (lon_ex != lon_csv) or (lat_ex != lat_csv)
                    else:
                        bad = (float(abs(lon_ex - lon_csv)) > LON_LAT_TOL) or (float(abs(lat_ex - lat_csv)) > LON_LAT_TOL)
                    if bad:
                        coord_mismatch[int(ci)] = coord_mismatch.get(int(ci), 0) + 1
            except Exception:
                coord_mismatch[int(ci)] = coord_mismatch.get(int(ci), 0) + 1

    return good, coord_mismatch, year_counts

if not CLUSTERS_CSV.exists():
    raise FileNotFoundError(f"Missing clusters CSV: {CLUSTERS_CSV}")

cl = pd.read_csv(CLUSTERS_CSV, float_precision="high")
cl["country"] = cl["country"].astype(str).str.upper().str[:2]
cl["year"]    = pd.to_numeric(cl["year"], errors="coerce").astype("Int64")

if "cluster_index" not in cl.columns and "cluster" in cl.columns:
    cl = cl.rename(columns={"cluster": "cluster_index"})
if "cluster_index" not in cl.columns:
    raise ValueError("clusters CSV must contain 'cluster' or 'cluster_index'.")

for c in ["cluster_index","lat","lon"]:
    if c not in cl.columns:
        raise ValueError(f"clusters CSV missing required column: {c}")

cl["cluster_index"] = pd.to_numeric(cl["cluster_index"], errors="coerce").astype("Int64")
cl["lat"] = pd.to_numeric(cl["lat"], errors="coerce")
cl["lon"] = pd.to_numeric(cl["lon"], errors="coerce")
cl = cl.dropna(subset=["country","year","cluster_index","lat","lon"]).copy()
cl["year"] = cl["year"].astype(int)
cl["cluster_index"] = cl["cluster_index"].astype(int)

SURVEY_LOOKUP = {}
for (cc, yr), g in cl.groupby(["country","year"]):
    SURVEY_LOOKUP[(cc, int(yr))] = g.set_index("cluster_index")[["lat","lon"]]

if not PROCESSED_DIR.exists():
    raise FileNotFoundError(f"Missing processed dir: {PROCESSED_DIR}")

round_rx = re.compile(r"^([A-Z]{2})_(\d{4})$", re.IGNORECASE)
round_dirs = [p for p in PROCESSED_DIR.iterdir() if p.is_dir() and round_rx.match(p.name)]
round_dirs = sorted(round_dirs, key=lambda p: (p.name[:2].upper(), int(p.name.split("_")[1])))

print(f"Found processed round folders: {len(round_dirs)} in {PROCESSED_DIR}")

n_written = 0
skipped = []
for out_dir in tqdm(round_dirs, desc="Writing good/bad"):
    m = round_rx.match(out_dir.name)
    cc = m.group(1).upper()
    yr = int(m.group(2))

    if (cc, yr) not in SURVEY_LOOKUP:
        skipped.append((cc, yr, "no_rows_in_clusters_csv"))
        continue

    expected_latlon = SURVEY_LOOKUP[(cc, yr)]
    expected = set(map(int, expected_latlon.index.tolist()))

    good_set, coord_mismatch, year_counts = scan_processed_folder(
        out_dir,
        expected_latlon if CHECK_COORDS else None
    )

    missing = expected - good_set

    bad_rows = []
    for ci in sorted(map(int, missing)):
        bad_rows.append({"cluster_index": ci, "reason": "missing_after_processing:1", "n_bad_examples": 1})

    if CHECK_COORDS and coord_mismatch:
        for ci, cnt in sorted(coord_mismatch.items()):

            if ci in expected:
                bad_rows.append({"cluster_index": int(ci), "reason": f"coord_mismatch_in_processed_examples:{int(cnt)}", "n_bad_examples": int(cnt)})

    bad_df = pd.DataFrame(bad_rows).drop_duplicates(subset=["cluster_index","reason"], keep="first") if bad_rows else pd.DataFrame()

    extra_meta = {
        "expected_clusters": int(len(expected)),
        "observed_clusters": int(len(good_set)),
        "missing_clusters": int(len(missing)),
        "coord_mismatch_clusters": int(len(coord_mismatch)) if CHECK_COORDS else 0,
        "year_counts_in_processed_examples": year_counts,
    }

    write_lists(out_dir, cc, yr, good_set, bad_df, extra_meta)
    n_written += 1

print(f"Done. Wrote good/bad lists for {n_written} round folders.")
if skipped:
    print("Skipped rounds (first 20):")
    for s in skipped[:20]:
        print(" -", s)

import os, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import geopandas as gpd
import requests

PROCESSED_DIR = Path(r"C:\Users\d-rosseboe\Documents\Master\TFRecords\processed")
SURVEYS_DIR   = Path(r"C:\Users\d-rosseboe\Documents\Master\Master_Thesis\Surveys")
CLUSTERS_CSV  = SURVEYS_DIR / "clusters_yeh_spec.csv"

GOOD_LIST_CSV = "good_clusters.csv"
BAD_LIST_CSV  = "bad_clusters.csv"
DONE_MARKER   = "_SUCCESS.json"

MAX_MAPS = None

if not MAKE_MAPS:
    print("MAKE_MAPS=False; skipping.")
    raise SystemExit

if not PROCESSED_DIR.exists():
    raise FileNotFoundError(f"Missing processed dir: {PROCESSED_DIR}")
if not CLUSTERS_CSV.exists():
    raise FileNotFoundError(f"Missing clusters CSV: {CLUSTERS_CSV}")

cl_all = pd.read_csv(CLUSTERS_CSV, float_precision="high")
cl_all["country"] = cl_all["country"].astype(str).str.upper().str[:2]
cl_all["year"]    = pd.to_numeric(cl_all["year"], errors="coerce").astype("Int64")

if "cluster_index" not in cl_all.columns and "cluster" in cl_all.columns:
    cl_all = cl_all.rename(columns={"cluster": "cluster_index"})
if "cluster_index" not in cl_all.columns:
    raise ValueError("clusters CSV must contain 'cluster' or 'cluster_index'.")

for c in ["cluster_index","lat","lon","wealthpooled"]:
    if c not in cl_all.columns:
        raise ValueError(f"clusters CSV missing required column for plotting: {c}")

cl_all["cluster_index"] = pd.to_numeric(cl_all["cluster_index"], errors="coerce").astype("Int64")
cl_all["lat"] = pd.to_numeric(cl_all["lat"], errors="coerce")
cl_all["lon"] = pd.to_numeric(cl_all["lon"], errors="coerce")
cl_all["wealthpooled"] = pd.to_numeric(cl_all["wealthpooled"], errors="coerce")
cl_all = cl_all.dropna(subset=["country","year","cluster_index","lat","lon","wealthpooled"]).copy()
cl_all["year"] = cl_all["year"].astype(int)
cl_all["cluster_index"] = cl_all["cluster_index"].astype(int)

round_rx = re.compile(r"^([A-Z]{2})_(\d{4})$", re.IGNORECASE)
round_dirs = [p for p in PROCESSED_DIR.iterdir() if p.is_dir() and round_rx.match(p.name)]
round_dirs = sorted(round_dirs, key=lambda p: (p.name[:2].upper(), int(p.name.split("_")[1])))

print(f"Found processed round folders: {len(round_dirs)} in {PROCESSED_DIR}")

def read_cluster_list(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "cluster_index" not in df.columns:
        return set()
    return set(pd.to_numeric(df["cluster_index"], errors="coerce").dropna().astype(int).tolist())

def read_bad_reasons(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["reason", "n_clusters", "n_bad_examples_sum"])
    df = pd.read_csv(path)
    if "reason" not in df.columns or "cluster_index" not in df.columns:
        return pd.DataFrame(columns=["reason", "n_clusters", "n_bad_examples_sum"])
    df = df.copy()
    df["cluster_index"] = pd.to_numeric(df["cluster_index"], errors="coerce")
    df = df.dropna(subset=["cluster_index"]).copy()
    df["cluster_index"] = df["cluster_index"].astype(int)
    df["reason"] = df["reason"].astype(str)

    if "n_bad_examples" in df.columns:
        df["n_bad_examples"] = pd.to_numeric(df["n_bad_examples"], errors="coerce").fillna(0).astype(int)
    else:
        df["n_bad_examples"] = 1

    agg = (df.groupby("reason", as_index=False)
             .agg(n_clusters=("cluster_index", "nunique"),
                  n_bad_examples_sum=("n_bad_examples", "sum"))
             .sort_values(["n_clusters","n_bad_examples_sum"], ascending=False))
    return agg

WORLD10, ISO_A2_COL = None, None
ISO_OVERRIDES = {"LB":"LR", "MD":"MG"}

NE10_URL = "https://github.com/nvkelso/natural-earth-vector/archive/refs/heads/master.zip"
NE10_DIR = NE_CACHE_DIR / "ne10"
NE10_ZIP = NE_CACHE_DIR / "ne10.zip"

def ensure_ne10():
    if NE10_DIR.exists():
        return
    NE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading Natural Earth vector data...")
    r = requests.get(NE10_URL, timeout=120)
    r.raise_for_status()
    NE10_ZIP.write_bytes(r.content)
    with zipfile.ZipFile(NE10_ZIP, "r") as z:
        z.extractall(NE10_DIR)

def load_admin0_10m():
    ensure_ne10()
    for root, _, fs in os.walk(NE10_DIR):
        if "ne_10m_admin_0_countries.shp" in fs:
            return gpd.read_file(os.path.join(root, "ne_10m_admin_0_countries.shp"))
    raise FileNotFoundError("ne_10m_admin_0_countries.shp not found.")

try:
    WORLD10 = load_admin0_10m()
    ISO_A2_COL = next((c for c in WORLD10.columns if str(c).upper().startswith("ISO_A2")), None)
except Exception as e:
    WORLD10, ISO_A2_COL = None, None
    print("Could not load Natural Earth 10m borders:", e)

def get_country_geom(cc):
    if WORLD10 is None or ISO_A2_COL is None:
        return None
    cc2 = ISO_OVERRIDES.get(cc, cc)
    sub = WORLD10.loc[WORLD10[ISO_A2_COL].astype(str) == cc2]
    return None if sub.empty else sub

def plot_country_year_map(country, year, survey_df, good_set, bad_set, bad_reasons_tbl=None):
    dfp = survey_df.copy()

    all_clusters = set(dfp["cluster_index"].astype(int).tolist())
    good = set(map(int, good_set))
    bad  = set(map(int, bad_set))

    missing = all_clusters - good - bad
    combined_bad = bad | missing

    n_total = len(all_clusters)
    n_good = len(good & all_clusters)
    n_bad  = len(combined_bad & all_clusters)
    pct_good = (100.0 * n_good / n_total) if n_total else 0.0
    pct_bad  = (100.0 * n_bad  / n_total) if n_total else 0.0

    good_df = dfp[dfp["cluster_index"].isin(good)].copy()
    bad_df  = dfp[dfp["cluster_index"].isin(combined_bad)].copy()

    fig, ax = plt.subplots(figsize=(7, 7))

    geom = get_country_geom(country)
    if geom is not None:
        geom.boundary.plot(ax=ax, linewidth=0.8)
        geom.plot(ax=ax, alpha=0.05)

    sc = None
    if len(good_df) > 0:
        vmin = float(good_df["wealthpooled"].quantile(0.01))
        vmax = float(good_df["wealthpooled"].quantile(0.99))
        sc = ax.scatter(
            good_df["lon"].astype(float),
            good_df["lat"].astype(float),
            c=good_df["wealthpooled"].astype(float),
            s=25, cmap="viridis", vmin=vmin, vmax=vmax, label="accepted"
        )

    if len(bad_df) > 0:
        ax.scatter(
            bad_df["lon"].astype(float),
            bad_df["lat"].astype(float),
            s=30, facecolors="none", edgecolors="red", linewidths=1.0, label="rejected/missing"
        )

    if sc is not None:
        fig.colorbar(sc, ax=ax).set_label("wealthpooled")

    title = f"{country} {year} — Good: {n_good}/{n_total} ({pct_good:.1f}%), Bad: {n_bad}/{n_total} ({pct_bad:.1f}%)"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", "box")
    ax.legend()

    if bad_reasons_tbl is not None and len(bad_reasons_tbl):
        show = bad_reasons_tbl.copy()

        show = show.head(8)

        tbl = ax.table(
            cellText=show[["reason","n_clusters","n_bad_examples_sum"]].values.tolist(),
            colLabels=["reason","n_clusters","n_bad_examples_sum"],
            loc="upper center",
            bbox=[0.0, 1.02, 1.0, 0.30],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)

    plt.tight_layout()
    plt.show()

to_plot = []
for d in round_dirs:
    m = round_rx.match(d.name)
    cc = m.group(1).upper()
    yr = int(m.group(2))

    sdf = cl_all[(cl_all["country"] == cc) & (cl_all["year"] == yr)].copy()
    if sdf.empty:
        continue

    good_set = read_cluster_list(d / GOOD_LIST_CSV)
    bad_set  = read_cluster_list(d / BAD_LIST_CSV)
    reasons  = read_bad_reasons(d / BAD_LIST_CSV)

    to_plot.append((cc, yr, sdf, good_set, bad_set, reasons))

print(f"Plotting maps for {len(to_plot)} country-year combinations.")

n_done = 0
for (cc, yr, sdf, good_set, bad_set, reasons) in tqdm(to_plot, desc="Plotting country-year maps"):
    plot_country_year_map(cc, yr, sdf, good_set, bad_set, bad_reasons_tbl=reasons)
    n_done += 1
    if MAX_MAPS is not None and n_done >= int(MAX_MAPS):
        break

import json, math, pickle, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

_need = ["RAW_DIR","PROCESSED_DIR","BAND_OUT_DIR","DD_DIR","FORCE_REBUILD","BANDS_ALL","PATCH","CROP","PIXELS_PATCH"]
_missing = [k for k in _need if k not in globals()]
if _missing:
    raise RuntimeError(f"Missing required globals (run earlier setup cells first): {_missing}")

SCAN_ROOT = Path(PROCESSED_DIR)
if not SCAN_ROOT.exists():
    SCAN_ROOT = Path(RAW_DIR)

print("SCAN_ROOT:", SCAN_ROOT)

BAND_JSON   = Path(BAND_OUT_DIR) / "band_stats_summary.json"
BAND_CSV    = Path(BAND_OUT_DIR) / "band_stats_summary.csv"

PICKLE_OUT  = Path(DD_DIR) / "tfrecord_scan_records.pkl"
NPZ_OUT     = Path(DD_DIR) / "tfrecord_scan_arrays.npz"
KEYMAP_PKL  = Path(DD_DIR) / "tfrecord_key_to_path.pkl"
MANIFEST    = Path(DD_DIR) / "scan_manifest.json"

TEMP_SCALE  = globals().get("TEMP_SCALE", None)
TEMP_OFFSET = globals().get("TEMP_OFFSET", None)

def _as_int_local(f, k):
    if k in f and f[k].int64_list.value: return int(f[k].int64_list.value[0])
    if k in f and f[k].float_list.value: return int(f[k].float_list.value[0])
    return None

def _as_float_local(f, k):
    if k in f and f[k].float_list.value: return float(f[k].float_list.value[0])
    if k in f and f[k].int64_list.value: return float(f[k].int64_list.value[0])
    return None

def _as_str_local(f, k):
    if k in f and f[k].bytes_list.value:
        try: return f[k].bytes_list.value[0].decode("utf-8")
        except Exception: return None
    return None

def _as_arr_local(f, k):
    if k not in f: return None
    return np.asarray(f[k].float_list.value, dtype=np.float32)

def _nl_center_mean_local(nl_arr, patch=PATCH, crop=CROP):
    if nl_arr is None or nl_arr.size != patch * patch:
        return np.nan, np.nan
    z = nl_arr.reshape(patch, patch)
    center_val = float(z[patch//2, patch//2])
    s = (patch - crop) // 2
    zc = z[s:s+crop, s:s+crop]
    return center_val, float(zc.mean())

_as_int   = globals().get("_as_int", _as_int_local)
_as_float = globals().get("_as_float", _as_float_local)
_as_str   = globals().get("_as_str", _as_str_local)
_as_arr   = globals().get("_as_arr", _as_arr_local)
_nl_center_mean = globals().get("_nl_center_mean", _nl_center_mean_local)

def good_mask(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return np.isfinite(arr) & (arr > 0)

def init_band_stats(bands):
    return {b: {"n_pixels":0,"n_good":0,"sum":0.0,"sum_sq":0.0,"min":float("inf"),"max":float("-inf")} for b in bands}

def update_band_stats(stats, band, arr):
    if arr is None: return
    arr = np.asarray(arr, dtype=np.float32)
    if band == "TEMP1" and (TEMP_SCALE is not None) and (TEMP_OFFSET is not None):
        arr = arr * np.float32(TEMP_SCALE) + np.float32(TEMP_OFFSET)
    m = good_mask(arr)
    s = stats[band]
    s["n_pixels"] += int(arr.size)
    good = arr[m]
    if good.size == 0: return
    s["n_good"] += int(good.size)
    s["sum"]    += float(good.sum())
    s["sum_sq"] += float((good**2).sum())
    s["min"]     = min(s["min"], float(good.min()))
    s["max"]     = max(s["max"], float(good.max()))

def list_per_example_tfrecords(scan_root: Path):
    scan_root = Path(scan_root)

    cand = sorted(str(p) for p in scan_root.glob("*/*.tfrecord"))
    if cand:
        return cand

    return sorted(str(p) for p in scan_root.rglob("*.tfrecord"))

def read_one_example(path: str):
    raw = next(iter(tf.data.TFRecordDataset(path).take(1).as_numpy_iterator()))
    return tf.train.Example.FromString(raw)

need_scan = FORCE_REBUILD or (not (BAND_JSON.exists() and BAND_CSV.exists()
                                  and PICKLE_OUT.exists() and NPZ_OUT.exists()
                                  and KEYMAP_PKL.exists() and MANIFEST.exists()))

if not need_scan:
    print("SKIP scan (all caches exist and FORCE_REBUILD=False). Loading...")
    band_stats_json = json.loads(BAND_JSON.read_text())
    scan_records    = pickle.loads(PICKLE_OUT.read_bytes())
    npz             = np.load(NPZ_OUT)
    years_arr       = np.asarray(npz["years"], dtype=np.int32)
    nls_center_arr  = np.asarray(npz["nls_center"], dtype=np.float32)
    nls_mean_arr    = np.asarray(npz["nls_mean"], dtype=np.float32)
    key_to_path     = pickle.loads(KEYMAP_PKL.read_bytes())
else:
    tf_paths = list_per_example_tfrecords(SCAN_ROOT)
    if not tf_paths:
        raise FileNotFoundError(f"No TFRecords found under: {SCAN_ROOT}")

    print(f"Scanning {len(tf_paths)} per-example TFRecords (keeping ALL rows)...")

    band_stats = init_band_stats(BANDS_ALL)

    scan_records = []
    key_to_path  = {}
    cluster_stats = defaultdict(lambda: {"n": 0, "n_bad": 0})

    NL_BAND = "NIGHTLIGHTS"
    examples_with_neg_nl = 0
    total_neg_pixels_nl = 0
    global_min_nl = float("inf")
    global_max_nl = float("-inf")

    for p in tqdm(tf_paths, desc="One-pass scan"):
        try:
            ex = read_one_example(p)
            f  = ex.features.feature
        except Exception:
            continue

        cc  = (_as_str(f, "country") or "").upper()[:2] or None
        yr  = _as_int(f, "year")
        clu = _as_int(f, "cluster_index") if "cluster_index" in f else _as_int(f, "cluster")

        key = (cc, int(yr) if yr is not None else None, int(clu) if clu is not None else None)
        if None in key:
            continue

        if key not in key_to_path:
            key_to_path[key] = p

        band_arrays = {}
        good_example = True
        for b in BANDS_ALL:
            arr = _as_arr(f, b)
            band_arrays[b] = arr
            if arr is None or arr.size != PIXELS_PATCH:
                good_example = False

        for b in BANDS_ALL:
            arr = band_arrays[b]
            if arr is not None and arr.size == PIXELS_PATCH:
                update_band_stats(band_stats, b, arr)

        nl_center, nl_mean = _nl_center_mean(band_arrays.get(NL_BAND))

        nl = band_arrays.get(NL_BAND)
        if nl is not None and nl.size == PIXELS_PATCH:
            finite = np.isfinite(nl)
            if finite.any():
                global_min_nl = min(global_min_nl, float(nl[finite].min()))
                global_max_nl = max(global_max_nl, float(nl[finite].max()))
            neg_mask = finite & (nl < 0)
            n_neg = int(neg_mask.sum())
            if n_neg > 0:
                examples_with_neg_nl += 1
                total_neg_pixels_nl += n_neg

        cluster_stats[key]["n"] += 1
        if not good_example:
            cluster_stats[key]["n_bad"] += 1

        scan_records.append({
            "country": key[0], "year": key[1], "cluster": key[2],
            "nls_center": float(nl_center),
            "nls_mean": float(nl_mean),
            "good_example": bool(good_example),
            "tf_path": p,
        })

    rows = []
    for b in BANDS_ALL:
        s = band_stats[b]
        n_pixels = int(s["n_pixels"])
        n_good   = int(s["n_good"])
        if n_good > 0:
            mean = s["sum"] / n_good
            ex2  = s["sum_sq"] / n_good
            var  = max(ex2 - mean**2, 0.0)
            std  = math.sqrt(var)
            bmin = s["min"]
            bmax = s["max"]
        else:
            mean = std = bmin = bmax = float("nan")
        frac_good = float(n_good) / n_pixels if n_pixels > 0 else float("nan")
        rows.append({
            "band": b, "n_pixels": n_pixels, "n_good": n_good, "frac_good": frac_good,
            "mean_good": mean, "std_good": std, "min_good": bmin, "max_good": bmax
        })

    df_stats = pd.DataFrame(rows).sort_values("band").reset_index(drop=True)
    try:
        nb_display(df_stats)
    except Exception:
        print(df_stats)

    band_stats_json = {r["band"]: {k: r[k] for k in ["n_pixels","n_good","frac_good","mean_good","std_good","min_good","max_good"]} for r in rows}
    BAND_JSON.write_text(json.dumps(band_stats_json, indent=2))
    df_stats.to_csv(BAND_CSV, index=False)

    df_rec = pd.DataFrame(scan_records)
    years_arr      = np.asarray(df_rec["year"].values, dtype=np.int32)
    nls_center_arr = np.asarray(df_rec["nls_center"].values, dtype=np.float32)
    nls_mean_arr   = np.asarray(df_rec["nls_mean"].values, dtype=np.float32)

    PICKLE_OUT.write_bytes(pickle.dumps(scan_records))
    np.savez_compressed(NPZ_OUT, years=years_arr, nls_center=nls_center_arr, nls_mean=nls_mean_arr)
    KEYMAP_PKL.write_bytes(pickle.dumps(key_to_path))

    good_keys = [k for k, st in cluster_stats.items() if st["n"] > 0 and st["n_bad"] == 0]
    bad_keys  = [k for k, st in cluster_stats.items() if st["n"] > 0 and st["n_bad"] > 0]

    MANIFEST.write_text(json.dumps({
        "scan_root": str(SCAN_ROOT),
        "n_files_scanned": int(len(tf_paths)),
        "n_records": int(len(scan_records)),
        "n_unique_keys": int(len(cluster_stats)),
        "n_good_keys": int(len(good_keys)),
        "n_bad_keys": int(len(bad_keys)),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nl_neg_examples": int(examples_with_neg_nl),
        "nl_neg_pixels": int(total_neg_pixels_nl),
        "nl_finite_min": float(global_min_nl) if np.isfinite(global_min_nl) else None,
        "nl_finite_max": float(global_max_nl) if np.isfinite(global_max_nl) else None,
    }, indent=2))

    print("Saved scan caches:")
    print(" -", PICKLE_OUT)
    print(" -", NPZ_OUT)
    print(" -", KEYMAP_PKL)
    print(" -", BAND_JSON)
    print(" -", BAND_CSV)
    print(" -", MANIFEST)

print(f"Ready: scan_records={len(scan_records)} | key_to_path={len(key_to_path)}")

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

LOC_DICT_PATH = Path(DD_DIR) / "dhs_loc_dict.pkl"

scan_records = pickle.loads((Path(DD_DIR) / "tfrecord_scan_records.pkl").read_bytes())

tfrecord_keys = set()
for r in scan_records:
    c, y, cl = r.get("country"), r.get("year"), r.get("cluster")
    if c is None or y is None or cl is None:
        continue
    tfrecord_keys.add((str(c).upper()[:2], int(y), int(cl)))

print("Unique (country,year,cluster) in TFRecords:", len(tfrecord_keys))

df_loc = pd.read_csv(CLUSTERS_CSV, float_precision="high")

df_loc["country"] = df_loc["country"].astype(str).str.upper().str[:2]
df_loc["year"]    = pd.to_numeric(df_loc["year"], errors="coerce").astype("Int64")

if "cluster" in df_loc.columns and "cluster_index" not in df_loc.columns:
    df_loc = df_loc.rename(columns={"cluster": "cluster_index"})
if "cluster_index" not in df_loc.columns:
    raise ValueError("clusters_yeh_spec.csv must contain 'cluster' or 'cluster_index'.")

df_loc["cluster_index"] = pd.to_numeric(df_loc["cluster_index"], errors="coerce").astype("Int64")

if "n_households" not in df_loc.columns and "households" in df_loc.columns:
    df_loc = df_loc.rename(columns={"households":"n_households"})

need_cols = {"country","year","cluster_index","wealthpooled","lat","lon","n_households","urban"}
missing = need_cols - set(df_loc.columns)
if missing:
    raise ValueError(f"clusters CSV missing columns: {sorted(missing)}")

df_loc["wealthpooled"] = pd.to_numeric(df_loc["wealthpooled"], errors="coerce").astype(float)
df_loc["lat"]          = pd.to_numeric(df_loc["lat"], errors="coerce").astype(float)
df_loc["lon"]          = pd.to_numeric(df_loc["lon"], errors="coerce").astype(float)
df_loc["n_households"] = pd.to_numeric(df_loc["n_households"], errors="coerce").fillna(0).astype(int)

if df_loc["urban"].dtype == object:
    df_loc["urban"] = (df_loc["urban"].astype(str).str.upper() == "U")
else:
    df_loc["urban"] = (pd.to_numeric(df_loc["urban"], errors="coerce").fillna(0).astype(int) == 1)

if "svyid" in df_loc.columns:
    df_loc["svyid_year"] = df_loc["svyid"].astype(str).str[-4:].astype(int)
else:
    df_loc["svyid_year"] = df_loc["year"].astype(int)

df_id = df_loc.dropna(subset=["country","year","cluster_index","lat","lon"]).copy()

dupes = df_id.duplicated(subset=["country","year","cluster_index"], keep=False)
if dupes.any():
    n_dupe_rows = int(dupes.sum())
    n_dupe_keys = int(df_id.loc[dupes, ["country","year","cluster_index"]].drop_duplicates().shape[0])
    print(f"WARNING: duplicate (country,year,cluster) rows: {n_dupe_rows} across {n_dupe_keys}. Keeping first.")

df_id = df_id.drop_duplicates(subset=["country","year","cluster_index"], keep="first")
print("df_id rows after dropna & drop_duplicates:", len(df_id))

loc_dict = {}
skipped = 0

for _, r in df_id.iterrows():
    key = (str(r["country"]), int(r["year"]), int(r["cluster_index"]))
    if key not in tfrecord_keys:
        skipped += 1
        continue

    loc_dict[key] = {
        "lat":          float(r["lat"]),
        "lon":          float(r["lon"]),
        "country":      str(r["country"]),
        "year":         int(r["year"]),
        "cluster":      int(r["cluster_index"]),
        "country_year": f"{str(r['country'])}_{int(r['svyid_year'])}",
        "svyid_year":   int(r["svyid_year"]),
        "households":   int(r["n_households"]),
        "urban":        bool(r["urban"]),
        "wealthpooled": float(r["wealthpooled"]),
    }

LOC_DICT_PATH.write_bytes(pickle.dumps(loc_dict))
print("Saved loc_dict:", LOC_DICT_PATH)
print("loc_dict entries:", len(loc_dict))
print("Skipped clusters lacking TFRecords:", skipped)

def get_lon_for_distance(lat_deg, d_km):
    lat_r = np.abs(lat_deg) * np.pi / 180.0
    r = RADIUS_EARTH * np.cos(lat_r)
    lon = d_km / r
    return lon * 180.0 / np.pi

def get_lat_for_distance(d_km):
    lat = d_km / RADIUS_EARTH
    return lat * 180.0 / np.pi

def print_loc_stats(locs):
    min_lat, min_lon = np.min(locs, axis=0)
    max_lat, max_lon = np.max(locs, axis=0)
    print(f"Lat min/max: {min_lat:.6f}, {max_lat:.6f}")
    print(f"Lon min/max: {min_lon:.6f}, {max_lon:.6f}")
    side_km = CROP * SCALE_M / 1000.0
    far_lat = max(abs(min_lat), abs(max_lat))
    side_lat = get_lat_for_distance(side_km)
    side_lon = get_lon_for_distance(far_lat, side_km)
    print(f"maximum side_lat: {side_lat:.6f}")
    print(f"maximum side_lon: {side_lon:.6f}")

def plot_locs_histogram(locs):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[8, 4])
    axs[0].hist(locs[:, 0], bins=100, orientation="horizontal")
    axs[0].set(xlabel="count", ylabel="latitude")
    axs[1].hist(locs[:, 1], bins=100, orientation="vertical")
    axs[1].set(xlabel="count", ylabel="longitude")
    savefig_dd("lat_lon_hist.png")
    plt.show()

print_loc_stats(LOCS)
plot_locs_histogram(LOCS)

import pickle, numpy as np, pandas as pd
from pathlib import Path

PICKLE_OUT     = Path(DD_DIR) / "tfrecord_scan_records.pkl"
NPZ_OUT        = Path(DD_DIR) / "tfrecord_scan_arrays.npz"
LOC_DICT_PATH  = Path(DD_DIR) / "dhs_loc_dict.pkl"

OUT_CSV = Path(DD_DIR) / "dhs_combined_df.csv"
OUT_PQ  = Path(DD_DIR) / "dhs_combined_df.parquet"

scan_records = pickle.loads(PICKLE_OUT.read_bytes())
npz = np.load(NPZ_OUT)

tf_df = pd.DataFrame(scan_records)
years      = np.asarray(npz["years"], dtype=np.int32)
nls_center = np.asarray(npz["nls_center"], dtype=np.float32)
nls_mean   = np.asarray(npz["nls_mean"], dtype=np.float32)

if not (len(tf_df) == len(years) == len(nls_center) == len(nls_mean)):
    raise AssertionError("NPZ arrays not aligned with scan_records length")

tf_df["country"] = tf_df["country"].astype(str).str.upper().str[:2]
tf_df["year"]    = pd.to_numeric(tf_df["year"], errors="coerce").astype(int)
tf_df["cluster"] = pd.to_numeric(tf_df["cluster"], errors="coerce").astype(int)

tf_df["nl_center"] = nls_center
tf_df["nl_mean"]   = nls_mean
tf_df["key"]       = list(zip(tf_df["country"], tf_df["year"], tf_df["cluster"]))

drop_if_present = ["lat","lon","urban","households","wealthpooled","svyid_year","country_year"]
tf_df = tf_df.drop(columns=[c for c in drop_if_present if c in tf_df.columns], errors="ignore")

loc_dict = pickle.loads(LOC_DICT_PATH.read_bytes())
ldf = pd.DataFrame([{"key": k, **v} for k, v in loc_dict.items()])

if ("country" not in ldf.columns) or ("year" not in ldf.columns) or ("cluster" not in ldf.columns):
    ldf["country"] = [k[0] for k in ldf["key"]]
    ldf["year"]    = [k[1] for k in ldf["key"]]
    ldf["cluster"] = [k[2] for k in ldf["key"]]

ldf["country"] = ldf["country"].astype(str).str.upper().str[:2]
ldf["year"]    = pd.to_numeric(ldf["year"], errors="coerce").astype(int)
ldf["cluster"] = pd.to_numeric(ldf["cluster"], errors="coerce").astype(int)
ldf["key"]     = list(zip(ldf["country"], ldf["year"], ldf["cluster"]))

need_loc_cols = ["lat","lon","urban","households","wealthpooled"]
missing = [c for c in need_loc_cols if c not in ldf.columns]
if missing:
    raise KeyError(f"loc_dict is missing required fields: {missing}. Rebuild dhs_loc_dict.pkl from clusters_yeh_spec.csv.")

if "svyid_year" not in ldf.columns:
    ldf["svyid_year"] = ldf["year"]
if "country_year" not in ldf.columns:
    ldf["country_year"] = ldf["country"].astype(str) + "_" + ldf["svyid_year"].astype(int).astype(str)

merged = tf_df.merge(
    ldf[["key","lat","lon","urban","households","wealthpooled","svyid_year","country_year"]],
    on="key",
    how="left",
    validate="many_to_one",
    suffixes=("_tf", "")
)

if "lat" not in merged.columns:
    raise KeyError(f"'lat' not present after merge. Columns are: {list(merged.columns)[:30]} ...")

if merged["lat"].isna().any():
    missing_n = int(merged["lat"].isna().sum())
    example = merged.loc[merged["lat"].isna(), "key"].head(5).tolist()
    raise AssertionError(f"{missing_n} TF examples had no match in loc_dict. Example missing keys: {example}")

df = pd.DataFrame({

    "country":       merged["country"].astype(str),
    "year":          pd.to_numeric(merged["svyid_year"], errors="coerce").astype(int),
    "cluster":       pd.to_numeric(merged["cluster"], errors="coerce").astype(int),
    "cluster_index": pd.to_numeric(merged["cluster"], errors="coerce").astype(int),

    "lat":           merged["lat"].astype(float),
    "lon":           merged["lon"].astype(float),
    "wealthpooled":  merged["wealthpooled"].astype(float),
    "urban":         merged["urban"].astype(int),
    "nl_mean":       merged["nl_mean"].astype(float),
    "nl_center":     merged["nl_center"].astype(float),
    "households":    merged["households"].astype(int),

    "tfrecord_year": merged["year"].astype(int),
})

try:
    nb_display(df.head())
except Exception:
    print(df.head())

df.to_csv(OUT_CSV, index=False)
try:
    df.to_parquet(OUT_PQ, index=False)
except Exception as e:
    print("Parquet write failed (install pyarrow):", e)

print("Saved combined df:")
print(" -", OUT_CSV)
print(" -", OUT_PQ)

LOCS = df[["lat","lon"]].to_numpy(dtype=np.float32)
print("LOCS shape:", LOCS.shape)

OOC_FOLDS   = DD_DIR / "dhs_ooc_folds.pkl"
OUT_PNG_OOC = DD_DIR / "africa_ooc_test_folds.png"

FOLDS  = ["A","B","C","D","E"]
SPLITS = ["train","val","test"]

need_ooc = FORCE_REBUILD or (not OOC_FOLDS.exists())

if need_ooc:
    COUNTRIES = np.array(sorted(df["country"].unique()))
    country_indices = defaultdict(list)
    for i in range(len(df)):
        country_indices[df["country"].iloc[i]].append(i)
    for c in list(country_indices.keys()):
        country_indices[c] = np.asarray(country_indices[c], dtype=np.int64)

    def make_ooc_folds_balanced(country_indices, folds=FOLDS):
        order = sorted(country_indices.items(), key=lambda kv: len(kv[1]), reverse=True)
        fold_countries = {f: [] for f in folds}
        for i, (cc, _) in enumerate(order):
            fold_countries[folds[i % len(folds)]].append(cc)

        ooc = {f: {s: None for s in SPLITS} for f in folds}
        for f in folds:
            test_countries  = fold_countries[f]
            val_countries   = fold_countries[folds[(folds.index(f)+1) % len(folds)]]
            train_countries = [c for c in COUNTRIES if c not in set(test_countries + val_countries)]

            ooc[f]["test"]  = np.sort(np.concatenate([country_indices[c] for c in test_countries]))  if test_countries else np.array([], dtype=int)
            ooc[f]["val"]   = np.sort(np.concatenate([country_indices[c] for c in val_countries]))   if val_countries else np.array([], dtype=int)
            ooc[f]["train"] = np.sort(np.concatenate([country_indices[c] for c in train_countries])) if train_countries else np.array([], dtype=int)
        return ooc

    ooc_folds = make_ooc_folds_balanced(country_indices)
    OOC_FOLDS.write_bytes(pickle.dumps(ooc_folds))
    print("Saved:", OOC_FOLDS)
else:
    ooc_folds = pickle.loads(OOC_FOLDS.read_bytes())
    print("Loaded:", OOC_FOLDS)

NE110_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world  = gpd.read_file(NE110_URL)
africa = world[world["CONTINENT"] == "Africa"].to_crs("EPSG:4326")

FOLD_COLORS = {"A":"#1f77b4","B":"#ff7f0e","C":"#d62728","D":"#2ca02c","E":"#9467bd"}

fig, ax = plt.subplots(figsize=(14,10))
ax.set_facecolor("#a6bddb")
africa.plot(ax=ax, color="#fff6d5", edgecolor="black", linewidth=0.5)

for f in FOLDS:
    idx = np.asarray(ooc_folds[f]["test"], dtype=int)
    if idx.size == 0:
        continue
    ax.scatter(LOCS[idx,1], LOCS[idx,0], s=6, c=FOLD_COLORS[f], label=f, alpha=0.85)

handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=FOLD_COLORS[f],
                  markersize=8, linestyle="") for f in FOLDS]
ax.legend(handles, FOLDS, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.set_xlim(-25,55); ax.set_ylim(-36,38)
ax.set_xticks(np.arange(-20,61,10)); ax.set_yticks(np.arange(-30,41,10))
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("DHS Out-of-Country TEST folds (A–E)")

savefig_dd(OUT_PNG_OOC)
plt.show()

INC_FOLDS   = DD_DIR / "dhs_incountry_folds.pkl"
OUT_PNG_INC = DD_DIR / "africa_incountry_test_folds.png"

need_inc = FORCE_REBUILD or (not INC_FOLDS.exists())

def get_lat_for_distance_km(d):
    return (d / RADIUS_EARTH) * 180.0 / np.pi

def get_lon_for_distance_km(lat_deg, d):
    r = RADIUS_EARTH * np.cos(np.abs(lat_deg) * np.pi / 180.0)
    return (d / r) * 180.0 / np.pi

def create_folds(locs, min_dist, dist_metric, fold_names):
    locs = np.asarray(locs, dtype=float)
    locs_to_indices = defaultdict(list)
    for i, loc in enumerate(locs):
        locs_to_indices[tuple(loc)].append(i)

    uniq = np.unique(locs, axis=0)
    _, labels = _dbscan(X=uniq, eps=min_dist, min_samples=2, metric=dist_metric)

    clusters = defaultdict(list)
    neg = -1
    for u, c in zip(uniq, labels):
        idxs = locs_to_indices[tuple(u)]
        if c < 0:
            c = neg
            neg -= 1
        clusters[c].extend(idxs)

    ordered = sorted(clusters.keys(), key=lambda c: -len(clusters[c]))
    fold_bins = {f: [] for f in fold_names}
    for c in ordered:
        f = min(fold_bins, key=lambda k: len(fold_bins[k]))
        fold_bins[f].extend(clusters[c])

    for f in fold_bins:
        fold_bins[f] = np.sort(np.asarray(fold_bins[f], dtype=np.int64))
    return fold_bins

if need_inc:
    side_km = CROP * SCALE_M / 1000.0
    far_lat = float(max(abs(LOCS[:,0].min()), abs(LOCS[:,0].max())))
    side_lat = get_lat_for_distance_km(side_km)
    side_lon = get_lon_for_distance_km(far_lat, side_km)
    MIN_DIST = float(math.hypot(side_lat, side_lon))

    test_folds = create_folds(LOCS, min_dist=MIN_DIST, dist_metric="euclidean", fold_names=FOLDS)

    incountry_folds = {}
    for i, f in enumerate(FOLDS):
        incountry_folds[f] = {}
        incountry_folds[f]["test"] = np.asarray(test_folds[f], dtype=np.int64)
        val_f = FOLDS[(i+1) % len(FOLDS)]
        incountry_folds[f]["val"] = np.asarray(test_folds[val_f], dtype=np.int64)
        train_fs = [FOLDS[(i+2)%5], FOLDS[(i+3)%5], FOLDS[(i+4)%5]]
        incountry_folds[f]["train"] = np.sort(np.concatenate([np.asarray(test_folds[x], dtype=np.int64) for x in train_fs]))

    INC_FOLDS.write_bytes(pickle.dumps(incountry_folds))
    print("Saved:", INC_FOLDS)
else:
    incountry_folds = pickle.loads(INC_FOLDS.read_bytes())
    print("Loaded:", INC_FOLDS)

NE110_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world  = gpd.read_file(NE110_URL)
africa = world[world["CONTINENT"] == "Africa"].to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(14,10))
ax.set_facecolor("#a6bddb")
africa.plot(ax=ax, color="#fff6d5", edgecolor="black", linewidth=0.5)

for f in FOLDS:
    idx = np.asarray(incountry_folds[f]["test"], dtype=int)
    if idx.size == 0:
        continue
    ax.scatter(LOCS[idx,1], LOCS[idx,0], s=6, c=FOLD_COLORS[f], label=f, alpha=0.85)

handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=FOLD_COLORS[f],
                  markersize=8, linestyle="") for f in FOLDS]
ax.legend(handles, FOLDS, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.set_xlim(-25,55); ax.set_ylim(-36,38)
ax.set_xticks(np.arange(-20,61,10)); ax.set_yticks(np.arange(-30,41,10))
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("DHS In-country TEST folds (A–E)")

savefig_dd(OUT_PNG_INC)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_countries_by_size(df_in):
    counts = df_in.groupby("country").size().sort_values(ascending=False)
    fig, ax = plt.subplots(1,1, figsize=(9,4))
    counts.plot(kind="bar", ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Count by country")
    ax.grid(True, axis="y")
    savefig_dd("countries_by_size.png")
    plt.show()

def boxplot_df(df_in, y, by, fig_name, figsize=(8,5), ylabel="", title=""):
    data   = df_in.groupby(by)[y].apply(list)
    labels = data.index
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.boxplot(list(data), patch_artist=True)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.grid(True, axis="y")
    savefig_dd(fig_name)
    plt.show()

plot_countries_by_size(df)

with pd.option_context("display.max_rows", 200):
    nb_display(df.groupby(["country","year"]).size().rename("count").to_frame())

nb_display(df.groupby("country")["wealthpooled"].describe())

boxplot_df(df, "wealthpooled", "country",
           fig_name="wealthpooled_by_country.png",
           figsize=(9,5), ylabel="wealthpooled",
           title="Wealthpooled by country")

boxplot_df(df, "wealthpooled", "year",
           fig_name="wealthpooled_by_year.png",
           figsize=(5,4), ylabel="wealthpooled",
           title="Wealthpooled by year")

viirs_mask = np.ones(len(df), dtype=bool)

boxplot_df(df[viirs_mask], "nl_center", ["country","year"],
           fig_name="viirs_center_by_country_year.png",
           figsize=(10,6), ylabel="nl_center",
           title="Center VIIRS distribution by country/year")

boxplot_df(df[viirs_mask], "nl_mean", ["country","year"],
           fig_name="viirs_mean_by_country_year.png",
           figsize=(10,6), ylabel="nl_mean",
           title="Mean VIIRS distribution by country/year")

print("share households < 16:", float((df["households"] < 16).mean()))

fig, ax = plt.subplots(1,1, figsize=(6,4))
df["households"].plot.hist(bins=20, grid=True, ax=ax)
ax.set(xlabel="# of households", ylabel="count", title="Households histogram")
savefig_dd("households_hist.png")
plt.show()

nb_display(df["households"].describe(percentiles=np.arange(0,1,0.01)).to_frame().T)

boxplot_df(df, "households", "country",
           fig_name="households_by_country.png",
           figsize=(9,5), ylabel="households",
           title="Households by country")

boxplot_df(df, "households", "year",
           fig_name="households_by_year.png",
           figsize=(5,4), ylabel="households",
           title="Households by year")

boxplot_df(df, "households", ["country","year"],
           fig_name="households_by_country_year.png",
           figsize=(12,6), ylabel="households",
           title="Households by country_year")

import numpy as np
import matplotlib.pyplot as plt

FOLDS  = ["A","B","C","D","E"]
SPLITS = ["train","val","test"]
DMSP_VIIRS_YEAR = 2012

def plot_labels_by_fold(labels, folds, title, fig_name):
    fig, axs = plt.subplots(1,5, sharey=True, figsize=(9,2.8))
    for f, ax in zip(FOLDS, axs.flat):
        data = [labels[folds[f][split]] for split in SPLITS]
        ax.boxplot(data, patch_artist=True, widths=0.8)
        plt.setp(ax, xticks=[1,2,3], xticklabels=SPLITS)
        ax.grid(True, axis="y")
        ax.set_title(f"Fold: {f}")
    axs[0].set_ylabel("wealthpooled")
    if title:
        fig.suptitle(title, y=1.03)
    savefig_dd(fig_name)
    plt.show()

def plot_urban_by_fold(urban, folds, title, fig_name):
    fig, axs = plt.subplots(1,5, sharey=True, figsize=(9,2.8))
    for f, ax in zip(FOLDS, axs.flat):
        data = [urban[folds[f][split]].mean() for split in SPLITS]
        ax.bar([0,1,2], data, width=0.8)
        plt.setp(ax, xticks=[0,1,2], xticklabels=SPLITS)
        ax.grid(True, axis="y")
        ax.set_title(f"Fold: {f}")
    axs[0].set_ylabel("urban fraction")
    if title:
        fig.suptitle(title, y=1.03)
    savefig_dd(fig_name)
    plt.show()

def plot_nl_by_fold_viirs(df_in, folds, col, title, fig_name):
    fig, axs = plt.subplots(1, 5, sharey=True, figsize=(11, 2.8))
    bin_edges = np.linspace(df_in[col].min() - 0.1, df_in[col].max() + 0.1, 100)
    centers = np.convolve(bin_edges, [.5, .5], mode="valid")
    for f, ax in zip(FOLDS, axs.flat):
        for split in SPLITS:
            idx = folds[f][split]
            sub = df_in.loc[idx, [col, "year"]]
            vals = sub.loc[sub["year"] >= DMSP_VIIRS_YEAR, col].values
            if vals.size == 0:
                continue
            hist, _ = np.histogram(vals, bins=bin_edges)
            if hist.sum() > 0:
                ax.plot(centers, hist / hist.sum(), label=split)
        ax.set_xlabel(col)
        ax.set_yscale("log")
        ax.set_title(f"Fold {f}")
        ax.grid(True)
    axs[0].legend()
    axs[0].set_ylabel("fraction")
    if title:
        fig.suptitle(title, y=1.03)
    savefig_dd(fig_name)
    plt.show()

def nl_boxplots_by_fold_viirs(df_in, folds, col, title, fig_name):
    fig, axs = plt.subplots(1, 5, sharey=True, figsize=(9, 2.8))
    for f, ax in zip(FOLDS, axs.flat):
        data = []
        for split in SPLITS:
            idx = folds[f][split]
            sub = df_in.loc[idx, [col, "year"]]
            vals = sub.loc[sub["year"] >= DMSP_VIIRS_YEAR, col].values
            data.append(vals)
        ax.boxplot(data, patch_artist=True, widths=0.8)
        plt.setp(ax, xticks=[1, 2, 3], xticklabels=SPLITS)
        ax.set_title(f"Fold {f}")
        ax.grid(True, axis="y")
    axs[0].set_ylabel(col)
    if title:
        fig.suptitle(title, y=1.03)
    savefig_dd(fig_name)
    plt.show()

plot_labels_by_fold(df["wealthpooled"].values, ooc_folds,
                    title="Wealthpooled by fold (OOC)",
                    fig_name="wealthpooled_by_fold_ooc.png")

plot_labels_by_fold(df["wealthpooled"].values, incountry_folds,
                    title="Wealthpooled by fold (incountry)",
                    fig_name="wealthpooled_by_fold_incountry.png")

plot_urban_by_fold(df["urban"].values.astype(float), ooc_folds,
                   title="Urban/rural by fold (OOC)",
                   fig_name="urban_fraction_by_fold_ooc.png")

plot_urban_by_fold(df["urban"].values.astype(float), incountry_folds,
                   title="Urban/rural by fold (incountry)",
                   fig_name="urban_fraction_by_fold_incountry.png")

plot_nl_by_fold_viirs(df, ooc_folds,       "nl_mean",   "VIIRS nl_mean (OOC)",       "viirs_nl_mean_ooc.png")
plot_nl_by_fold_viirs(df, incountry_folds, "nl_mean",   "VIIRS nl_mean (incountry)", "viirs_nl_mean_incountry.png")
nl_boxplots_by_fold_viirs(df, ooc_folds,       "nl_center", "VIIRS nl_center (OOC)",       "viirs_nl_center_ooc.png")
nl_boxplots_by_fold_viirs(df, incountry_folds, "nl_center", "VIIRS nl_center (incountry)", "viirs_nl_center_incountry.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

viirs_mask = np.ones(len(df), dtype=bool)
VIIRS_ZERO = float(df.loc[viirs_mask, "nl_mean"].min())

zeros_df = pd.DataFrame(
    {"VIIRS": [VIIRS_ZERO,
              df.loc[viirs_mask, "nl_center"].min(),
              df.loc[viirs_mask, "nl_mean"].min()]},
    index=["True 0", "min nl_center", "min nl_mean"]
)

with pd.option_context("display.precision", 9):
    nb_display(zeros_df)

viirs_zero_mask = viirs_mask & (df["nl_mean"].values == VIIRS_ZERO)

zeros_label_df = pd.DataFrame({"label_when_VIIRS_is_all_zero": df["wealthpooled"].values[viirs_zero_mask]})
nb_display(zeros_label_df.describe().T)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.hist(df["wealthpooled"].values[viirs_zero_mask], bins=50)
ax.set(xlabel="label", ylabel="count", title="Histogram of labels when VIIRS is all-0")
ax.grid(True)
savefig_dd("viirs_all_zero_hist.png")
plt.show()

df["zero_nl_viirs"] = viirs_zero_mask
zero_nl_counts = df.groupby("country")["zero_nl_viirs"].sum().astype(int)
zero_nl_frac   = zero_nl_counts / df.groupby("country").size()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
zero_nl_counts.plot.barh(width=0.8, ax=axs[0], grid=True)
zero_nl_frac.plot.barh(width=0.8, ax=axs[1], grid=True)
axs[0].set_xlabel("count")
axs[1].set_xlabel("fraction")
fig.suptitle("All-0 VIIRS Nightlights", y=1.02)
savefig_dd("viirs_all_zero_by_country.png")
plt.show()

if not MAKE_PANELS:
    print("MAKE_PANELS=False; skipping.")
else:
    LOC_DICT  = DD_DIR / "dhs_loc_dict.pkl"
    KEYMAP_PKL = DD_DIR / "tfrecord_key_to_path.pkl"
    if not (LOC_DICT.exists() and KEYMAP_PKL.exists()):
        raise FileNotFoundError("Missing LOC_DICT or KEYMAP_PKL. Run earlier cells first.")

    loc_dict   = pickle.loads(LOC_DICT.read_bytes())
    key_to_path = pickle.loads(KEYMAP_PKL.read_bytes())

    def fetch_image(path):
        for raw in tf.data.TFRecordDataset(path).take(1).as_numpy_iterator():
            ex = tf.train.Example.FromString(raw)
            f = ex.features.feature
            chans = []
            for b in BANDS_ALL:
                a = _as_arr(f, b)
                a = _center_crop(a, patch=PATCH, crop=CROP)
                chans.append(np.zeros((PIXELS_CROP,), np.float32) if a is None else a)
            return np.stack(chans, axis=-1).reshape(CROP, CROP, len(BANDS_ALL))
        return None

    def _save_and_embed(fig, save_name):
        out_path = DD_DIR / save_name
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print("Saved:", out_path)
        if _NBImage is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            buf.seek(0)
            _nbdisplay(_NBImage(data=buf.getvalue()))
        plt.close(fig)

    def plot_image_by_band(img, title, save_name):
        k = len(BAND_ORDER)
        fig, axes = plt.subplots(1, k, figsize=(1.9*k, 1.9))
        axes = np.atleast_1d(axes).ravel()
        for i, b in enumerate(BAND_ORDER):
            chan = img[:, :, BANDS_ALL.index(b)]
            axes[i].imshow(chan)
            axes[i].set_title(b, fontsize=8)
            axes[i].axis("off")
        fig.suptitle(title, fontsize=10)
        _save_and_embed(fig, save_name)

    rows = []
    for (cc, yr, clu), v in loc_dict.items():
        key = (str(cc).upper()[:2], int(yr), int(clu))
        path = key_to_path.get(key)
        if path is None:
            continue
        rows.append({
            "country": key[0], "year": key[1], "cluster": key[2],
            "lat": float(v["lat"]), "lon": float(v["lon"]),
            "wealthpooled": float(v["wealthpooled"]),
            "tf_path": path,
        })

    cluster_df = pd.DataFrame(rows)
    print("Clusters with imagery:", len(cluster_df))
    nb_display(cluster_df.head())

    labels = cluster_df["wealthpooled"].values

    top_idx = np.argsort(labels)[::-1][:K_PANELS]
    for rank, idx in enumerate(tqdm(top_idx, desc="Top wealth panels"), start=1):
        r = cluster_df.iloc[int(idx)]
        img = fetch_image(r["tf_path"])
        if img is None:
            continue
        title = (f"{rank}-th highest wealth: {r['wealthpooled']:.06f}, "
                 f"loc={r['country']} {int(r['year'])} ({r['lat']:.06f}, {r['lon']:.06f})")
        plot_image_by_band(img, title, f"panel_high_{rank}.png")

    low_idx = np.argsort(labels)[:K_PANELS]
    for rank, idx in enumerate(tqdm(low_idx, desc="Low wealth panels"), start=1):
        r = cluster_df.iloc[int(idx)]
        img = fetch_image(r["tf_path"])
        if img is None:
            continue
        title = (f"{rank}-th lowest wealth: {r['wealthpooled']:.06f}, "
                 f"loc={r['country']} {int(r['year'])} ({r['lat']:.06f}, {r['lon']:.06f})")
        plot_image_by_band(img, title, f"panel_low_{rank}.png")

import re, json, time
from pathlib import Path
from tqdm.auto import tqdm
import tensorflow as tf

TF_ROOT       = Path(r"C:\Users\d-rosseboe\Documents\Master\TFRecords")
PROCESSED_DIR = TF_ROOT / "processed"
MERGED_DIR    = TF_ROOT / "merged_rounds"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

OVERWRITE_OUTPUT = False
OUTPUT_GZIP      = False
DONE_SUFFIX_JSON = "__MERGE_META.json"

round_rx = re.compile(r"^([A-Z]{2})_(\d{4})$", re.IGNORECASE)

if not PROCESSED_DIR.exists():
    raise FileNotFoundError(f"Missing processed dir: {PROCESSED_DIR}")

round_dirs = [p for p in PROCESSED_DIR.iterdir() if p.is_dir() and round_rx.match(p.name)]
round_dirs = sorted(round_dirs, key=lambda p: (p.name[:2].upper(), int(p.name.split("_")[1])))

print(f"Found round folders: {len(round_dirs)} in {PROCESSED_DIR}")
print(f"Output folder: {MERGED_DIR}")
print(f"OUTPUT_GZIP={OUTPUT_GZIP} | OVERWRITE_OUTPUT={OVERWRITE_OUTPUT}")

tf_opts = tf.io.TFRecordOptions(compression_type="GZIP") if OUTPUT_GZIP else None

total_rounds_written = 0
total_records_written = 0

for rd in tqdm(round_dirs, desc="Merging rounds"):
    m = round_rx.match(rd.name)
    cc = m.group(1).upper()
    yr = int(m.group(2))

    src_files = sorted(rd.glob("*.tfrecord"))
    if not src_files:
        continue

    out_name = f"{cc}_{yr}.tfrecord" + (".gz" if OUTPUT_GZIP else "")
    out_path = MERGED_DIR / out_name
    meta_path = MERGED_DIR / f"{cc}_{yr}{DONE_SUFFIX_JSON}"
    tmp_path = MERGED_DIR / (out_name + ".tmp")

    if out_path.exists() and (not OVERWRITE_OUTPUT):

        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "country": cc, "year": yr,
                "status": "skipped_existing_output",
                "output_path": str(out_path),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2))
        continue

    n_records = 0
    bad_files = []

    try:
        if tmp_path.exists():
            tmp_path.unlink()

        writer = tf.io.TFRecordWriter(str(tmp_path), options=tf_opts) if tf_opts else tf.io.TFRecordWriter(str(tmp_path))

        for f in src_files:
            try:
                for rec in tf.data.TFRecordDataset(str(f)).as_numpy_iterator():
                    writer.write(rec)
                    n_records += 1
            except Exception as e:
                bad_files.append({"file": str(f), "error": str(e)})

        writer.close()

        if out_path.exists():
            out_path.unlink()
        tmp_path.replace(out_path)

        meta_path.write_text(json.dumps({
            "country": cc,
            "year": yr,
            "n_source_files": int(len(src_files)),
            "n_records_written": int(n_records),
            "n_bad_files": int(len(bad_files)),
            "bad_files": bad_files[:50],
            "output_path": str(out_path),
            "compression": "GZIP" if OUTPUT_GZIP else "NONE",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))

        total_rounds_written += 1
        total_records_written += n_records
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

print(f"Done. Wrote {total_rounds_written} merged files.")
print(f"Total records written: {total_records_written}")
print(f"Merged outputs in: {MERGED_DIR}")
