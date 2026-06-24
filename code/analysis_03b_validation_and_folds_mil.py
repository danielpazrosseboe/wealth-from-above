FORCE_REBUILD = True
DISABLE_GPU   = False
MAKE_MAPS     = True
MAKE_PANELS   = True
K_PANELS      = 5

import os, sys
from pathlib import Path

def _pick_first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p.resolve()
    return None

HOME = Path.home()

PROJECT_ROOT = HOME / "Downloads" / "CNN MIL"
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

TF_ROOT = _pick_first_existing([
    PROJECT_ROOT / "TFRecords_MIL_CLEAN",
    PROJECT_ROOT / "TFRecords_MIL",
])

CLUSTERS_CSV = _pick_first_existing([
    PROJECT_ROOT / "clusters_yeh_spec.csv",
    HOME / "Downloads" / "clusters_yeh_spec.csv",
])

VALIDATION_DIR = PROJECT_ROOT / "Data Validation"
BAND_OUT_DIR   = VALIDATION_DIR / "Band Analysis"
DD_DIR         = PROJECT_ROOT / "Data Distribution"
NE_CACHE_DIR   = PROJECT_ROOT / "_cache" / "natural_earth"
AUDIT_DIR      = VALIDATION_DIR / "Round Audit"

for p in [VALIDATION_DIR, BAND_OUT_DIR, DD_DIR, NE_CACHE_DIR, AUDIT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MIL VALIDATION PIPELINE (CLEANED TFRECORDS)")
print("=" * 60)
print(f"TF_ROOT      : {TF_ROOT}")
print(f"CLUSTERS_CSV : {CLUSTERS_CSV}")
print(f"DD_DIR       : {DD_DIR}")

if TF_ROOT is None:
    raise FileNotFoundError("Could not find TFRecord directory")
if CLUSTERS_CSV is None:
    raise FileNotFoundError("Could not find clusters_yeh_spec.csv")

import math, json, time, pickle, zipfile, io, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.auto import tqdm

try:
    import geopandas as gpd
    import requests
    from sklearn.cluster import dbscan as _dbscan
    from matplotlib.lines import Line2D
except ImportError as e:
    print(f"Optional package missing: {e}")

try:
    from IPython.display import display as nb_display
    from IPython.display import Image as _NBImage, display as _nbdisplay
except:
    nb_display = print
    _NBImage = None
    _nbdisplay = print

if DISABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        tf.config.set_visible_devices([], "GPU")
    except:
        pass

plt.rcParams["figure.dpi"] = 120
print(f"TensorFlow version: {tf.__version__}")

TARGET_SIZE = 224
PATCH = TARGET_SIZE
CROP = TARGET_SIZE
K_TILES = 9
CENTER_PATCH_INDEX = 4
SCALE_M = 30
RADIUS_EARTH = 6356.7523

BANDS_MS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "TEMP1"]
BANDS_ALL = BANDS_MS + ["NIGHTLIGHTS"]
BAND_ORDER = BANDS_ALL[:]

FOLDS = ["A", "B", "C", "D", "E"]
SPLITS = ["train", "val", "test"]

GOOD_LIST_CSV = "good_clusters.csv"
BAD_LIST_CSV = "bad_clusters.csv"

def savefig_dd(name_or_path, dpi=200):
    p = Path(name_or_path)
    if not p.is_absolute():
        p = DD_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    print("Saved figure:", p)

print("Imports and constants loaded.")

tf_files = sorted(list(TF_ROOT.glob("*.tfrecord")))
print(f"Found {len(tf_files)} TFRecord files in {TF_ROOT}")

if len(tf_files) == 0:
    raise FileNotFoundError(f"No .tfrecord files found in {TF_ROOT}")

round_index = []
for tf_path in tf_files:
    m = re.match(r"([A-Z]{2})_(\d{4})_MIL\.tfrecord", tf_path.name, re.IGNORECASE)
    if m:
        cc = m.group(1).upper()
        yr = int(m.group(2))
        round_index.append((cc, yr, str(tf_path)))

print(f"Parsed {len(round_index)} country-year rounds")
for cc, yr, path in round_index[:5]:
    print(f"  {cc}_{yr}: {Path(path).name}")

import shutil
from collections import defaultdict

CLEAN_DIR = PROJECT_ROOT / "TFRecords_MIL_CLEAN"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

existing_clean = list(CLEAN_DIR.glob("*.tfrecord"))
need_cleaning = FORCE_REBUILD or (len(existing_clean) == 0)

if not need_cleaning:
    print(f" Clean records already found in: {CLEAN_DIR}")
    print("Skipping cleaning step.")

    TF_ROOT = CLEAN_DIR
    tf_files = sorted(list(TF_ROOT.glob("*.tfrecord")))

else:
    print(f"--- STARTING DATA CLEANING ---")
    print(f"Input Source : {TF_ROOT}")
    print(f"Clean Output : {CLEAN_DIR}")

    feature_description = {band: tf.io.VarLenFeature(tf.float32) for band in BANDS_ALL}

    def check_record_validity(raw_bytes):
        try:
            example = tf.io.parse_single_example(raw_bytes, feature_description)
            for band in BANDS_ALL:
                data = tf.sparse.to_dense(example[band]).numpy()

                if np.isnan(data).any(): return False, band, "NaN"

                if np.isinf(data).any(): return False, band, "Inf"

                if data.min() < -1000.0: return False, band, "Negative"
            return True, None, None
        except Exception as e:
            return False, "Parse Error", str(e)

    total_kept = 0
    total_dropped = 0
    global_stats = defaultdict(lambda: {"NaN": 0, "Inf": 0, "Negative": 0, "Other": 0})

    for input_file in tqdm(tf_files, desc="Cleaning Shards"):
        filename = input_file.name
        output_path = CLEAN_DIR / filename

        raw_dataset = tf.data.TFRecordDataset(str(input_file))

        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for raw_record in raw_dataset:
                is_valid, bad_band, reason = check_record_validity(raw_record.numpy())

                if is_valid:
                    writer.write(raw_record.numpy())
                    total_kept += 1
                else:
                    total_dropped += 1
                    if bad_band in BANDS_ALL:
                        global_stats[bad_band][reason] += 1
                    else:
                        global_stats["SYSTEM"]["Other"] += 1

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Total Kept    : {total_kept}")
    print(f"Total Dropped : {total_dropped}")

    if total_dropped > 0:
        print("-" * 55)
        print(f"{'BAND':<15} | {'NaN':<8} | {'Inf':<8} | {'Negative':<8} | {'TOTAL':<8}")
        print("-" * 55)
        sorted_bands = sorted(global_stats.items(), key=lambda x: sum(x[1].values()), reverse=True)
        for band, counts in sorted_bands:
            total = sum(counts.values())
            print(f"{band:<15} | {counts['NaN']:<8} | {counts['Inf']:<8} | {counts['Negative']:<8} | {total:<8}")
    else:
        print(" No corruptions found.")
    print("=" * 60)

    TF_ROOT = CLEAN_DIR
    tf_files = sorted(list(TF_ROOT.glob("*.tfrecord")))
    print(f" TF_ROOT updated to: {TF_ROOT}")

BAND_JSON = BAND_OUT_DIR / "band_stats_summary.json"
BAND_CSV = BAND_OUT_DIR / "band_stats_summary.csv"
PICKLE_OUT = DD_DIR / "tfrecord_scan_records.pkl"
NPZ_ARRAYS_OUT = DD_DIR / "tfrecord_scan_arrays.npz"
KEYMAP_PKL = DD_DIR / "tfrecord_key_to_path.pkl"
MANIFEST = DD_DIR / "scan_manifest.json"

need_scan = FORCE_REBUILD or not all([
    BAND_JSON.exists(), BAND_CSV.exists(),
    PICKLE_OUT.exists(), NPZ_ARRAYS_OUT.exists(),
    KEYMAP_PKL.exists(), MANIFEST.exists()
])

feature_description = {
    "country": tf.io.FixedLenFeature([], tf.string),
    "year": tf.io.FixedLenFeature([], tf.int64),
    "cluster": tf.io.FixedLenFeature([], tf.int64),
    "cluster_index": tf.io.FixedLenFeature([], tf.int64),
    "K": tf.io.FixedLenFeature([], tf.int64),
    "patch_size": tf.io.FixedLenFeature([], tf.int64),
}
for band in BANDS_ALL:
    feature_description[band] = tf.io.VarLenFeature(tf.float32)

def parse_tfrecord(raw_record):
    return tf.io.parse_single_example(raw_record, feature_description)

if not need_scan:
    print("Loading cached scan results...")
    band_stats_json = json.loads(BAND_JSON.read_text())
    scan_records = pickle.loads(PICKLE_OUT.read_bytes())
    npz_data = np.load(NPZ_ARRAYS_OUT)
    years_arr = npz_data["years"]
    nls_center_arr = npz_data["nls_center"]
    nls_mean_arr = npz_data["nls_mean"]
    key_to_path = pickle.loads(KEYMAP_PKL.read_bytes())
else:
    print(f"Scanning {len(tf_files)} TFRecord files...")

    band_aggs = {band: {"count": 0, "sum": 0.0, "sum_sq": 0.0,
                        "min": float("inf"), "max": float("-inf")}
                 for band in BANDS_ALL}

    scan_records = []
    key_to_path = {}
    cluster_stats = defaultdict(lambda: {"n": 0, "n_bad": 0})

    NL_INDEX = BANDS_ALL.index("NIGHTLIGHTS")
    RED_INDEX = BANDS_ALL.index("RED")

    total_records = 0
    record_idx_in_file = {}

    for tf_path in tqdm(tf_files, desc="Scanning TFRecords"):
        tf_path_str = str(tf_path)
        record_idx_in_file[tf_path_str] = 0

        dataset = tf.data.TFRecordDataset(tf_path_str)

        for raw_record in dataset:
            example = parse_tfrecord(raw_record)

            cc = example["country"].numpy().decode("utf-8").upper()[:2]
            yr = int(example["year"].numpy())
            clu = int(example["cluster"].numpy())
            K = int(example["K"].numpy())
            H = int(example["patch_size"].numpy())

            key = (cc, yr, clu)
            total_records += 1

            if key not in key_to_path:
                key_to_path[key] = {
                    "path": tf_path_str,
                    "index": record_idx_in_file[tf_path_str]
                }

            band_data = {}
            for band in BANDS_ALL:
                arr = tf.sparse.to_dense(example[band]).numpy()
                band_data[band] = arr

                valid = arr[np.isfinite(arr)]
                if len(valid) > 0:
                    band_aggs[band]["count"] += len(valid)
                    band_aggs[band]["sum"] += valid.sum()
                    band_aggs[band]["sum_sq"] += (valid ** 2).sum()
                    band_aggs[band]["min"] = min(band_aggs[band]["min"], valid.min())
                    band_aggs[band]["max"] = max(band_aggs[band]["max"], valid.max())

            nl_arr = band_data["NIGHTLIGHTS"]
            if nl_arr.size == K * H * H:
                nl_cube = nl_arr.reshape(K, H, H)
                center_patch = nl_cube[CENTER_PATCH_INDEX]
                nl_center = float(center_patch[H//2, H//2])
                nl_mean = float(np.nanmean(center_patch))
            else:
                nl_center = float("nan")
                nl_mean = float("nan")

            red_arr = band_data["RED"]
            if red_arr.size == K * H * H:
                red_cube = red_arr.reshape(K, H, H)
                valid_red = np.isfinite(red_cube) & (red_cube != 0)
                good_example = bool(valid_red.mean() >= 0.20)
            else:
                good_example = False

            cluster_stats[key]["n"] += 1
            if not good_example:
                cluster_stats[key]["n_bad"] += 1

            scan_records.append({
                "country": cc,
                "year": yr,
                "cluster": clu,
                "nls_center": nl_center,
                "nls_mean": nl_mean,
                "good_example": good_example,
                "tf_path": tf_path_str,
            })

            record_idx_in_file[tf_path_str] += 1

    print("\n" + "=" * 60)
    print("BAND STATISTICS")
    print("=" * 60)

    rows = []
    print(f"\n{'Band':<12} {'Count':>14} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 76)

    for band in BANDS_ALL:
        agg = band_aggs[band]
        n = agg["count"]

        if n > 0:
            mean = agg["sum"] / n
            variance = (agg["sum_sq"] / n) - (mean ** 2)
            variance = max(0.0, variance)
            std = np.sqrt(variance)
            bmin = agg["min"]
            bmax = agg["max"]
        else:
            mean = std = bmin = bmax = float("nan")

        rows.append({
            "band": band,
            "n_pixels": int(n),
            "global_mean": float(mean),
            "global_std": float(std),
            "min": float(bmin),
            "max": float(bmax),
        })

        print(f"{band:<12} {n:>14,} {mean:>12.6f} {std:>12.6f} {bmin:>12.6f} {bmax:>12.6f}")

    df_stats = pd.DataFrame(rows)

    band_stats_json = {
        r["band"]: {"global_mean": r["global_mean"], "global_std": r["global_std"]}
        for r in rows
    }
    BAND_JSON.write_text(json.dumps(band_stats_json, indent=2))
    df_stats.to_csv(BAND_CSV, index=False)

    df_rec = pd.DataFrame(scan_records)
    years_arr = df_rec["year"].values.astype(np.int32)
    nls_center_arr = df_rec["nls_center"].values.astype(np.float32)
    nls_mean_arr = df_rec["nls_mean"].values.astype(np.float32)

    PICKLE_OUT.write_bytes(pickle.dumps(scan_records))
    np.savez_compressed(NPZ_ARRAYS_OUT, years=years_arr, nls_center=nls_center_arr, nls_mean=nls_mean_arr)
    KEYMAP_PKL.write_bytes(pickle.dumps(key_to_path))

    good_keys = [k for k, st in cluster_stats.items() if st["n"] > 0 and st["n_bad"] == 0]
    bad_keys = [k for k, st in cluster_stats.items() if st["n"] > 0 and st["n_bad"] > 0]

    MANIFEST.write_text(json.dumps({
        "tf_root": str(TF_ROOT),
        "n_files": len(tf_files),
        "n_records": total_records,
        "n_unique_keys": len(cluster_stats),
        "n_good_keys": len(good_keys),
        "n_bad_keys": len(bad_keys),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

    print(f"\nSaved:")
    print(f"  - {BAND_JSON}")
    print(f"  - {BAND_CSV}")
    print(f"  - {PICKLE_OUT}")
    print(f"  - {KEYMAP_PKL}")
    print(f"  - {MANIFEST}")

print(f"\nTotal scan records: {len(scan_records)}")
print(f"Total unique keys: {len(key_to_path)}")

LOC_DICT_PATH = DD_DIR / "dhs_loc_dict.pkl"

df_loc = pd.read_csv(CLUSTERS_CSV, float_precision="high")

df_loc["country"] = df_loc["country"].astype(str).str.upper().str[:2]
df_loc["year"] = pd.to_numeric(df_loc["year"], errors="coerce").astype("Int64")

if "cluster" in df_loc.columns and "cluster_index" not in df_loc.columns:
    df_loc = df_loc.rename(columns={"cluster": "cluster_index"})
if "cluster_index" not in df_loc.columns:
    raise ValueError("clusters CSV must contain 'cluster' or 'cluster_index'")

df_loc["cluster_index"] = pd.to_numeric(df_loc["cluster_index"], errors="coerce").astype("Int64")

if "n_households" not in df_loc.columns and "households" in df_loc.columns:
    df_loc = df_loc.rename(columns={"households": "n_households"})

required = ["country", "year", "cluster_index", "wealthpooled", "lat", "lon"]
missing = [c for c in required if c not in df_loc.columns]
if missing:
    raise ValueError(f"clusters CSV missing columns: {missing}")

df_loc["wealthpooled"] = pd.to_numeric(df_loc["wealthpooled"], errors="coerce")
df_loc["lat"] = pd.to_numeric(df_loc["lat"], errors="coerce")
df_loc["lon"] = pd.to_numeric(df_loc["lon"], errors="coerce")

if "n_households" in df_loc.columns:
    df_loc["n_households"] = pd.to_numeric(df_loc["n_households"], errors="coerce").fillna(0).astype(int)
else:
    df_loc["n_households"] = 0

if "urban" in df_loc.columns:
    if df_loc["urban"].dtype == object:
        df_loc["urban"] = (df_loc["urban"].astype(str).str.upper() == "U")
    else:
        df_loc["urban"] = (pd.to_numeric(df_loc["urban"], errors="coerce").fillna(0).astype(int) == 1)
else:
    df_loc["urban"] = False

df_loc = df_loc.dropna(subset=["country", "year", "cluster_index", "lat", "lon"]).copy()
df_loc = df_loc.drop_duplicates(subset=["country", "year", "cluster_index"], keep="first")

print(f"Clusters in CSV after cleaning: {len(df_loc)}")

tfrecord_keys = set(key_to_path.keys())
print(f"Unique keys in TFRecords: {len(tfrecord_keys)}")

loc_dict = {}
skipped = 0

for _, r in df_loc.iterrows():
    key = (str(r["country"]), int(r["year"]), int(r["cluster_index"]))

    if key not in tfrecord_keys:
        skipped += 1
        continue

    loc_dict[key] = {
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "country": str(r["country"]),
        "year": int(r["year"]),
        "cluster": int(r["cluster_index"]),
        "households": int(r["n_households"]),
        "urban": bool(r["urban"]),
        "wealthpooled": float(r["wealthpooled"]),
    }

LOC_DICT_PATH.write_bytes(pickle.dumps(loc_dict))

print(f"\nSaved: {LOC_DICT_PATH}")
print(f"loc_dict entries: {len(loc_dict)}")
print(f"Skipped (not in TFRecords): {skipped}")

OUT_CSV = DD_DIR / "dhs_combined_df.csv"

rows = []
missing_loc = 0

for rec in scan_records:
    key = (rec["country"], rec["year"], rec["cluster"])

    if key not in loc_dict:
        missing_loc += 1
        continue

    loc = loc_dict[key]

    rows.append({
        "country": rec["country"],
        "year": rec["year"],
        "cluster": rec["cluster"],
        "cluster_index": rec["cluster"],
        "lat": loc["lat"],
        "lon": loc["lon"],
        "wealthpooled": loc["wealthpooled"],
        "urban": int(loc["urban"]),
        "households": loc["households"],
        "nl_center": rec["nls_center"],
        "nl_mean": rec["nls_mean"],
        "good_example": rec["good_example"],
    })

df = pd.DataFrame(rows)

print(f"Combined dataframe: {len(df)} rows")
print(f"Missing location info: {missing_loc}")

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

nb_display(df.head())
nb_display(df.describe())

LOCS = df[["lat", "lon"]].to_numpy(dtype=np.float32)
print(f"\nLOCS shape: {LOCS.shape}")

OOC_FOLDS_PATH = DD_DIR / "dhs_ooc_folds.pkl"

need_ooc = FORCE_REBUILD or not OOC_FOLDS_PATH.exists()

if need_ooc:
    print("Creating OOC folds...")

    COUNTRIES = np.array(sorted(df["country"].unique()))
    print(f"Countries: {len(COUNTRIES)} -> {list(COUNTRIES)}")

    country_indices = defaultdict(list)
    for i in range(len(df)):
        country_indices[df["country"].iloc[i]].append(i)

    for c in country_indices:
        country_indices[c] = np.asarray(country_indices[c], dtype=np.int64)

    def make_ooc_folds_balanced(country_indices, folds=FOLDS):
        order = sorted(country_indices.items(), key=lambda kv: len(kv[1]), reverse=True)

        fold_countries = {f: [] for f in folds}
        for i, (cc, _) in enumerate(order):
            fold_countries[folds[i % len(folds)]].append(cc)

        ooc = {f: {s: None for s in SPLITS} for f in folds}

        for f in folds:
            test_countries = fold_countries[f]
            val_countries = fold_countries[folds[(folds.index(f) + 1) % len(folds)]]
            train_countries = [c for c in COUNTRIES if c not in set(test_countries + val_countries)]

            test_idx = np.sort(np.concatenate([country_indices[c] for c in test_countries])) if test_countries else np.array([], dtype=np.int64)
            val_idx = np.sort(np.concatenate([country_indices[c] for c in val_countries])) if val_countries else np.array([], dtype=np.int64)
            train_idx = np.sort(np.concatenate([country_indices[c] for c in train_countries])) if train_countries else np.array([], dtype=np.int64)

            ooc[f]["test"] = test_idx
            ooc[f]["val"] = val_idx
            ooc[f]["train"] = train_idx

        return ooc

    ooc_folds = make_ooc_folds_balanced(country_indices)
    OOC_FOLDS_PATH.write_bytes(pickle.dumps(ooc_folds))
    print(f"Saved: {OOC_FOLDS_PATH}")
else:
    ooc_folds = pickle.loads(OOC_FOLDS_PATH.read_bytes())
    print(f"Loaded: {OOC_FOLDS_PATH}")

print("\nOOC Fold Summary:")
print(f"{'Fold':<6} {'Train':>8} {'Val':>8} {'Test':>8}")
print("-" * 34)
for f in FOLDS:
    print(f"{f:<6} {len(ooc_folds[f]['train']):>8} {len(ooc_folds[f]['val']):>8} {len(ooc_folds[f]['test']):>8}")

INC_FOLDS_PATH = DD_DIR / "dhs_incountry_folds.pkl"

need_inc = FORCE_REBUILD or not INC_FOLDS_PATH.exists()

def get_lat_for_distance_km(d_km):
    return (d_km / RADIUS_EARTH) * 180.0 / np.pi

def get_lon_for_distance_km(lat_deg, d_km):
    r = RADIUS_EARTH * np.cos(np.abs(lat_deg) * np.pi / 180.0)
    return (d_km / r) * 180.0 / np.pi

def create_folds_dbscan(locs, min_dist, fold_names):
    locs = np.asarray(locs, dtype=float)

    locs_to_indices = defaultdict(list)
    for i, loc in enumerate(locs):
        locs_to_indices[tuple(loc)].append(i)

    uniq = np.unique(locs, axis=0)
    _, labels = _dbscan(X=uniq, eps=min_dist, min_samples=2, metric="euclidean")

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
    print("Creating in-country folds...")

    side_km = CROP * SCALE_M / 1000.0
    far_lat = float(max(abs(LOCS[:, 0].min()), abs(LOCS[:, 0].max())))
    side_lat = get_lat_for_distance_km(side_km)
    side_lon = get_lon_for_distance_km(far_lat, side_km)
    MIN_DIST = float(math.hypot(side_lat, side_lon))

    print(f"Patch side: {side_km:.2f} km")
    print(f"Min distance threshold: {MIN_DIST:.6f} degrees")

    test_folds = create_folds_dbscan(LOCS, min_dist=MIN_DIST, fold_names=FOLDS)

    incountry_folds = {}
    for i, f in enumerate(FOLDS):
        incountry_folds[f] = {}
        incountry_folds[f]["test"] = np.asarray(test_folds[f], dtype=np.int64)
        val_f = FOLDS[(i + 1) % len(FOLDS)]
        incountry_folds[f]["val"] = np.asarray(test_folds[val_f], dtype=np.int64)
        train_fs = [FOLDS[(i + 2) % 5], FOLDS[(i + 3) % 5], FOLDS[(i + 4) % 5]]
        incountry_folds[f]["train"] = np.sort(np.concatenate([
            np.asarray(test_folds[x], dtype=np.int64) for x in train_fs
        ]))

    INC_FOLDS_PATH.write_bytes(pickle.dumps(incountry_folds))
    print(f"Saved: {INC_FOLDS_PATH}")
else:
    incountry_folds = pickle.loads(INC_FOLDS_PATH.read_bytes())
    print(f"Loaded: {INC_FOLDS_PATH}")

print("\nIn-Country Fold Summary:")
print(f"{'Fold':<6} {'Train':>8} {'Val':>8} {'Test':>8}")
print("-" * 34)
for f in FOLDS:
    print(f"{f:<6} {len(incountry_folds[f]['train']):>8} {len(incountry_folds[f]['val']):>8} {len(incountry_folds[f]['test']):>8}")

print("Writing audit lists per country-year...")

cl = pd.read_csv(CLUSTERS_CSV, float_precision="high")
cl["country"] = cl["country"].astype(str).str.upper().str[:2]
cl["year"] = pd.to_numeric(cl["year"], errors="coerce").astype("Int64")

if "cluster_index" not in cl.columns and "cluster" in cl.columns:
    cl = cl.rename(columns={"cluster": "cluster_index"})
cl["cluster_index"] = pd.to_numeric(cl["cluster_index"], errors="coerce").astype("Int64")
cl = cl.dropna(subset=["country", "year", "cluster_index"]).copy()

expected_by_round = {
    k: set(g["cluster_index"].astype(int).tolist())
    for k, g in cl.groupby(["country", "year"])
}

observed_by_round = defaultdict(set)
for rec in scan_records:
    key_round = (rec["country"], rec["year"])
    observed_by_round[key_round].add(rec["cluster"])

written = 0
for (cc, yr), expected in tqdm(expected_by_round.items(), desc="Writing audit files"):
    out_dir = AUDIT_DIR / f"{cc}_{yr}"
    out_dir.mkdir(parents=True, exist_ok=True)

    observed = observed_by_round.get((cc, yr), set())
    missing = sorted(expected - observed)
    good = sorted(expected & observed)

    pd.DataFrame({"cluster_index": good}).to_csv(out_dir / GOOD_LIST_CSV, index=False)
    pd.DataFrame([
        {"cluster_index": int(ci), "reason": "missing_in_tfrecords", "n_bad_examples": 1}
        for ci in missing
    ]).to_csv(out_dir / BAD_LIST_CSV, index=False)

    written += 1

print(f"Wrote audit files for {written} country-year combinations")

try:
    NE110_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(NE110_URL)
    africa = world[world["CONTINENT"] == "Africa"].to_crs("EPSG:4326")

    FOLD_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#d62728", "D": "#2ca02c", "E": "#9467bd"}

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#a6bddb")
    africa.plot(ax=ax, color="#fff6d5", edgecolor="black", linewidth=0.5)

    for f in FOLDS:
        idx = np.asarray(ooc_folds[f]["test"], dtype=int)
        if idx.size > 0:
            ax.scatter(LOCS[idx, 1], LOCS[idx, 0], s=6, c=FOLD_COLORS[f], label=f, alpha=0.85)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=FOLD_COLORS[f],
                      markersize=8, linestyle="") for f in FOLDS]
    ax.legend(handles, FOLDS, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xlim(-25, 55)
    ax.set_ylim(-36, 38)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("DHS Out-of-Country TEST Folds (A–E)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    savefig_dd("africa_ooc_test_folds.png")
    plt.show()
except Exception as e:
    print(f"Could not create OOC map: {e}")

try:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#a6bddb")
    africa.plot(ax=ax, color="#fff6d5", edgecolor="black", linewidth=0.5)

    for f in FOLDS:
        idx = np.asarray(incountry_folds[f]["test"], dtype=int)
        if idx.size > 0:
            ax.scatter(LOCS[idx, 1], LOCS[idx, 0], s=6, c=FOLD_COLORS[f], label=f, alpha=0.85)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=FOLD_COLORS[f],
                      markersize=8, linestyle="") for f in FOLDS]
    ax.legend(handles, FOLDS, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xlim(-25, 55)
    ax.set_ylim(-36, 38)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("DHS In-Country TEST Folds (A–E)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    savefig_dd("africa_incountry_test_folds.png")
    plt.show()
except Exception as e:
    print(f"Could not create in-country map: {e}")

from pandas.errors import EmptyDataError

MAX_MAPS = None

if not MAKE_MAPS:
    print("MAKE_MAPS=False; skipping.")
else:

    cl_all = pd.read_csv(CLUSTERS_CSV, float_precision="high")
    cl_all["country"] = cl_all["country"].astype(str).str.upper().str[:2]
    cl_all["year"] = pd.to_numeric(cl_all["year"], errors="coerce").astype("Int64")

    if "cluster_index" not in cl_all.columns and "cluster" in cl_all.columns:
        cl_all = cl_all.rename(columns={"cluster": "cluster_index"})

    cl_all["cluster_index"] = pd.to_numeric(cl_all["cluster_index"], errors="coerce").astype("Int64")
    cl_all["lat"] = pd.to_numeric(cl_all["lat"], errors="coerce")
    cl_all["lon"] = pd.to_numeric(cl_all["lon"], errors="coerce")
    cl_all["wealthpooled"] = pd.to_numeric(cl_all["wealthpooled"], errors="coerce")
    cl_all = cl_all.dropna(subset=["country", "year", "cluster_index", "lat", "lon", "wealthpooled"]).copy()
    cl_all["year"] = cl_all["year"].astype(int)
    cl_all["cluster_index"] = cl_all["cluster_index"].astype(int)

    def read_cluster_list(path):
        if not path.exists():
            return set()
        try:
            df = pd.read_csv(path)
            if "cluster_index" not in df.columns:
                return set()
            return set(pd.to_numeric(df["cluster_index"], errors="coerce").dropna().astype(int).tolist())
        except:
            return set()

    def read_bad_reasons(path):
        cols = ["reason", "n_clusters", "n_bad_examples_sum"]
        if not path.exists():
            return pd.DataFrame(columns=cols)
        try:
            df = pd.read_csv(path)
            if "reason" not in df.columns or "cluster_index" not in df.columns:
                return pd.DataFrame(columns=cols)
            df["n_bad_examples"] = df.get("n_bad_examples", 1)
            agg = df.groupby("reason", as_index=False).agg(
                n_clusters=("cluster_index", "nunique"),
                n_bad_examples_sum=("n_bad_examples", "sum")
            ).sort_values("n_clusters", ascending=False)
            return agg
        except:
            return pd.DataFrame(columns=cols)

    ISO_OVERRIDES = {"LB": "LR", "MD": "MG"}
    NE10_DIR = NE_CACHE_DIR / "ne10"
    NE10_ZIP = NE_CACHE_DIR / "ne10.zip"
    NE10_URL = "https://github.com/nvkelso/natural-earth-vector/archive/refs/heads/master.zip"

    def ensure_ne10():
        if NE10_DIR.exists():
            return
        print("Downloading Natural Earth 10m data...")
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
        raise FileNotFoundError("ne_10m_admin_0_countries.shp not found")

    try:
        WORLD10 = load_admin0_10m()
        ISO_A2_COL = next((c for c in WORLD10.columns if str(c).upper().startswith("ISO_A2")), None)
    except:
        WORLD10, ISO_A2_COL = None, None

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
        bad = set(map(int, bad_set))
        missing = all_clusters - good - bad
        combined_bad = bad | missing

        n_total = len(all_clusters)
        n_good = len(good & all_clusters)
        n_bad = len(combined_bad & all_clusters)
        pct_good = (100.0 * n_good / n_total) if n_total else 0.0
        pct_bad = (100.0 * n_bad / n_total) if n_total else 0.0

        good_df = dfp[dfp["cluster_index"].isin(good)].copy()
        bad_df = dfp[dfp["cluster_index"].isin(combined_bad)].copy()

        fig, ax = plt.subplots(figsize=(7, 7))

        geom = get_country_geom(country)
        if geom is not None:
            geom.boundary.plot(ax=ax, linewidth=0.8)
            geom.plot(ax=ax, alpha=0.05)

        sc = None
        if len(good_df) > 0:
            vmin = float(good_df["wealthpooled"].quantile(0.01))
            vmax = float(good_df["wealthpooled"].quantile(0.99))
            sc = ax.scatter(good_df["lon"], good_df["lat"], c=good_df["wealthpooled"],
                           s=25, cmap="viridis", vmin=vmin, vmax=vmax, label="accepted")

        if len(bad_df) > 0:
            ax.scatter(bad_df["lon"], bad_df["lat"], s=30, facecolors="none",
                      edgecolors="red", linewidths=1.0, label="rejected/missing")

        if sc is not None:
            fig.colorbar(sc, ax=ax).set_label("wealthpooled")

        title = f"{country} {year} — Good: {n_good}/{n_total} ({pct_good:.1f}%), Bad: {n_bad}/{n_total} ({pct_bad:.1f}%)"
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", "box")
        ax.legend()

        if bad_reasons_tbl is not None and len(bad_reasons_tbl):
            show = bad_reasons_tbl.head(8)
            tbl = ax.table(
                cellText=show[["reason", "n_clusters", "n_bad_examples_sum"]].values.tolist(),
                colLabels=["reason", "n_clusters", "n_bad_examples_sum"],
                loc="upper center", bbox=[0.0, 1.02, 1.0, 0.30])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)

        plt.tight_layout()
        plt.show()

    to_plot = []
    for cc, yr, _ in round_index:
        sdf = cl_all[(cl_all["country"] == cc) & (cl_all["year"] == yr)].copy()
        if sdf.empty:
            continue

        audit_dir = AUDIT_DIR / f"{cc}_{yr}"
        good_set = read_cluster_list(audit_dir / GOOD_LIST_CSV)
        bad_set = read_cluster_list(audit_dir / BAD_LIST_CSV)
        reasons = read_bad_reasons(audit_dir / BAD_LIST_CSV)

        to_plot.append((cc, yr, sdf, good_set, bad_set, reasons))

    print(f"Plotting maps for {len(to_plot)} country-year combinations")

    n_done = 0
    for (cc, yr, sdf, good_set, bad_set, reasons) in tqdm(to_plot, desc="Plotting maps"):
        plot_country_year_map(cc, yr, sdf, good_set, bad_set, bad_reasons_tbl=reasons)
        n_done += 1
        if MAX_MAPS is not None and n_done >= MAX_MAPS:
            break

def boxplot_df(df_in, col, group_cols, fig_name, figsize=(10, 6), ylabel=None, title=None):
    grouped = df_in.groupby(group_cols)[col].apply(list)
    labels = [f"{g}" for g in grouped.index]
    data = [v for v in grouped.values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel or col)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y")
    savefig_dd(fig_name)
    plt.show()

viirs_mask = np.ones(len(df), dtype=bool)

boxplot_df(df[viirs_mask], "nl_center", ["country", "year"],
           fig_name="viirs_center_by_country_year.png",
           ylabel="nl_center", title="Center VIIRS distribution by country/year")

boxplot_df(df[viirs_mask], "nl_mean", ["country", "year"],
           fig_name="viirs_mean_by_country_year.png",
           ylabel="nl_mean", title="Mean VIIRS distribution by country/year")

DMSP_VIIRS_YEAR = 2012

def plot_nl_by_fold_viirs(df_in, folds, col, title, fig_name):
    fig, axs = plt.subplots(1, 5, sharey=True, figsize=(11, 2.8))
    bin_edges = np.linspace(df_in[col].min() - 0.1, df_in[col].max() + 0.1, 100)
    centers = np.convolve(bin_edges, [.5, .5], mode="valid")

    for f, ax in zip(FOLDS, axs.flat):
        for split in SPLITS:
            idx = folds[f][split]
            sub = df_in.iloc[idx]
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
            sub = df_in.iloc[idx]
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

plot_nl_by_fold_viirs(df, ooc_folds, "nl_mean", "VIIRS nl_mean (OOC)", "viirs_nl_mean_ooc.png")
plot_nl_by_fold_viirs(df, incountry_folds, "nl_mean", "VIIRS nl_mean (incountry)", "viirs_nl_mean_incountry.png")
nl_boxplots_by_fold_viirs(df, ooc_folds, "nl_center", "VIIRS nl_center (OOC)", "viirs_nl_center_ooc.png")
nl_boxplots_by_fold_viirs(df, incountry_folds, "nl_center", "VIIRS nl_center (incountry)", "viirs_nl_center_incountry.png")

DD_DIR.mkdir(parents=True, exist_ok=True)
if not MAKE_PANELS:
    print("MAKE_PANELS=False; skipping.")
else:

    LOC_DICT_PATH = DD_DIR / "loc_dict.pkl"
    KEYMAP_PKL = DD_DIR / "tfrecord_key_to_path.pkl"

    if not (LOC_DICT_PATH.exists() and KEYMAP_PKL.exists()):
        print(" Missing LOC_DICT or KEYMAP. Skipping panels.")
        print(f"Check {LOC_DICT_PATH} and {KEYMAP_PKL}")
    else:
        print("Loading location data and key map...")
        loc_dict = pickle.loads(LOC_DICT_PATH.read_bytes())
        key_to_path = pickle.loads(KEYMAP_PKL.read_bytes())

        rows = []
        for (cc, yr, clu), v in loc_dict.items():
            rows.append({
                "country": str(cc).upper()[:2],
                "year": int(yr),
                "cluster": int(clu),
                "lat": float(v["lat"]),
                "lon": float(v["lon"]),
                "wealthpooled": float(v["wealthpooled"]),
            })

        cluster_df = pd.DataFrame(rows).drop_duplicates(
            subset=["country", "year", "cluster"]
        ).reset_index(drop=True)
        print(f"Clusters in loc_dict: {len(cluster_df)}")

        def fetch_cluster_image_robust(key):
            target_cc, target_yr, target_clu = key

            entry = key_to_path.get(key)
            if entry is None:
                print(f"Key {key} not found in key_to_path map.")
                return None

            tf_path = entry.get("path")
            if not tf_path:

                if isinstance(entry, str):
                    tf_path = entry
                else:
                    print(f"Invalid entry structure for {key}: {entry}")
                    return None

            target_idx = entry.get("index")

            dataset = tf.data.TFRecordDataset(tf_path)

            found_img = None

            for i, raw_record in enumerate(dataset):

                if target_idx is not None:
                    if i != target_idx:
                        continue

                example = parse_tfrecord(raw_record)

                if target_idx is None:
                    rec_cc = example["country"].numpy().decode("utf-8").upper()[:2]
                    rec_yr = int(example["year"].numpy())
                    rec_clu = int(example["cluster"].numpy())

                    if (rec_cc, rec_yr, rec_clu) != key:
                        continue

                K = int(example["K"].numpy())
                H = int(example["patch_size"].numpy())

                img = np.zeros((H, H, len(BANDS_ALL)), dtype=np.float32)
                for band_idx, band in enumerate(BANDS_ALL):
                    arr = tf.sparse.to_dense(example[band]).numpy()
                    if arr.size == K * H * H:
                        cube = arr.reshape(K, H, H)

                        center_idx = K // 2
                        img[:, :, band_idx] = cube[center_idx]

                found_img = img
                break

            return found_img

        def _save_and_embed(fig, save_name):
            out_path = DD_DIR / save_name
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            print("Saved panel:", out_path.name)
            try:
                if _NBImage is not None:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
                    buf.seek(0)
                    _nbdisplay(_NBImage(data=buf.getvalue()))
            except:
                pass
            plt.close(fig)

        def plot_image_by_band(img, title, save_name):
            if img is None: return
            k = len(BAND_ORDER)
            fig, axes = plt.subplots(1, k, figsize=(1.9 * k, 1.9))
            axes = np.atleast_1d(axes).ravel()

            for i, b in enumerate(BAND_ORDER):
                try:
                    idx = BANDS_ALL.index(b)
                    chan = img[:, :, idx]

                    vmin, vmax = np.percentile(chan[np.isfinite(chan)], [2, 98])
                    axes[i].imshow(chan, vmin=vmin, vmax=vmax, cmap='viridis' if b=='NIGHTLIGHTS' else 'gray')
                except:
                    pass

                axes[i].set_title(b, fontsize=8)
                axes[i].axis("off")

            fig.suptitle(title, fontsize=10)
            _save_and_embed(fig, save_name)

        labels = cluster_df["wealthpooled"].values

        top_idx = np.argsort(labels)[::-1][:K_PANELS]

        low_idx = np.argsort(labels)[:K_PANELS]

        print(f"\nGeneraring {K_PANELS} Top Wealth Panels...")
        for rank, idx in enumerate(tqdm(top_idx, desc="Top Wealth"), start=1):
            r = cluster_df.iloc[int(idx)]
            key = (r["country"], int(r["year"]), int(r["cluster"]))

            img = fetch_cluster_image_robust(key)
            if img is not None:
                title = (f"#{rank} Rich: {r['wealthpooled']:.2f} | {r['country']} {int(r['year'])}")
                plot_image_by_band(img, title, f"panel_high_{rank}.png")

        print(f"\nGeneraring {K_PANELS} Low Wealth Panels...")
        for rank, idx in enumerate(tqdm(low_idx, desc="Low Wealth"), start=1):
            r = cluster_df.iloc[int(idx)]
            key = (r["country"], int(r["year"]), int(r["cluster"]))

            img = fetch_cluster_image_robust(key)
            if img is not None:
                title = (f"#{rank} Poor: {r['wealthpooled']:.2f} | {r['country']} {int(r['year'])}")
                plot_image_by_band(img, title, f"panel_low_{rank}.png")

print("\nDone.")
