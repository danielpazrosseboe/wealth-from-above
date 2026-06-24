import subprocess
import sys
try:
    import pyproj
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyproj"])

import ee
import numpy as np
import pandas as pd
import os
import time
import random
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from google.colab import drive
from pyproj import Transformer
from tqdm.auto import tqdm

logging.getLogger("google_auth_httplib2").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

logging.getLogger('googleapiclient.http').setLevel(logging.ERROR)

warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

if not Path("/content/drive").exists():
    drive.mount("/content/drive")

try:
    ee.Initialize(project="master-thesis-measles", opt_url="https://earthengine-highvolume.googleapis.com")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="master-thesis-measles", opt_url="https://earthengine-highvolume.googleapis.com")

CSV_PATH   = "/content/drive/My Drive/Master_Thesis/Surveys/clusters_yeh_spec.csv"
OUTPUT_DIR = "/content/drive/My Drive/Master_Thesis/Dataset_Shards_Landsat_30m_K9_Overlap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_SIZE      = 3
TARGET_SIZE    = 224
SCALE          = 30
GRID_SPACING_M = 2500.0
JITTER_R_M     = 6000.0
BANDS = ["BLUE","GREEN","RED","NIR","SWIR1","SWIR2","TEMP1","NIGHTLIGHTS"]

THREADS    = 50
SHARD_SIZE = 1000

def make_offsets_3x3(spacing_m=GRID_SPACING_M):
    offs = []
    steps = [-1.0 * spacing_m, 0.0, 1.0 * spacing_m]
    for dx in steps:
        for dy in steps:
            offs.append((dx, dy))
    return offs

def get_retry(func, retries=3, delay=2.0):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(delay * (2 ** i) + random.random())

def year_window(yr: int):
    if 2017 <= yr <= 2019: return "2017-01-01", "2019-12-31"
    if 2020 <= yr <= 2022: return "2020-01-01", "2022-12-31"
    return "2023-01-01", "2025-12-31"

def mask_landsat_qa(img):
    qa = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 0).eq(0)        .And(qa.bitwiseAnd(1 << 1).eq(0))        .And(qa.bitwiseAnd(1 << 2).eq(0))        .And(qa.bitwiseAnd(1 << 3).eq(0))        .And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(mask)

def apply_scale_factors(img):
    opticalBands = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermalBand = img.select("ST_B10").multiply(0.00341802).add(149.0)
    return img.addBands(opticalBands, None, True)              .addBands(thermalBand, None, True)

def build_composite(roi_3857, start, end):

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    col = l8.merge(l9)

    def prep_image(img):

        img = mask_landsat_qa(img)

        img = apply_scale_factors(img)

        return img.select(
            ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"],
            ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "TEMP1"]
        )

    processed_col = col.filterBounds(roi_3857).filterDate(start, end).map(prep_image)

    s2 = processed_col.median()

    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterDate(start, end)
             .select(["avg_rad"], ["NIGHTLIGHTS"])
             .median()
             .resample("bilinear"))

    return s2.addBands(viirs).select(BANDS).toFloat()

def chip_request(img, patch_cx, patch_cy):
    half = (TARGET_SIZE * SCALE) / 2.0
    xmin, xmax = patch_cx - half, patch_cx + half
    ymin, ymax = patch_cy - half, patch_cy + half

    req = {
        "expression": img,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": TARGET_SIZE, "height": TARGET_SIZE},
            "affineTransform": {
                "scaleX": SCALE, "shearX": 0, "translateX": xmin,
                "shearY": 0, "scaleY": -SCALE, "translateY": ymax
            },
            "crsCode": "EPSG:3857"
        }
    }
    return req

def fetch_cluster_data(row):
    unique_id = "UNKNOWN"
    try:
        cc = str(row["country"]).strip().upper()[:2]
        yr = int(row["year"])
        clu = int(row["cluster"])
        lat, lon = float(row["lat"]), float(row["lon"])
        unique_id = f"{cc}_{yr}_{clu}"

        start, end = year_window(yr)

        local_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        cx, cy = local_transformer.transform(lon, lat)

        roi_padding = 7000.0

        roi = ee.Geometry.Rectangle(
            [cx - roi_padding, cy - roi_padding, cx + roi_padding, cy + roi_padding],
            "EPSG:3857",
            False
        )

        img = build_composite(roi, start, end)

        offsets = make_offsets_3x3()
        bag = []

        for dx, dy in offsets:
            pcx, pcy = cx + dx, cy + dy
            req = chip_request(img, pcx, pcy)
            raw = get_retry(lambda: ee.data.computePixels(req))
            arr = np.dstack([raw[b] for b in BANDS]).astype(np.float16)
            bag.append(arr)

        bag = np.stack(bag, axis=0)

        red = bag[..., BANDS.index("RED")]
        valid = np.isfinite(red) & (red != 0)

        if valid.mean() < 0.20:
            return None
        if np.nansum(bag) == 0:
            return None

        return unique_id, bag

    except Exception as e:
        tqdm.write(f"[ERR] {unique_id}: {e}")
        return None

print("--- STARTING FULL EXPORT (LANDSAT 30m + THERMAL, K=9) ---")
df = pd.read_csv(CSV_PATH)
if "cluster" not in df.columns and "cluster_index" in df.columns:
    df = df.rename(columns={"cluster_index": "cluster"})

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["country"] = df["country"].astype(str)
df = df.dropna(subset=["lat", "lon", "year", "country", "cluster"]).copy()
df = df.query("year >= 2017").copy()
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
df = df.dropna(subset=["cluster"]).copy()

chunks = [df.iloc[i:i + SHARD_SIZE] for i in range(0, len(df), SHARD_SIZE)]
print(f"Total Clusters: {len(df)}")
print(f"Total Shards:   {len(chunks)} (Size {SHARD_SIZE})")
print(f"Threads:        {THREADS}")

for si, chunk in enumerate(tqdm(chunks, desc="Total Progress", unit="shard")):
    shard_name = f"shard_{si:04d}.npz"
    shard_path = os.path.join(OUTPUT_DIR, shard_name)

    if os.path.exists(shard_path):
        tqdm.write(f"Skipping {shard_name} (Exists)")
        continue

    t0 = time.time()
    results_X = []
    results_ids = []

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(fetch_cluster_data, row) for _, row in chunk.iterrows()]

        for fut in tqdm(as_completed(futures), total=len(chunk), desc=f"Shard {si}", leave=False, unit="cluster"):
            res = fut.result()
            if res is not None:
                uid, x = res
                results_ids.append(uid)
                results_X.append(x)

    dt = time.time() - t0

    if results_X:
        X_shard = np.stack(results_X, axis=0)
        ids_shard = np.array(results_ids)
        np.savez_compressed(shard_path, X=X_shard, ids=ids_shard)
        tqdm.write(f"   -> Saved {shard_name} | {len(results_X)}/{len(chunk)} kept | {dt:.1f}s")
    else:
        tqdm.write(f"   [WARN] No valid data fetched for {shard_name}")

print("All Done.")
