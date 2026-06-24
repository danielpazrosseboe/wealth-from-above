from google.colab import auth
auth.authenticate_user()

import math, time
from typing import Tuple
import pandas as pd
from tqdm.auto import tqdm
import ee

ee.Authenticate()
ee.Initialize(project='master-thesis-measles')

CSV_PATH     = "/content/drive/My Drive/Master_Thesis/Surveys/clusters_yeh_spec.csv"
SCALE        = 30
EXPORT_TILE_RADIUS = 127
CHUNK_SIZE   = 10
DRIVE_FOLDER = "TFRecords"
BANDS = ['BLUE','GREEN','RED','NIR','SWIR1','SWIR2','TEMP1','LON','LAT','NIGHTLIGHTS']

def three_year_window(y: int) -> Tuple[str, str]:
    if 2017 <= y <= 2019:
        return "2017-01-01", "2019-12-31"
    elif 2020 <= y <= 2022:
        return "2020-01-01", "2022-12-31"
    elif 2023 <= y <= 2025:
        return "2023-01-01", "2025-12-31"
    else:
        raise ValueError(f"Unsupported survey year for three-year window: {y}")

def mask_c2_qa(img):
    qa = img.select('QA_PIXEL')
    keep = (qa.bitwiseAnd(1 << 3).eq(0)
            .And(qa.bitwiseAnd(1 << 4).eq(0))
            .And(qa.bitwiseAnd(1 << 5).eq(0))
            .And(qa.bitwiseAnd(1 << 7).eq(0)))
    return img.updateMask(keep)

def landsat_stack(roi, start, end):
    sel = ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10']
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(roi).filterDate(start, end)
          .map(mask_c2_qa).select(sel))
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterBounds(roi).filterDate(start, end)
          .map(mask_c2_qa).select(sel))
    return (l8.merge(l9)).median().rename(
        ['BLUE','GREEN','RED','NIR','SWIR1','SWIR2','TEMP1']
    )

def composite_viirs(start, end):
    return (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
            .filterDate(start, end).median()
            .select(['avg_rad'], ['NIGHTLIGHTS']))

def add_lonlat(img):
    latlon = ee.Image.pixelLonLat().select(['longitude','latitude'], ['LON','LAT'])
    return img.addBands(latlon)

def df_to_features(df):
    feats = []
    for _, r in df.iterrows():
        geom = ee.Geometry.Point([float(r['lon']), float(r['lat'])])
        feats.append(ee.Feature(geom, {
            'country':       str(r['country']),
            'year':          int(r['year']),
            'cluster_index': int(r['cluster']),
            'lat':           float(r['lat']),
            'lon':           float(r['lon']),
            'wealthpooled':  float(r['wealthpooled']),
            'households':    float(r.get('n_households', 1.0)),
            'urban_rural':   float(r.get('urban', float('nan'))),
        }))
    return ee.FeatureCollection(feats)

def make_array_image(img, radius_px):
    kern = ee.Kernel.square(radius=radius_px, units='pixels')
    arr_imgs = [img.select([b]).neighborhoodToArray(kern).rename(b) for b in BANDS]
    return ee.Image.cat(arr_imgs)

def export_chunk(yr, base_img, points_fc, fname):
    start, end = three_year_window(yr)
    img = add_lonlat(base_img).addBands(composite_viirs(start, end)).select(BANDS)
    arr_img = make_array_image(img, EXPORT_TILE_RADIUS)

    def _sample_point(f):
        s = (arr_img.sample(region=f.geometry(),
                            scale=SCALE,
                            projection='EPSG:3857',
                            numPixels=1,
                            dropNulls=False,
                            tileScale=12).first())
        out = ee.Feature(None).copyProperties(
            f,
            ['country','year','cluster_index','lat','lon','wealthpooled',
             'households','urban_rural']
        )
        return out.setMulti(s.toDictionary(ee.List(BANDS)))

    samples = points_fc.map(_sample_point)

    selectors = ['country','year','cluster_index','lat','lon','wealthpooled',
                 'households','urban_rural'] + BANDS

    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=fname,
        folder=DRIVE_FOLDER,
        fileNamePrefix=fname,
        fileFormat='TFRecord',
        selectors=selectors
    )
    task.start()
    return task

def run_exports():
    df = pd.read_csv(CSV_PATH)
    need = {'country','year','cluster','lat','lon','wealthpooled'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{CSV_PATH} missing columns: {sorted(missing)}")

    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['lat','lon','year','cluster','wealthpooled'])

    df = df.loc[df['year'] >= 2017].copy()
    if df.empty:
        raise SystemExit("No rows with year >= 2017")

    groups = list(df.groupby(['country','year']))
    n_tasks = 0

    for (cc, yr), g in tqdm(groups, desc="Exporting (country,year)"):
        yr = int(yr)

        start, end = three_year_window(yr)

        g = g.reset_index(drop=True)
        roi = ee.Geometry.MultiPoint(g[['lon','lat']].values.tolist())

        base = landsat_stack(roi, start, end)

        n = len(g)
        n_chunks = math.ceil(n / CHUNK_SIZE)
        for i in range(n_chunks):
            sl = slice(i*CHUNK_SIZE, min((i+1)*CHUNK_SIZE, n))
            fc = df_to_features(g.iloc[sl].copy())
            fname = f"{cc}_{yr}_{i:02d}"
            export_chunk(yr, base, fc, fname)
            n_tasks += 1
            time.sleep(0.25)

    print(f"Started {n_tasks} export tasks to Drive folder '{DRIVE_FOLDER}'.")
    print("Monitor progress in the Earth Engine Tasks tab.")

run_exports()

import os
import glob
import shutil
import time
import gc
import random
import tensorflow as tf

SRC_DIR = "/content/drive/My Drive/TFRecords"
TMP_DIR = "/tmp/tfrecord_unzip"
os.makedirs(TMP_DIR, exist_ok=True)

CHUNK_SIZE = 10
SLEEP_BETWEEN_CHUNKS = 15
SLEEP_EACH_FILE = 1.0
MAX_RETRIES_PER_FILE = 5

def list_pending_gz(src_dir):
    gz = sorted(glob.glob(os.path.join(src_dir, "*.tfrecord.gz")))
    pending = []
    for gz_path in gz:
        out_drive = gz_path.replace(".tfrecord.gz", ".tfrecord")
        if not os.path.exists(out_drive):
            pending.append(gz_path)
    return pending

def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False

def unzip_one(gz_path):
    base_gz = os.path.basename(gz_path)
    out_drive = gz_path.replace(".tfrecord.gz", ".tfrecord")
    base_out = os.path.basename(out_drive)

    tmp_gz = os.path.join(TMP_DIR, base_gz)
    tmp_out = tmp_gz.replace(".tfrecord.gz", ".tfrecord")

    shutil.copy2(gz_path, tmp_gz)

    with tf.io.TFRecordWriter(tmp_out) as writer:
        ds = tf.data.TFRecordDataset(tmp_gz, compression_type="GZIP")
        for rec in ds:
            writer.write(rec.numpy())

    shutil.move(tmp_out, out_drive)

    safe_remove(tmp_gz)
    safe_remove(gz_path)

    return base_out

def internal_restart_cleanup():

    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
    time.sleep(SLEEP_BETWEEN_CHUNKS)

total_unzipped = 0
total_errors = 0

while True:
    pending = list_pending_gz(SRC_DIR)
    if not pending:
        break

    chunk = pending[:CHUNK_SIZE]
    print(f"\nPending gz: {len(pending)} | Processing chunk: {len(chunk)}")

    for i, gz_path in enumerate(chunk, start=1):
        base_gz = os.path.basename(gz_path)
        out_drive = gz_path.replace(".tfrecord.gz", ".tfrecord")

        if os.path.exists(out_drive):

            safe_remove(gz_path)
            continue

        ok = False
        for attempt in range(1, MAX_RETRIES_PER_FILE + 1):
            try:
                out_name = unzip_one(gz_path)
                total_unzipped += 1
                print(f"  [{i}/{len(chunk)}] OK: {base_gz} -> {out_name} (total={total_unzipped})")
                ok = True
                break
            except Exception as e:
                total_errors += 1

                tmp_gz = os.path.join(TMP_DIR, base_gz)
                tmp_out = tmp_gz.replace(".tfrecord.gz", ".tfrecord")
                safe_remove(tmp_out)
                safe_remove(tmp_gz)

                backoff = min(60, (2 ** attempt)) + random.uniform(0, 2.0)
                print(f"    [Attempt {attempt}/{MAX_RETRIES_PER_FILE}] ERROR: {base_gz} -> {e}")
                print(f"    Sleeping {backoff:.1f}s then retrying...")
                time.sleep(backoff)

        if not ok:
            print(f"  [FAIL] Giving up on: {base_gz} (will remain .gz)")

        time.sleep(SLEEP_EACH_FILE)

    print("=== Internal restart: clearing TF + GC + sleep ===")
    internal_restart_cleanup()

print("\nDone.")
print(f"  Total unzipped: {total_unzipped}")
print(f"  Total errors: {total_errors}")

from pprint import pprint
import os
import re
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import geopandas as gpd

os.environ["CUDA_VISIBLE_DEVICES"] = ""

TFRECORD_ROOT_DIR = "/content/drive/My Drive/TFRecords"
SURVEYS_ROOT_DIR  = "/content/drive/My Drive/Master_Thesis/Surveys"
OUT_ROOT_DIR      = os.path.join(TFRECORD_ROOT_DIR, "processed")

os.makedirs(OUT_ROOT_DIR, exist_ok=True)

print("TFRECORD_ROOT_DIR:", TFRECORD_ROOT_DIR)
print("SURVEYS_ROOT_DIR :", SURVEYS_ROOT_DIR)
print("OUT_ROOT_DIR     :", OUT_ROOT_DIR)

TFRECORD_GLOB = os.path.join(TFRECORD_ROOT_DIR, "*.tfrecord")

def discover_survey_years_from_clusters(surveys_root):
    cl_path = os.path.join(surveys_root, "clusters_yeh_spec.csv")
    if not os.path.exists(cl_path):
        print(f"Warning: {cl_path} not found; returning empty mapping.")
        return {}

    df = pd.read_csv(cl_path, usecols=["country", "year"])
    df = df.dropna(subset=["country", "year"])
    df["country"] = df["country"].astype(str).str.upper()
    df["year"]    = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])

    return (
        df.drop_duplicates()
          .groupby("country")["year"]
          .apply(lambda s: sorted(int(x) for x in s.unique()))
          .to_dict()
    )

country_to_years = discover_survey_years_from_clusters(SURVEYS_ROOT_DIR)
print("\nDiscovered survey years from clusters_yeh_spec.csv:")
print(f"  Countries with surveys: {len(country_to_years)}")

def processed_exists_for_raw(raw_path, out_root_dir):
    base = os.path.basename(raw_path)
    m = re.match(r"^([A-Z]{2})_(\d{4})_(\d{2,3})\.tfrecord$", base)
    if not m:
        return False
    cc, year = m.group(1), m.group(2)

    out_dir = os.path.join(out_root_dir, f"{cc}_{year}")
    if not os.path.isdir(out_dir):
        return False

    any_out = glob.glob(os.path.join(out_dir, "*.tfrecord")) + glob.glob(os.path.join(out_dir, "*.npz")) + glob.glob(os.path.join(out_dir, "*"))
    return len(any_out) > 0

def discover_processing_jobs(tfrecord_glob, country_to_years, out_root_dir):
    jobs = []
    rx = re.compile(r"^([A-Z]{2})_(\d{4})_(\d{2,3})\.tfrecord$")

    tf_paths = sorted(glob.glob(tfrecord_glob))
    print(f"\nFound {len(tf_paths)} TFRecord files (uncompressed) from pattern.")

    skipped = 0
    for path in tf_paths:
        base = os.path.basename(path)
        m = rx.match(base)
        if not m:
            continue

        cc   = m.group(1)
        year = int(m.group(2))

        if processed_exists_for_raw(path, out_root_dir):
            skipped += 1
            continue

        known_years = country_to_years.get(cc, [])
        if year not in known_years:
            print(
                f"Warning: year {year} for country '{cc}' not found in "
                f"clusters_yeh_spec mapping (TFRecord: {base})"
            )

        jobs.append({
            "country":       cc,
            "year_range":    f"{year}-{str(year)[-2:]}",
            "tfrecord_path": path,
            "years":         [year],
        })

    print(f"Skipped {skipped} raw TFRecords because processed outputs already exist.")
    return jobs

PROCESSING_JOBS = discover_processing_jobs(TFRECORD_GLOB, country_to_years, OUT_ROOT_DIR)

print("\nDiscovered processing jobs (per raw file):")
print(f"  Total raw jobs: {len(PROCESSING_JOBS)}")
print(f"  Countries in jobs: {sorted({j['country'] for j in PROCESSING_JOBS})}")

GROUPED_JOBS = defaultdict(list)
for job in PROCESSING_JOBS:
    cc = job["country"]
    for y in job["years"]:
        GROUPED_JOBS[(cc, int(y))].append(job["tfrecord_path"])

print("\nGrouped jobs per (country, year):")
print(f"  Total (country, year) groups: {len(GROUPED_JOBS)}")

GOOD_CLUSTERS = defaultdict(set)
BAD_CLUSTERS  = defaultdict(set)

import requests
import zipfile

NE_URL = "https://github.com/nvkelso/natural-earth-vector/archive/refs/heads/master.zip"
NE_DIR = "/content/ne_data_10m"
NE_ZIP = "/content/ne_data_10m.zip"

def ensure_natural_earth():
    if os.path.exists(NE_DIR):
        return
    print("Downloading Natural Earth 1:10m admin-0 country borders ...")
    r = requests.get(NE_URL, timeout=120)
    r.raise_for_status()
    with open(NE_ZIP, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(NE_ZIP, "r") as z:
        z.extractall(NE_DIR)
    print("Natural Earth data extracted.")

def load_admin0_10m():
    ensure_natural_earth()
    for root, _, fs in os.walk(NE_DIR):
        if "ne_10m_admin_0_countries.shp" in fs:
            shp_path = os.path.join(root, "ne_10m_admin_0_countries.shp")
            print("Loading admin-0 countries shapefile from:", shp_path)
            return gpd.read_file(shp_path)
    raise FileNotFoundError("Natural Earth admin_0 countries (10m) shapefile not found.")

try:
    WORLD10 = load_admin0_10m()

    ISO_A2_COL = next(
        (c for c in WORLD10.columns if str(c).upper().startswith("ISO_A2")),
        None
    )
    print("Natural Earth 10m admin-0 borders loaded.")
except Exception as e:
    WORLD10 = None
    ISO_A2_COL = None
    print("Could not load Natural Earth 10m borders:", e)

ISO_OVERRIDES = {
    "LB": "LR",
    "MD": "MG",
}

def get_country_geom(cc):
    if WORLD10 is None or ISO_A2_COL is None:
        return None
    cc2 = ISO_OVERRIDES.get(cc, cc)
    mask = WORLD10[ISO_A2_COL].astype(str) == cc2
    sub = WORLD10.loc[mask]
    if sub.empty:
        return None
    return sub

def plot_country_year_map(country, year, survey_df, good_clusters, bad_clusters):

    df = survey_df.copy()

    if "cluster_index" not in df.columns and "cluster" in df.columns:
        df = df.rename(columns={"cluster": "cluster_index"})

    required_cols = {"cluster_index", "lat", "lon", "wealthpooled"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Skipping plot for {country} {year} – missing columns: {missing}")
        return

    df = df[list(required_cols)].dropna()

    all_clusters = set(df["cluster_index"].astype(int).tolist())
    good_clusters = {int(c) for c in good_clusters}
    bad_clusters  = {int(c) for c in bad_clusters}

    missing_clusters = all_clusters - good_clusters - bad_clusters
    combined_bad = bad_clusters | missing_clusters

    good_df = df[df["cluster_index"].isin(good_clusters)].astype(np.float32)
    bad_df  = df[df["cluster_index"].isin(combined_bad)].astype(np.float32)

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
            good_df["lon"], good_df["lat"],
            c=good_df["wealthpooled"],
            s=25,
            cmap="viridis",
            vmin=vmin, vmax=vmax,
            label="accepted clusters",
        )

    if len(bad_df) > 0:
        ax.scatter(
            bad_df["lon"], bad_df["lat"],
            s=30,
            facecolors="none",
            edgecolors="red",
            linewidths=1.0,
            label="rejected / missing clusters",
        )

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Wealth index (wealthpooled)")

    ax.set_title(f"{country} {year}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", "box")
    ax.legend()
    plt.show()

REQUIRED_KEYS = [
    "BLUE", "GREEN", "LAT", "LON", "NIGHTLIGHTS", "NIR", "RED",
    "SWIR1", "SWIR2", "TEMP1",
    "cluster_index", "lon", "wealthpooled", "year",
]

OPTIONAL_KEYS = [
    "ASP", "ELEV", "SLO", "country", "households", "indx",
    "system:index", "urban_rural", "wealth", "wealthpooled5country",
]

NON_NEGATIVE_BANDS = [
    "RED", "BLUE", "GREEN", "NIGHTLIGHTS", "NIR",
    "SWIR1", "SWIR2", "TEMP1",
]

def contains_neg(feature_map, band):
    arr = np.asarray(feature_map[band].float_list.value)
    return np.any(arr < 0)

def validate_example(feature_map, record_num, cluster_indices, surveys):

    def print_info():
        print(f"  Record {record_num} debug info:")
        if "year" in feature_map and feature_map["year"].float_list.value:
            print("  year:", int(feature_map["year"].float_list.value[0]))
        if "cluster_index" in feature_map and feature_map["cluster_index"].float_list.value:
            print("  cluster_index:", int(feature_map["cluster_index"].float_list.value[0]))

    missing_req_keys = [key for key in REQUIRED_KEYS if key not in feature_map]
    if len(missing_req_keys) > 0:
        print(f"Record {record_num} missing required keys: {missing_req_keys}")
        print_info()
        return False

    year = int(feature_map["year"].float_list.value[0])
    years_allowed = cluster_indices.keys()
    if year not in years_allowed:
        print(f"Record {record_num} has invalid year {year}")
        print_info()
        return False

    cluster_index = int(feature_map["cluster_index"].float_list.value[0])
    if cluster_index not in cluster_indices[year]:
        print(f"Record {record_num} contains invalid cluster index {cluster_index}")
        print_info()
        return False

    lon = np.float32(feature_map["lon"].float_list.value[0])
    survey = surveys[year]
    expected_lon = np.float32(
        survey.loc[survey["cluster_index"] == cluster_index, "lon"].iloc[0]
    )
    if lon != expected_lon:
        print("cluster index:", cluster_index)
        print(f"Record {record_num} contains invalid lon {lon}. Should be {expected_lon}")
        print_info()
        return False

    negative_bands = [band for band in NON_NEGATIVE_BANDS if contains_neg(feature_map, band)]
    if len(negative_bands) > 0:
        print(f"Record {record_num} contains negative bands: {negative_bands}")
        for band in negative_bands:
            band_arr = np.asarray(feature_map[band].float_list.value)
            count = np.sum(band_arr < 0)
            min_val = np.float32(np.min(band_arr))
            print(f'  Band "{band}" - count: {count}, min value: {min_val}')
        print_info()

    return True

def validate_tfrecords_for_country_year(tfrecord_paths, out_root_dir, country, years):

    clusters_path = os.path.join(SURVEYS_ROOT_DIR, "clusters_yeh_spec.csv")
    if not os.path.exists(clusters_path):
        print(f"ERROR: {clusters_path} not found. Cannot validate TFRecords.")
        return

    cl_all = pd.read_csv(clusters_path, float_precision="high")
    cl_all["country"] = cl_all["country"].astype(str).str.upper()
    cl_all["year"]    = pd.to_numeric(cl_all["year"], errors="coerce").astype("Int64")
    cl_all = cl_all.dropna(subset=["country", "year"])

    if "cluster" not in cl_all.columns and "cluster_index" not in cl_all.columns:
        raise ValueError("clusters_yeh_spec.csv must contain 'cluster' or 'cluster_index' column.")

    surveys          = {}
    cluster_indices  = {}
    out_dirs         = {}
    num_good_records = {}

    for year in years:
        y = int(year)
        survey_csv = cl_all[
            (cl_all["country"] == country) &
            (cl_all["year"]    == y)
        ].copy()

        if survey_csv.empty:
            print(f"WARNING: no rows in clusters_yeh_spec for {country}, {y}; skipping this year.")
            continue

        if "cluster_index" not in survey_csv.columns and "cluster" in survey_csv.columns:
            survey_csv = survey_csv.rename(columns={"cluster": "cluster_index"})

        if "cluster_index" not in survey_csv.columns:
            raise ValueError("Expected 'cluster_index' column after renaming from 'cluster'.")

        surveys[y]          = survey_csv
        cluster_indices[y]  = set(survey_csv["cluster_index"])
        out_dir             = os.path.join(out_root_dir, f"{country}_{y}")
        os.makedirs(out_dir, exist_ok=True)
        out_dirs[y]         = out_dir
        num_good_records[y] = 0

    if not surveys:
        print(f"No valid surveys loaded for {country}; skipping these files.")
        return

    global_idx = -1

    for tf_path in sorted(tfrecord_paths):
        iterator = tf.compat.v1.io.tf_record_iterator(tf_path)

        for local_idx, record_str in enumerate(iterator):
            global_idx += 1
            ex = tf.train.Example.FromString(record_str)
            feature_map = ex.features.feature

            is_valid_record = validate_example(
                feature_map, global_idx, cluster_indices, surveys
            )

            if is_valid_record:
                year = int(feature_map["year"].float_list.value[0])
                if year not in surveys:

                    continue

                cluster_index = int(feature_map["cluster_index"].float_list.value[0])
                survey = surveys[year]

                lat_val = np.float32(
                    survey.loc[survey["cluster_index"] == cluster_index, "lat"].iloc[0]
                )

                if "lat" in feature_map and feature_map["lat"].float_list.value:
                    existing_lat = np.float32(feature_map["lat"].float_list.value[0])
                    if existing_lat != lat_val:
                        print(
                            f"Warning: record {global_idx} for {country} {year}, "
                            f"cluster_index {cluster_index} has lat={existing_lat}, "
                            f"but clusters_yeh_spec has lat={lat_val}. Overwriting."
                        )
                        feature_map["lat"].float_list.value[0] = lat_val
                else:
                    feature_map["lat"].float_list.value.append(lat_val)

                out_dir  = out_dirs[year]
                out_path = os.path.join(out_dir, f"{num_good_records[year]}.tfrecord")

                with tf.io.TFRecordWriter(out_path) as writer:
                    writer.write(ex.SerializeToString())

                if cluster_index in cluster_indices[year]:
                    cluster_indices[year].remove(cluster_index)
                num_good_records[year] += 1
                GOOD_CLUSTERS[(country, year)].add(cluster_index)

            else:

                year_feat = feature_map.get("year")
                cl_feat   = feature_map.get("cluster_index")
                if (
                    year_feat is not None
                    and cl_feat is not None
                    and year_feat.float_list.value
                    and cl_feat.float_list.value
                ):
                    y_bad  = int(year_feat.float_list.value[0])
                    cl_bad = int(cl_feat.float_list.value[0])
                    BAD_CLUSTERS[(country, y_bad)].add(cl_bad)

            if (global_idx + 1) % 100 == 0:
                print(
                    f"  Processed {global_idx + 1} records for {country} "
                    f"{years} so far..."
                )

    if global_idx >= 0:
        total_good_records = sum(num_good_records.values())
        total_seen = global_idx + 1
        print(
            f"Finished {country} {years}: "
            f"{total_seen} records seen, {total_good_records} good, "
            f"{total_seen - total_good_records} bad"
        )
    else:
        print(f"No records found in any of {tfrecord_paths}")

if not GROUPED_JOBS:
    print("No grouped jobs to run. Check TFRECORD_ROOT_DIR and SURVEYS_ROOT_DIR.")
else:
    print(f"\nWill process {len(GROUPED_JOBS)} (country, year) groups.\n")
    for (country, year), paths in sorted(GROUPED_JOBS.items()):
        years = [int(year)]
        print("==============================================")
        print(f"Processing country={country}, year={year}")
        print(f"  Raw TFRecord files in group: {len(paths)}")
        print("==============================================")

        validate_tfrecords_for_country_year(
            tfrecord_paths=paths,
            out_root_dir=OUT_ROOT_DIR,
            country=country,
            years=years,
        )

clusters_path = os.path.join(SURVEYS_ROOT_DIR, "clusters_yeh_spec.csv")
if not os.path.exists(clusters_path):
    print(f"ERROR: {clusters_path} not found. Cannot plot maps.")
else:
    cl_all = pd.read_csv(clusters_path, float_precision="high")
    cl_all["country"] = cl_all["country"].astype(str).str.upper()
    cl_all["year"]    = pd.to_numeric(cl_all["year"], errors="coerce").astype("Int64")
    cl_all = cl_all.dropna(subset=["country", "year"])

    if "cluster_index" not in cl_all.columns and "cluster" in cl_all.columns:
        cl_all = cl_all.rename(columns={"cluster": "cluster_index"})

    keys_good = set(GOOD_CLUSTERS.keys())
    keys_bad  = set(BAD_CLUSTERS.keys())
    all_keys  = sorted(keys_good | keys_bad)

    print(f"Plotting maps for {len(all_keys)} country-year combinations.")

    for (cc, year) in all_keys:
        year_int = int(year)
        survey_df = cl_all[
            (cl_all["country"] == cc) &
            (cl_all["year"]    == year_int)
        ].copy()

        if survey_df.empty:
            print(f"Skipping {cc} {year_int} – no rows in clusters_yeh_spec.")
            continue

        good_set = GOOD_CLUSTERS.get((cc, year_int), set())
        bad_set  = BAD_CLUSTERS.get((cc, year_int), set())

        print(f"\nCountry {cc}, year {year_int}: {len(good_set)} accepted, {len(bad_set)} rejected (before missing).")
        plot_country_year_map(
            country=cc,
            year=year_int,
            survey_df=survey_df,
            good_clusters=good_set,
            bad_clusters=bad_set,
        )
