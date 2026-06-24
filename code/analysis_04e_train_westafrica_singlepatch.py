import os, glob, json, pickle, random, math, shutil, gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.utils import register_keras_serializable

from tqdm.auto import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import psutil

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

from google.colab import drive
if not os.path.exists("/content/drive"):
    drive.mount("/content/drive")

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

SEED = 123
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("TF version:", tf.__version__)
print("Compute policy:", policy.name)

AUTOTUNE = tf.data.AUTOTUNE

TARGET_H = 224
BATCH_SIZE = 64

EPOCHS_INCOUNTRY = 150
EPOCHS_OOC       = 200

TFRECORD_MERGED_DIR = "/content/drive/My Drive/TFRecords/merged_rounds"

DD_BASE       = "/content/drive/My Drive/Data Distribution"
COMBINED_CSV  = os.path.join(DD_BASE, "dhs_combined_df.csv")
INC_FOLDS_PKL = os.path.join(DD_BASE, "dhs_incountry_folds.pkl")
OOC_FOLDS_PKL = os.path.join(DD_BASE, "dhs_ooc_folds.pkl")

GLOBAL_STATS_JSON = "/content/drive/My Drive/Data Validation/Band Analysis/band_stats_summary.json"

ROOT_OUT = os.path.join(DD_BASE, "cnn_results_merged_retrain_best_hp_milstyle")
CKPT_DIR = os.path.join(ROOT_OUT, "checkpoints")
for d in [ROOT_OUT, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

SSD_ROOT      = "/content/local_cache"
LOCAL_RAW_DIR = os.path.join(SSD_ROOT, "raw_tfrecords")
SSD_META_DIR  = os.path.join(SSD_ROOT, "metadata")
SSD_CKPT_DIR  = "/content/ckpts_tmp"

for d in [LOCAL_RAW_DIR, SSD_META_DIR, SSD_CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

TFRECORD_BAND_ORDER = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "TEMP1", "NIGHTLIGHTS"]
BANDS_MS = ["RED","GREEN","BLUE","SWIR1","SWIR2","TEMP1","NIR"]
BANDS_NL = ["NIGHTLIGHTS"]

CLIP_NEGATIVE = {"BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "NIGHTLIGHTS"}

TEMP_SCALE  = 0.00341802
TEMP_OFFSET = 149.0

WEST_AFRICA_CODES = {"BJ","SN","GM","MR","GH","CI","BF","ML","NG","SL","GN"}

EARLY_STOPPING_PATIENCE = 30
FOLDS_TO_RUN   = ["A", "B", "C"]
SCHEMES_TO_RUN = ["incountry"]
VARIANTS_TO_RUN = ["ms", "nl"]

PARAM_GRID_MS = [
    (1e-3, 0.0),
    (1e-3, 1e-6),
    (1e-3, 1e-5),
    (1e-3, 1e-4),
    (1e-3, 1e-3),

    (1e-4, 0.0),
    (1e-4, 1e-5),
    (1e-4, 1e-4),
    (1e-4, 1e-3),
    (1e-4, 1e-2),
]

PARAM_GRID_NL = [
    (1e-3, 0.0),
    (1e-3, 1e-6),
    (1e-4, 0.0),
    (1e-4, 1e-4),
]

CACHE_VAL_TEST_IN_RAM   = True
CACHE_TRAIN_IN_RAM      = True
TRAIN_CACHE_SAFETY_FRAC = 0.60

print("TFRECORD_MERGED_DIR:", TFRECORD_MERGED_DIR)
print("DD_BASE:", DD_BASE)
print("ROOT_OUT:", ROOT_OUT)
print("CKPT_DIR:", CKPT_DIR)

PRETRAIN_DRIVE = "/content/drive/My Drive/TFRecords_MIL/resnet18_preact_imagenet_rgb_from_npz.weights.h5"

PRETRAIN_SSD_DIR = os.path.join(SSD_ROOT, "pretrained")
PRETRAIN_SSD = os.path.join(PRETRAIN_SSD_DIR, "resnet18_preact_imagenet_rgb_from_npz.weights.h5")
os.makedirs(PRETRAIN_SSD_DIR, exist_ok=True)

if not os.path.exists(PRETRAIN_SSD):
    if os.path.exists(PRETRAIN_DRIVE):
        print("[PRETRAIN] Copying pretrained weights to SSD...")
        shutil.copy(PRETRAIN_DRIVE, PRETRAIN_SSD)
    else:
        raise FileNotFoundError(f"Missing pretrained weights on Drive: {PRETRAIN_DRIVE}")

print("[PRETRAIN] SSD weights path:", PRETRAIN_SSD)

print("\n--- METADATA & FILE SYNC (NON-MIL) ---")

for src_path, dst_name in [
    (COMBINED_CSV,  "dhs_combined_df.csv"),
    (INC_FOLDS_PKL, "dhs_incountry_folds.pkl"),
    (OOC_FOLDS_PKL, "dhs_ooc_folds.pkl"),
]:
    dst_path = os.path.join(SSD_META_DIR, dst_name)
    if not os.path.exists(dst_path) and os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

stats_dst = os.path.join(SSD_META_DIR, "band_stats_summary.json")
if not os.path.exists(stats_dst) and os.path.exists(GLOBAL_STATS_JSON):
    shutil.copy(GLOBAL_STATS_JSON, stats_dst)

meta_path = os.path.join(SSD_META_DIR, "dhs_combined_df.csv")
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Missing metadata CSV: {meta_path}")

df_meta = pd.read_csv(meta_path)

def _pick(df, cands):
    return next(c for c in cands if c in df.columns)

C_COUNTRY = _pick(df_meta, ["country", "CC"])
C_CLUSTER = _pick(df_meta, ["cluster", "clu"])
C_YEAR    = _pick(df_meta, ["year", "survey_year"])

df_meta[C_COUNTRY] = df_meta[C_COUNTRY].astype(str).str.upper().str.strip().str[:2]
df_meta = df_meta[df_meta[C_COUNTRY].isin(WEST_AFRICA_CODES)].reset_index(drop=True)

df_meta["_key"] = (
    df_meta[C_COUNTRY] + "|" +
    df_meta[C_YEAR].astype(str) + "|" +
    df_meta[C_CLUSTER].astype(str)
)

if "wealthpooled" not in df_meta.columns:
    raise KeyError("dhs_combined_df.csv is missing 'wealthpooled' column.")
KEY_TO_LABEL = dict(zip(df_meta["_key"], df_meta["wealthpooled"].astype(np.float32)))

with open(os.path.join(SSD_META_DIR, "dhs_incountry_folds.pkl"), "rb") as f:
    INCOUNTRY_FOLDS = pickle.load(f)
with open(os.path.join(SSD_META_DIR, "dhs_ooc_folds.pkl"), "rb") as f:
    OOC_FOLDS = pickle.load(f)

with open(stats_dst, "r") as f:
    band_stats = json.load(f)

first_band = list(band_stats.keys())[0]
keys_in_json = band_stats[first_band].keys()
print(f"DEBUG: Found keys in stats file: {list(keys_in_json)}")

if "global_mean" in keys_in_json:
    mean_key, std_key = "global_mean", "global_std"
elif "mean_good" in keys_in_json:
    mean_key, std_key = "mean_good", "std_good"
elif "mean" in keys_in_json:
    mean_key, std_key = "mean", "std"
else:
    raise KeyError(f"Could not find mean/std keys in {keys_in_json}")

print(f"Using stats keys: {mean_key} / {std_key}")

BAND_MEAN = {}
BAND_STD = {}
for b in TFRECORD_BAND_ORDER:
    if b in band_stats:
        BAND_MEAN[b] = float(band_stats[b][mean_key])
        BAND_STD[b]  = float(band_stats[b][std_key])
    else:
        print(f"WARNING: Band {b} missing from stats file. Using default 0/1.")
        BAND_MEAN[b] = 0.0
        BAND_STD[b]  = 1.0

FULL_IDX_TO_KEY = pd.read_csv(meta_path)
FULL_IDX_TO_KEY[C_COUNTRY] = FULL_IDX_TO_KEY[C_COUNTRY].astype(str).str.upper().str.strip().str[:2]
FULL_IDX_TO_KEY["_key"] = (
    FULL_IDX_TO_KEY[C_COUNTRY] + "|" +
    FULL_IDX_TO_KEY[C_YEAR].astype(str) + "|" +
    FULL_IDX_TO_KEY[C_CLUSTER].astype(str)
)
FULL_IDX_TO_KEY = FULL_IDX_TO_KEY["_key"].to_dict()

print("Loaded df_meta rows (West Africa):", len(df_meta))
print("Example key:", next(iter(KEY_TO_LABEL.keys())))

print("\n--- TFRECORD SYNC + FILE MAP (MERGED TFRECORDS) ---")

all_drive_files = sorted(glob.glob(os.path.join(TFRECORD_MERGED_DIR, "*.tfrecord")))
if len(all_drive_files) == 0:
    raise FileNotFoundError(f"No TFRecords found in: {TFRECORD_MERGED_DIR}")

target_drive_paths = [f for f in all_drive_files if any(cc in os.path.basename(f).upper() for cc in WEST_AFRICA_CODES)]
if not target_drive_paths:
    target_drive_paths = all_drive_files

target_filenames = {os.path.basename(p): p for p in target_drive_paths}

existing_local_files = set(os.listdir(LOCAL_RAW_DIR))
missing_files = [path for fname, path in target_filenames.items() if fname not in existing_local_files]

if missing_files:
    print(f"Copying {len(missing_files)} missing TFRecord files to SSD...")
    def copy_worker(src):
        dst = os.path.join(LOCAL_RAW_DIR, os.path.basename(src))
        shutil.copy(src, dst)
        return 1
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(tqdm(ex.map(copy_worker, missing_files), total=len(missing_files), leave=False))
else:
    print("Files already synced.")

local_files = sorted(glob.glob(os.path.join(LOCAL_RAW_DIR, "*.tfrecord")))
print(f"Found {len(local_files)} merged TFRecord files.")

probe_path = local_files[0]

probe_spec = {
    "country": tf.io.FixedLenFeature([], tf.string),
    "year": tf.io.FixedLenFeature([], tf.float32),
    "cluster_index": tf.io.FixedLenFeature([], tf.float32),
}
for b in TFRECORD_BAND_ORDER:
    probe_spec[b] = tf.io.VarLenFeature(tf.float32)

raw0 = next(iter(tf.data.TFRecordDataset([probe_path])))
ex0 = tf.io.parse_single_example(raw0, probe_spec)
feature_keys = list(ex0.keys())
print("DEBUG: example feature keys (first record):", feature_keys)

print("[DTYPE] year dtype:", ex0["year"].dtype)
print("[DTYPE] cluster_index dtype:", ex0["cluster_index"].dtype)

probe_band = "BLUE"
n0 = int(tf.size(tf.sparse.to_dense(ex0[probe_band])).numpy())
RAW_H = int(round(math.sqrt(n0)))
if RAW_H * RAW_H != n0:
    raise ValueError(f"Cannot infer RAW_H cleanly from {probe_band} length={n0}")

RAW_K = 1
print(f"[INFER] RAW_H={RAW_H}, RAW_K={RAW_K} (from {probe_band} length={n0})")
print(f"[CROP]  RAW_H={RAW_H} -> TARGET_H={TARGET_H}")

INDEX_CACHE = os.path.join(SSD_META_DIR, "local_index.pkl")
if os.path.exists(INDEX_CACHE):
    print("Loading index cache...")
    with open(INDEX_CACHE, "rb") as f:
        tfrecord_index = pickle.load(f)
else:
    print("Indexing local files...")
    tfrecord_index = {}

    idx_spec = {
        "country": tf.io.FixedLenFeature([], tf.string),
        "year": tf.io.FixedLenFeature([], tf.float32),
        "cluster_index": tf.io.FixedLenFeature([], tf.float32),
    }

    def index_worker(path):
        local_idx = {}
        try:
            for i, raw in enumerate(tf.data.TFRecordDataset(path)):
                ex = tf.io.parse_single_example(raw, idx_spec)
                cc = ex["country"].numpy().decode("utf-8").upper()[:2]
                yy = int(round(float(ex["year"].numpy())))
                cl = int(round(float(ex["cluster_index"].numpy())))
                key = f"{cc}|{yy}|{cl}"
                local_idx[key] = (path, i)
        except Exception:
            pass
        return local_idx

    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in tqdm(ex.map(index_worker, local_files), total=len(local_files)):
            tfrecord_index.update(res)

    with open(INDEX_CACHE, "wb") as f:
        pickle.dump(tfrecord_index, f)

print(f"Index ready: {len(tfrecord_index)} clusters indexed.")

def _center_crop_hw(x, raw_h, target_h):
    if raw_h == target_h:
        return x
    start = (raw_h - target_h) // 2
    return x[:, start:start+target_h, start:start+target_h, :]

def _estimate_cache_bytes(n_examples, variant):
    c = 7 if variant == "ms" else 1
    x_bytes = n_examples * RAW_K * TARGET_H * TARGET_H * c * 2
    y_bytes = n_examples * 1 * 4
    overhead = int(0.15 * x_bytes)
    return x_bytes + y_bytes + overhead

def get_dataset_in_place(indices, variant, split, return_country=False):
    raw_keys = [FULL_IDX_TO_KEY[i] for i in indices if i in FULL_IDX_TO_KEY]
    target_keys = [k for k in raw_keys if (k in tfrecord_index) and (k in KEY_TO_LABEL)]
    if not target_keys:
        return None, 0
    n = len(target_keys)

    relevant_files = sorted({tfrecord_index[k][0] for k in target_keys})

    keys_tf = tf.constant(target_keys)
    valid_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tf, tf.ones([len(target_keys)], tf.int32)),
        default_value=0
    )

    label_vals = np.asarray([KEY_TO_LABEL[k] for k in target_keys], dtype=np.float32)
    label_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tf, tf.constant(label_vals, tf.float32)),
        default_value=tf.constant(np.nan, tf.float32)
    )

    bands = BANDS_MS if variant == "ms" else BANDS_NL
    C = len(bands)
    means = tf.constant([BAND_MEAN[b] for b in bands], dtype=tf.float32)
    stds  = tf.constant([BAND_STD[b]  for b in bands], dtype=tf.float32)

    def make_key(country_str, year_f32, cluster_f32):
        c = tf.strings.upper(country_str)
        c = tf.strings.substr(c, 0, 2)
        y = tf.cast(tf.round(year_f32), tf.int32)
        cl = tf.cast(tf.round(cluster_f32), tf.int32)
        return tf.strings.join([c, tf.as_string(y), tf.as_string(cl)], separator="|")

    def is_valid_key(raw):
        s = {
            "country": tf.io.FixedLenFeature([], tf.string),
            "year": tf.io.FixedLenFeature([], tf.float32),
            "cluster_index": tf.io.FixedLenFeature([], tf.float32),
        }
        e = tf.io.parse_single_example(raw, s)
        k = make_key(e["country"], e["year"], e["cluster_index"])
        return valid_table.lookup(k) == 1

    def parse(raw):
        spec = {
            "country": tf.io.FixedLenFeature([], tf.string),
            "year": tf.io.FixedLenFeature([], tf.float32),
            "cluster_index": tf.io.FixedLenFeature([], tf.float32),
        }
        for b in TFRECORD_BAND_ORDER:
            spec[b] = tf.io.VarLenFeature(tf.float32)

        e = tf.io.parse_single_example(raw, spec)

        k = make_key(e["country"], e["year"], e["cluster_index"])
        y = tf.cast(label_table.lookup(k), tf.float32)
        y = tf.reshape(y, [1])

        imgs = []
        for b in bands:
            flat = tf.sparse.to_dense(e[b])
            arr = tf.reshape(flat, [RAW_K, RAW_H, RAW_H])

            if b == "TEMP1":
                arr = arr * tf.cast(TEMP_SCALE, tf.float32) + tf.cast(TEMP_OFFSET, tf.float32)

            if b in CLIP_NEGATIVE:
                arr = tf.maximum(arr, 0.0)

            arr = tf.where(tf.math.is_finite(arr), arr, tf.zeros_like(arr))
            if b == "NIGHTLIGHTS":
                arr = tf.math.log1p(arr)

            imgs.append(arr)

        X = tf.stack(imgs, axis=-1)
        X = tf.ensure_shape(X, [RAW_K, RAW_H, RAW_H, C])

        X = _center_crop_hw(X, RAW_H, TARGET_H)
        X = tf.ensure_shape(X, [RAW_K, TARGET_H, TARGET_H, C])

        X = (X - means) / (stds + 1e-8)
        X = tf.clip_by_value(X, -10., 10.)
        X = tf.cast(X, tf.float16)

        if return_country:
            return X, y, e["country"]
        return X, y

    ds = tf.data.TFRecordDataset(relevant_files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.filter(is_valid_key)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)

    if return_country:
        ds = ds.filter(lambda x, y, c: tf.reduce_all(tf.math.is_finite(y)))
    else:
        ds = ds.filter(lambda x, y: tf.reduce_all(tf.math.is_finite(y)))

    avail = psutil.virtual_memory().available
    est   = _estimate_cache_bytes(n, variant)
    cache_ok_train = (split == "train") and CACHE_TRAIN_IN_RAM and (est < TRAIN_CACHE_SAFETY_FRAC * avail) and (not return_country)
    cache_ok_eval  = (split in ("val", "test")) and CACHE_VAL_TEST_IN_RAM

    if cache_ok_eval or cache_ok_train:
        if cache_ok_train:
            print(f"[CACHE] TRAIN {variant}: est={est/1e9:.1f}GB avail={avail/1e9:.1f}GB")
        ds = ds.cache()

    if split == "train" and (not return_country):
        ds = ds.shuffle(5000)

        def aug(x, y):

            if tf.random.uniform([]) > 0.5:
                x = tf.reverse(x, axis=[2])
            if tf.random.uniform([]) > 0.5:
                x = tf.reverse(x, axis=[1])

            c = tf.shape(x)[-1]
            x32 = tf.cast(x, tf.float32)

            if c > 1:
                noise = tf.random.uniform([], -0.2, 0.2, dtype=tf.float32)
                x32 = x32 + noise
            else:
                scale = tf.random.uniform([], 0.90, 1.10, dtype=tf.float32)
                noise = tf.random.normal([], mean=0.0, stddev=0.05, dtype=tf.float32)
                bias  = tf.random.uniform([], -0.03, 0.03, dtype=tf.float32)
                x32 = x32 * scale + noise + bias

            x32 = tf.clip_by_value(x32, -10.0, 10.0)
            x = tf.cast(x32, tf.float16)
            return x, y

        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, n

def conv3x3(filters, stride=1, weight_reg=None):
    return layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False,
        kernel_initializer="he_normal", kernel_regularizer=weight_reg
    )

def conv1x1(filters, stride=1, weight_reg=None):
    return layers.Conv2D(
        filters, 1, strides=stride, padding="same", use_bias=False,
        kernel_initializer="he_normal", kernel_regularizer=weight_reg
    )

@register_keras_serializable()
class PreActBlock(layers.Layer):
    def __init__(self, filters, stride=1, weight_reg=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.weight_reg = weight_reg

        self.bn1 = layers.BatchNormalization()
        self.conv1 = conv3x3(filters, stride, weight_reg)

        self.bn2 = layers.BatchNormalization()
        self.conv2 = conv3x3(filters, 1, weight_reg)

        self.shortcut = conv1x1(filters, stride, weight_reg) if stride != 1 else None

    def call(self, x, training=False):
        shortcut = x
        out = self.bn1(x, training=training)
        out = tf.nn.relu(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        return out + shortcut

    def get_config(self):
        return {
            **super().get_config(),
            "filters": self.filters,
            "stride": self.stride,
            "weight_reg": self.weight_reg,
        }

def _make_resnet18_backbone_generic(input_shape, l2_weight=0.0, name="resnet18_backbone"):
    weight_reg = keras.regularizers.l2(l2_weight) if (l2_weight and l2_weight > 0) else None
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        64, 7, strides=2, padding="same", use_bias=False,
        kernel_initializer="he_normal", kernel_regularizer=weight_reg,
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    def make_group(x, filters, num_blocks, stride_first):
        x = PreActBlock(filters, stride=stride_first, weight_reg=weight_reg)(x)
        for _ in range(1, num_blocks):
            x = PreActBlock(filters, stride=1, weight_reg=weight_reg)(x)
        return x

    x = make_group(x,  64, 2, 1)
    x = make_group(x, 128, 2, 2)
    x = make_group(x, 256, 2, 2)
    x = make_group(x, 512, 2, 2)

    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU()(x)
    features = layers.GlobalAveragePooling2D(name="gap")(x)

    return keras.Model(inputs, features, name=name)

def make_hs_conv1_weights_from_rgb(rgb_weights, target_in_channels, hs_weight_init="samescaled"):
    rgb_weights = np.asarray(rgb_weights, dtype=np.float32)
    F_h, F_w, C_rgb, num_filters = rgb_weights.shape
    assert C_rgb == 3, f"Expected 3 RGB channels, got {C_rgb}"

    if target_in_channels == 3:
        return rgb_weights.astype(np.float32)

    if target_in_channels < 3:
        rgb_mean = rgb_weights.mean(axis=2, keepdims=True)
        hs_weights = np.tile(rgb_mean, (1, 1, target_in_channels, 1))
        return hs_weights.astype(np.float32)

    num_hs = target_in_channels - 3

    if hs_weight_init == "random":
        mu = float(rgb_weights.mean())
        sd = float(rgb_weights.std())
        hs = tf.random.truncated_normal([F_h, F_w, num_hs, num_filters], mean=mu, stddev=sd, dtype=tf.float32).numpy()
        out = np.concatenate([rgb_weights, hs], axis=2)

    elif hs_weight_init == "same":
        rgb_mean = rgb_weights.mean(axis=2, keepdims=True)
        hs = np.tile(rgb_mean, (1, 1, num_hs, 1))
        out = np.concatenate([rgb_weights, hs], axis=2)

    elif hs_weight_init == "samescaled":
        rgb_mean = rgb_weights.mean(axis=2, keepdims=True)
        hs = np.tile(rgb_mean, (1, 1, num_hs, 1))
        scale = 3.0 / (3.0 + num_hs)
        out = np.concatenate([rgb_weights * scale, hs * scale], axis=2)

    else:
        raise ValueError(f"Unknown hs_weight_init: {hs_weight_init}")

    assert out.shape == (F_h, F_w, target_in_channels, num_filters)
    return out.astype(np.float32)

def init_backbone_from_imagenet_with_hs(backbone, imagenet_rgb_weights_h5_path, hs_weight_init="samescaled", verbose=True):
    rgb_backbone = _make_resnet18_backbone_generic(
        input_shape=(TARGET_H, TARGET_H, 3),
        l2_weight=0.0,
        name="resnet18_v2_rgb_tmp",
    )
    rgb_backbone.load_weights(imagenet_rgb_weights_h5_path)

    if verbose:
        print("[PRETRAIN] Loaded RGB weights from:", imagenet_rgb_weights_h5_path)

    for layer in backbone.layers:
        if not layer.weights:
            continue
        if isinstance(layer, PreActBlock):
            continue

        if layer.name == "conv1":
            rgb_w = rgb_backbone.get_layer("conv1").get_weights()[0]
            target_in_channels = layer.get_weights()[0].shape[2]
            new_w = make_hs_conv1_weights_from_rgb(rgb_w, target_in_channels, hs_weight_init=hs_weight_init)
            layer.set_weights([new_w])
            if verbose:
                print(f"[PRETRAIN] Initialized conv1 for C={target_in_channels} via '{hs_weight_init}'")
            continue

        try:
            rgb_layer = rgb_backbone.get_layer(layer.name)
        except ValueError:
            if verbose:
                print("[PRETRAIN] Skip (no RGB match):", layer.name)
            continue

        rw = rgb_layer.get_weights()
        lw = layer.get_weights()
        if len(rw) != len(lw) or any(a.shape != b.shape for a, b in zip(rw, lw)):
            if verbose:
                print("[PRETRAIN] Skip (shape mismatch):", layer.name)
            continue
        layer.set_weights(rw)

    ms_blocks  = [l for l in backbone.layers if isinstance(l, PreActBlock)]
    rgb_blocks = [l for l in rgb_backbone.layers if isinstance(l, PreActBlock)]
    if len(ms_blocks) != len(rgb_blocks):
        raise ValueError(f"PreActBlock count mismatch: target={len(ms_blocks)} rgb={len(rgb_blocks)}")

    for i, (b_tgt, b_rgb) in enumerate(zip(ms_blocks, rgb_blocks)):
        b_tgt.set_weights(b_rgb.get_weights())
        if verbose and i == 0:
            print("[PRETRAIN] Copied PreActBlocks by order (showing first copy).")

    if verbose:
        print("[PRETRAIN] Finished initializing backbone:", backbone.name)

def make_mil_model(variant, lr, l2):
    l2_val = float(l2) if l2 is not None else 0.0
    weight_reg = keras.regularizers.l2(l2_val) if (l2_val > 0) else None

    bands = len(BANDS_MS) if variant == "ms" else len(BANDS_NL)
    mil_inp = keras.Input((RAW_K, TARGET_H, TARGET_H, bands))

    backbone = _make_resnet18_backbone_generic(
        input_shape=(TARGET_H, TARGET_H, bands),
        l2_weight=l2_val,
        name=f"backbone_{variant}",
    )

    if variant == "ms":
        print("[PRETRAIN] Initializing MS backbone from SSD weights:", PRETRAIN_SSD)
        init_backbone_from_imagenet_with_hs(
            backbone=backbone,
            imagenet_rgb_weights_h5_path=PRETRAIN_SSD,
            hs_weight_init="samescaled",
            verbose=True,
        )

    td  = layers.TimeDistributed(backbone)(mil_inp)
    agg = layers.GlobalAveragePooling1D(name="gap")(td)
    out = layers.Dense(1, kernel_regularizer=weight_reg, dtype="float32")(agg)

    model = keras.Model(mil_inp, out, name=f"mil_{variant}")
    opt = keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)

    if variant == "nl":
        loss_fn = tf.keras.losses.Huber(delta=1.0)
    else:
        loss_fn = "mse"

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[keras.metrics.MeanSquaredError(name="mse"), "mae"],
        jit_compile=False
    )
    return model

print("\n--- STARTING GRID SEARCH TRAINING ---")

RES_CSV = os.path.join(ROOT_OUT, "cnn_training_log.csv")
results = pd.read_csv(RES_CSV).to_dict("records") if os.path.exists(RES_CSV) else []

def _already_done(ckpt_path):
    return any(r.get("ckpt_path") == ckpt_path for r in results) and os.path.exists(ckpt_path)

def _grid_for_variant(v):
    return PARAM_GRID_MS if v == "ms" else PARAM_GRID_NL

for scheme in SCHEMES_TO_RUN:
    folds_dict = INCOUNTRY_FOLDS if scheme == "incountry" else OOC_FOLDS
    epochs = EPOCHS_INCOUNTRY if scheme == "incountry" else EPOCHS_OOC
    print(f"\n>>> SCHEME: {scheme.upper()} <<<")

    for fold in FOLDS_TO_RUN:
        for variant in VARIANTS_TO_RUN:
            ds, counts = {}, {}
            for split in ["train", "val", "test"]:
                idxs = np.asarray(folds_dict[fold][split], dtype=np.int64)
                ds[split], counts[split] = get_dataset_in_place(idxs, variant, split, return_country=False)

            if counts["train"] == 0:
                print(f"No train data for {scheme}/{fold}/{variant}. Skipping variant.")
                continue

            for lr, l2 in _grid_for_variant(variant):
                tag = f"{scheme}_{fold}_{variant}_lr{lr:.0e}_l2{l2:.0e}"
                ckpt = os.path.join(CKPT_DIR, f"{tag}.keras")

                if _already_done(ckpt):
                    print(f"Skipping {tag} (Already done)")
                    continue

                print(f"Training {tag}...")

                tf.keras.backend.clear_session()
                gc.collect()

                model = make_mil_model(variant, lr=lr, l2=l2)

                tmp_best = os.path.join(SSD_CKPT_DIR, "tmp_best.keras")
                if os.path.exists(tmp_best):
                    try:
                        os.remove(tmp_best)
                    except Exception:
                        pass

                cb = [
                    keras.callbacks.ModelCheckpoint(tmp_best, monitor="val_mse", save_best_only=True, verbose=0),
                    keras.callbacks.EarlyStopping(
                        monitor="val_mse",
                        patience=EARLY_STOPPING_PATIENCE,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    keras.callbacks.TerminateOnNaN(),
                ]

                def ev_r2(d, n):
                    yt, yp = [], []
                    for x, y in d.take(math.ceil(n / BATCH_SIZE)):
                        yt.extend(y.numpy().reshape(-1).tolist())
                        yp.extend(model(x, training=False).numpy().reshape(-1).tolist())
                    yt = np.asarray(yt[:n], dtype=np.float32)
                    yp = np.asarray(yp[:n], dtype=np.float32)
                    return float(r2_score(yt, yp))

                try:
                    model.fit(
                        ds["train"],
                        validation_data=ds["val"],
                        epochs=epochs,
                        steps_per_epoch=math.ceil(counts["train"] / BATCH_SIZE),
                        validation_steps=math.ceil(counts["val"] / BATCH_SIZE),
                        callbacks=cb,
                        verbose=1,
                    )

                    if os.path.exists(tmp_best):
                        model.load_weights(tmp_best)

                    model.save(ckpt)

                    row = {
                        "scheme": scheme,
                        "fold": fold,
                        "variant": variant,
                        "lr": float(lr),
                        "l2": float(l2),
                        "ckpt_path": ckpt,
                        "val_r2": ev_r2(ds["val"], counts["val"]),
                        "test_r2": ev_r2(ds["test"], counts["test"]),
                    }
                    results.append(row)
                    pd.DataFrame(results).to_csv(RES_CSV, index=False)
                    print("Saved row:", row)

                except Exception as e:
                    print(f"Error {tag}: {e}")
                    import traceback
                    traceback.print_exc()

print("\nTraining loop finished. Log saved at:", RES_CSV)

print("\n--- RIDGE FUSION ---")

df_retrain = pd.DataFrame(results)
if df_retrain.empty and os.path.exists(RES_CSV):
    df_retrain = pd.read_csv(RES_CSV)

fusion_out = os.path.join(ROOT_OUT, "ridge_fusion_scaled_predictions.csv")
RIDGE_ALPHAS = [10.0 ** p for p in range(-4, 5)]

def get_features(model, ds):
    feat_model = keras.Model(model.input, model.get_layer("gap").output)
    fl, yl, cl = [], [], []
    for x, y, c in tqdm(ds, leave=False):
        fl.append(feat_model.predict(x, verbose=0))
        yl.append(y.numpy().reshape(-1))
        cl.append(c.numpy().reshape(-1))
    if not fl:
        return np.array([]), np.array([]), np.array([])

    cl = np.concatenate(cl, axis=0)
    cl = np.asarray([
        (z.decode("utf-8", errors="ignore") if isinstance(z, (bytes, bytearray, np.bytes_)) else str(z))
        for z in cl
    ], dtype=object)

    return (
        np.concatenate(fl, axis=0),
        np.concatenate(yl, axis=0),
        cl,
    )

fusion_rows = []

if not df_retrain.empty and "scheme" in df_retrain.columns:
    for scheme in df_retrain["scheme"].unique():
        folds_dict = INCOUNTRY_FOLDS if scheme == "incountry" else OOC_FOLDS
        for fold in df_retrain[df_retrain["scheme"] == scheme]["fold"].unique():
            sub = df_retrain[(df_retrain.scheme == scheme) & (df_retrain.fold == fold)]
            if not (("ms" in set(sub.variant)) and ("nl" in set(sub.variant))):
                continue

            ms_row = sub[sub.variant == "ms"].sort_values("val_r2", ascending=False).iloc[0]
            nl_row = sub[sub.variant == "nl"].sort_values("val_r2", ascending=False).iloc[0]

            print(f"\n>>> FUSION: {scheme}/{fold} <<<")
            print(f"  ms: val_r2={ms_row.val_r2:.4f}  ckpt={os.path.basename(ms_row.ckpt_path)}")
            print(f"  nl: val_r2={nl_row.val_r2:.4f}  ckpt={os.path.basename(nl_row.ckpt_path)}")

            tf.keras.backend.clear_session(); gc.collect()
            m_ms = keras.models.load_model(ms_row.ckpt_path, custom_objects={"PreActBlock": PreActBlock})
            tf.keras.backend.clear_session(); gc.collect()
            m_nl = keras.models.load_model(nl_row.ckpt_path, custom_objects={"PreActBlock": PreActBlock})

            data = {}
            for split in ["train", "val", "test"]:
                idxs = np.asarray(folds_dict[fold][split], dtype=np.int64)

                ds_ms, _ = get_dataset_in_place(idxs, "ms", split, return_country=True)
                ds_nl, _ = get_dataset_in_place(idxs, "nl", split, return_country=True)

                Xms, y, c = get_features(m_ms, ds_ms)
                Xnl, _, _ = get_features(m_nl, ds_nl)

                data[split] = (np.concatenate([Xms, Xnl], axis=1), y, c)

            X_tr, y_tr, _     = data["train"]
            X_val, y_val, _   = data["val"]
            X_te,  y_te, c_te = data["test"]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s  = scaler.transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_te_s  = scaler.transform(X_te)

            best_alpha, best_val_r2 = None, -1e9
            for a in RIDGE_ALPHAS:
                pred_val = Ridge(alpha=a).fit(X_tr_s, y_tr).predict(X_val_s)
                r2 = r2_score(y_val, pred_val)
                if r2 > best_val_r2:
                    best_val_r2 = r2
                    best_alpha = a

            X_tv = np.vstack([X_tr, X_val])
            y_tv = np.concatenate([y_tr, y_val])

            scaler_tv = StandardScaler().fit(X_tv)
            ridge = Ridge(alpha=best_alpha).fit(scaler_tv.transform(X_tv), y_tv)
            preds = ridge.predict(scaler_tv.transform(X_te))

            d_ms = Xms.shape[1]
            Xtv_ms, Xtv_nl = X_tv[:, :d_ms], X_tv[:, d_ms:]
            Xte_ms, Xte_nl = X_te[:, :d_ms], X_te[:, d_ms:]
            sc_ms = StandardScaler().fit(Xtv_ms)
            sc_nl = StandardScaler().fit(Xtv_nl)
            a_ms = max(RIDGE_ALPHAS, key=lambda a: r2_score(y_val, Ridge(alpha=a).fit(sc_ms.transform(X_tr[:, :d_ms]), y_tr).predict(sc_ms.transform(X_val[:, :d_ms]))))
            a_nl = max(RIDGE_ALPHAS, key=lambda a: r2_score(y_val, Ridge(alpha=a).fit(sc_nl.transform(X_tr[:, d_ms:]), y_tr).predict(sc_nl.transform(X_val[:, d_ms:]))))
            yhat_ms = Ridge(alpha=a_ms).fit(sc_ms.transform(Xtv_ms), y_tv).predict(sc_ms.transform(Xte_ms))
            yhat_nl = Ridge(alpha=a_nl).fit(sc_nl.transform(Xtv_nl), y_tv).predict(sc_nl.transform(Xte_nl))

            test_r2  = r2_score(y_te, preds)
            test_mse = mean_squared_error(y_te, preds)

            print(f"  Ridge alpha={best_alpha}  val_r2={best_val_r2:.4f}")
            print(f"  TEST: R2={test_r2:.4f}  MSE(utils)={test_mse:.4f}")

            fusion_rows.append(pd.DataFrame({
                "scheme": scheme,
                "fold": fold,
                "country": c_te,
                "y_true": y_te,
                "y_pred": preds,
                "yhat_ms": yhat_ms,
                "yhat_nl": yhat_nl,
                "test_r2_fold": test_r2,
                "test_mse_fold": test_mse,
                "ridge_alpha": best_alpha,
                "ms_ckpt": ms_row.ckpt_path,
                "nl_ckpt": nl_row.ckpt_path,
            }))

if fusion_rows:
    df_fusion = pd.concat(fusion_rows, ignore_index=True)
    df_fusion.to_csv(fusion_out, index=False)
    print(f"\nFusion complete. Saved: {fusion_out}")
else:
    print("No fusion rows produced (need BOTH ms and nl trained).")

import matplotlib.pyplot as plt

LEARNING_FRACS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

LEARNING_MAX_EPOCHS = 150
LEARNING_ES_PATIENCE = 20

def _subsample_indices(indices, frac, seed=123):
    indices = np.asarray(indices, dtype=np.int64)
    if frac >= 0.999:
        return indices
    n = len(indices)
    k = max(1, int(round(frac * n)))
    rng = np.random.default_rng(seed)
    pick = rng.choice(n, size=k, replace=False)
    return indices[pick]

def _eval_r2_model(model, ds, n):
    yt, yp = [], []
    for x, y in ds.take(math.ceil(n / BATCH_SIZE)):
        pred = model(x, training=False).numpy().reshape(-1)
        yt.extend(y.numpy().reshape(-1).tolist())
        yp.extend(pred.tolist())
    yt = np.asarray(yt[:n], dtype=np.float32)
    yp = np.asarray(yp[:n], dtype=np.float32)
    return float(r2_score(yt, yp))

def run_learning_curve_for_variant(
    scheme="incountry",
    fold="A",
    variant="ms",
    fracs=LEARNING_FRACS,
    seed=123,
    max_epochs=LEARNING_MAX_EPOCHS,
    es_patience=LEARNING_ES_PATIENCE,
):
    df_log = pd.read_csv(RES_CSV) if os.path.exists(RES_CSV) else pd.DataFrame(results)
    if df_log.empty:
        raise ValueError("No training log found. Train at least one model first.")

    df_sub = df_log[(df_log.scheme == scheme) & (df_log.fold == fold) & (df_log.variant == variant)]
    if df_sub.empty:
        raise ValueError(f"No entries found in log for {scheme}/{fold}/{variant}.")

    best = df_sub.sort_values("val_r2", ascending=False).iloc[0]
    best_lr, best_l2 = float(best.lr), float(best.l2)

    print(f"\n[LEARNING CURVE] {scheme}/{fold}/{variant} best params:")
    print(f"  lr={best_lr:.2e}  l2={best_l2:.2e}  (best logged val_r2={best.val_r2:.4f})")

    folds_dict = INCOUNTRY_FOLDS if scheme == "incountry" else OOC_FOLDS
    base_train = np.asarray(folds_dict[fold]["train"], dtype=np.int64)
    base_val   = np.asarray(folds_dict[fold]["val"],   dtype=np.int64)
    base_test  = np.asarray(folds_dict[fold]["test"],  dtype=np.int64)

    out_rows = []

    for frac in fracs:
        train_idxs = _subsample_indices(base_train, frac, seed=seed)

        ds_train, n_train = get_dataset_in_place(train_idxs, variant, "train", return_country=False)
        ds_val,   n_val   = get_dataset_in_place(base_val,   variant, "val",   return_country=False)
        ds_test,  n_test  = get_dataset_in_place(base_test,  variant, "test",  return_country=False)

        if n_train == 0:
            print(f"  frac={frac:.2f} -> n_train=0 (skip)")
            continue

        tf.keras.backend.clear_session()
        gc.collect()

        model = make_mil_model(variant, lr=best_lr, l2=best_l2)

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_mse",
                patience=es_patience,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.TerminateOnNaN(),
        ]

        print(f"  Training frac={frac:.2f}  n_train={n_train} ...")
        model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=max_epochs,
            steps_per_epoch=math.ceil(n_train / BATCH_SIZE),
            validation_steps=max(1, math.ceil(n_val / BATCH_SIZE)),
            callbacks=cb,
            verbose=0,
        )

        val_r2  = _eval_r2_model(model, ds_val,  n_val)
        test_r2 = _eval_r2_model(model, ds_test, n_test)

        out_rows.append({
            "scheme": scheme,
            "fold": fold,
            "variant": variant,
            "frac_train": float(frac),
            "n_train": int(n_train),
            "val_r2": float(val_r2),
            "test_r2": float(test_r2),
            "lr": best_lr,
            "l2": best_l2,
        })

        print(f"    -> val_r2={val_r2:.4f}  test_r2={test_r2:.4f}")

    return pd.DataFrame(out_rows)

def plot_learning_curves(df_list, title=None):
    plt.figure(figsize=(7.8, 5.0))

    for dfc in df_list:
        if dfc is None or dfc.empty:
            continue
        dfc = dfc.sort_values("frac_train")
        v = dfc["variant"].iloc[0].upper()
        plt.plot(dfc["frac_train"], dfc["test_r2"], marker="o", label=f"{v} (test R²)")

    plt.xlabel("% of data used in training")
    plt.ylabel("R² on test")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if title is None:
        title = "Learning curve: R² vs % of train data"
    plt.title(title)
    plt.show()

df_curve_ms = run_learning_curve_for_variant(scheme="incountry", fold="A", variant="ms")
df_curve_nl = run_learning_curve_for_variant(scheme="incountry", fold="A", variant="nl")
plot_learning_curves([df_curve_ms, df_curve_nl])

import geopandas as gpd
import pycountry

def _safe_country2_to_iso3(cc2):
    cc2 = str(cc2).upper().strip()[:2]
    try:
        c = pycountry.countries.get(alpha_2=cc2)
        return c.alpha_3 if c else None
    except Exception:
        return None

def compute_country_r2_from_predictions(df_pred, country_col="country", y_col="y_true", pred_col="y_pred", min_n=25):
    df = df_pred.copy()

    def _to_cc2(x):
        if isinstance(x, (bytes, bytearray, np.bytes_)):
            x = x.decode("utf-8", errors="ignore")
        return str(x).upper().strip()[:2]

    df["cc2"] = df[country_col].apply(_to_cc2)
    df = df[np.isfinite(df[y_col]) & np.isfinite(df[pred_col])].copy()

    rows = []
    for cc2, g in df.groupby("cc2"):
        n = len(g)
        if n < min_n:
            rows.append({"cc2": cc2, "n": n, "r2": np.nan})
        else:
            rows.append({"cc2": cc2, "n": n, "r2": float(r2_score(g[y_col].values, g[pred_col].values))})

    out = pd.DataFrame(rows).sort_values("r2", ascending=True)
    out["iso_a3"] = out["cc2"].apply(_safe_country2_to_iso3)
    return out

def plot_africa_r2_map(country_r2_df, value_col="r2", title="Per-country R²", vmin=0.50, vmax=0.85):
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    africa = world[world["continent"] == "Africa"].copy()

    merged = africa.merge(country_r2_df, how="left", left_on="iso_a3", right_on="iso_a3")

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    merged.plot(
        column=value_col,
        ax=ax,
        cmap="RdYlGn",
        legend=True,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "lightgrey", "label": "NA"},
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_title(title)
    ax.set_axis_off()
    plt.show()

if os.path.exists(fusion_out):
     df_pred = pd.read_csv(fusion_out)
     df_country_r2 = compute_country_r2_from_predictions(df_pred, min_n=25)
     display(df_country_r2)
     plot_africa_r2_map(
         df_country_r2,
         title="Satellite-predicted asset wealth — Mean test R² by country",
         vmin=0.50,
         vmax=0.85
     )
