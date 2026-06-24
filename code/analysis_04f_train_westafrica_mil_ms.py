import os, glob, json, pickle, random, math, shutil, gc, re
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
import matplotlib.pyplot as plt

keras.config.enable_unsafe_deserialization()

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

DRIVE_BASE    = "/content/drive/My Drive/TFRecords_MIL"
DRIVE_TF_ROOT = DRIVE_BASE
DRIVE_DD_BASE = os.path.join(DRIVE_BASE, "Data Distribution_MIL")

COMBINED_CSV  = os.path.join(DRIVE_DD_BASE, "dhs_combined_df.csv")
INC_FOLDS_PKL = os.path.join(DRIVE_DD_BASE, "dhs_incountry_folds.pkl")
OOC_FOLDS_PKL = os.path.join(DRIVE_DD_BASE, "dhs_ooc_folds.pkl")
GLOBAL_STATS_JSON = os.path.join(DRIVE_BASE, "Data Validation_MIL", "band_stats_summary.json")

SSD_ROOT      = "/content/local_cache"
LOCAL_RAW_DIR = os.path.join(SSD_ROOT, "raw_tfrecords")
SSD_META_DIR  = os.path.join(SSD_ROOT, "metadata")
SSD_CKPT_DIR  = "/content/ckpts_tmp"

ROOT_OUT = os.path.join(DRIVE_DD_BASE, "cnn_results_mil_west_africa_grid_per_variant")
CKPT_DIR = os.path.join(ROOT_OUT, "checkpoints")
LOGS_DIR = os.path.join(ROOT_OUT, "per_search_logs")

for d in [LOCAL_RAW_DIR, SSD_META_DIR, SSD_CKPT_DIR, ROOT_OUT, CKPT_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

EXPECTED_K = 9
EXPECTED_H = 224

TFRECORD_BAND_ORDER = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "TEMP1", "NIGHTLIGHTS"]
BANDS_MS = ["RED", "GREEN", "BLUE", "SWIR1", "SWIR2", "TEMP1", "NIR"]
BANDS_NL = ["NIGHTLIGHTS"]

CLIP_NEGATIVE = {"BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "NIGHTLIGHTS"}

WEST_AFRICA_CODES = {"BJ","SN","GM","MR","GH","CI","BF","ML","NG","SL","GN"}

BATCH_SIZE = 64
EPOCHS_INCOUNTRY = 150
EPOCHS_OOC       = 200

EARLY_STOPPING_PATIENCE = 20
FOLDS_TO_RUN   = ["A","C"]
SCHEMES_TO_RUN = ["incountry"]
VARIANTS_TO_RUN = ["ms", "nl"]

PARAMS_BY_FOLD = {
    "A": {
        "ms": [(3e-4, 0.0)],
        "nl": [(1e-4, 3e-5)],
    },
    "C": {
        "ms": [(3e-4, 0.0)],
        "nl": [(3e-5, 1e-4)],
    }
}

CACHE_VAL_TEST_IN_RAM   = True
CACHE_TRAIN_IN_RAM      = True
TRAIN_CACHE_SAFETY_FRAC = 0.60

print("ROOT_OUT:", ROOT_OUT)
print("CKPT_DIR:", CKPT_DIR)
print("Metadata Source:", COMBINED_CSV)

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

print("\n--- TFRECORD SYNC + INDEX ---")

print("Checking local TFRecords...")
all_drive_files = sorted(glob.glob(os.path.join(DRIVE_TF_ROOT, "*.tfrecord")))
if len(all_drive_files) == 0:
    raise FileNotFoundError(f"No TFRecords found in: {DRIVE_TF_ROOT}")

target_drive_paths = [f for f in all_drive_files if any(c in os.path.basename(f).upper() for c in WEST_AFRICA_CODES)]
if not target_drive_paths:
    raise FileNotFoundError("No West Africa TFRecords found (filenames didn't match codes).")

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

INDEX_CACHE = os.path.join(SSD_META_DIR, "local_index.pkl")
if os.path.exists(INDEX_CACHE):
    print("Loading index cache...")
    with open(INDEX_CACHE, "rb") as f:
        tfrecord_index = pickle.load(f)
else:
    print("Indexing local files...")
    local_files = sorted(glob.glob(os.path.join(LOCAL_RAW_DIR, "*.tfrecord")))
    tfrecord_index = {}

    def index_worker(path):
        local_idx = {}
        idx_spec = {
            "country": tf.io.FixedLenFeature([], tf.string),
            "year":    tf.io.FixedLenFeature([], tf.int64),
            "cluster": tf.io.FixedLenFeature([], tf.int64),
        }
        try:
            for i, raw in enumerate(tf.data.TFRecordDataset(path)):
                ex = tf.io.parse_single_example(raw, idx_spec)
                cc = ex["country"].numpy().decode("utf-8").upper()[:2]
                yy = int(ex["year"].numpy())
                cl = int(ex["cluster"].numpy())
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

def estimate_log1p_ntl_stats(sample_keys, max_examples=200, seed=123):
    rng = np.random.default_rng(seed)
    keys = list(sample_keys)
    if len(keys) == 0:
        return 0.0, 1.0

    if len(keys) > max_examples:
        keys = rng.choice(keys, size=max_examples, replace=False).tolist()

    n = 0
    mean = 0.0
    M2 = 0.0

    spec = {
        "K":           tf.io.FixedLenFeature([], tf.int64),
        "patch_size":  tf.io.FixedLenFeature([], tf.int64),
        "NIGHTLIGHTS": tf.io.VarLenFeature(tf.float32),
    }

    for k in tqdm(keys, desc="[NL] Estimating log1p(NTL) mean/std", leave=False):
        path, _ = tfrecord_index[k]

        try:
            for raw in tf.data.TFRecordDataset(path).take(999999):
                e = tf.io.parse_single_example(raw, spec)
                K = int(e["K"].numpy())
                H = int(e["patch_size"].numpy())
                if K != EXPECTED_K or H != EXPECTED_H:
                    continue

                arr = tf.reshape(tf.sparse.to_dense(e["NIGHTLIGHTS"]), [EXPECTED_K, EXPECTED_H, EXPECTED_H])
                arr = tf.maximum(arr, 0.0)
                arr = tf.where(tf.math.is_finite(arr), arr, tf.zeros_like(arr))
                arr = tf.math.log1p(arr)
                flat = tf.reshape(arr, [-1]).numpy().astype(np.float64)

                for x in flat:
                    n += 1
                    delta = x - mean
                    mean += delta / n
                    delta2 = x - mean
                    M2 += delta * delta2
                break
        except Exception:
            continue

    if n < 2:
        return 0.0, 1.0

    var = M2 / (n - 1)
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), std

all_fold_keys = []
for fold in FOLDS_TO_RUN:
    tr_idxs = INCOUNTRY_FOLDS[fold]["train"]
    raw_keys = [FULL_IDX_TO_KEY[i] for i in tr_idxs if i in FULL_IDX_TO_KEY]
    raw_keys = [k for k in raw_keys if (k in tfrecord_index) and (k in KEY_TO_LABEL)]
    all_fold_keys.extend(raw_keys)

all_fold_keys = list(dict.fromkeys(all_fold_keys))

NL_MEAN_LOG1P, NL_STD_LOG1P = estimate_log1p_ntl_stats(all_fold_keys, max_examples=120, seed=SEED)
print(f"[NL] Using log1p(NTL) stats: mean={NL_MEAN_LOG1P:.6f}, std={NL_STD_LOG1P:.6f}")

def _estimate_cache_bytes(n_examples, variant):

    c = 7 if variant == "ms" else 1
    bytes_per = 2 if variant == "ms" else 4
    x_bytes = n_examples * EXPECTED_K * EXPECTED_H * EXPECTED_H * c * bytes_per
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

    if variant == "nl":
        means = tf.constant([NL_MEAN_LOG1P], dtype=tf.float32)
        stds  = tf.constant([NL_STD_LOG1P],  dtype=tf.float32)
    else:
        means = tf.constant([BAND_MEAN[b] for b in bands], dtype=tf.float32)
        stds  = tf.constant([BAND_STD[b]  for b in bands], dtype=tf.float32)

    def make_key(country_str, year_i64, cluster_i64):
        c = tf.strings.upper(country_str)
        c = tf.strings.substr(c, 0, 2)
        return tf.strings.join([c, tf.as_string(year_i64), tf.as_string(cluster_i64)], separator="|")

    def is_valid_key(raw):
        s = {
            "country": tf.io.FixedLenFeature([], tf.string),
            "year":    tf.io.FixedLenFeature([], tf.int64),
            "cluster": tf.io.FixedLenFeature([], tf.int64),
        }
        e = tf.io.parse_single_example(raw, s)
        k = make_key(e["country"], e["year"], e["cluster"])
        return valid_table.lookup(k) == 1

    def parse(raw):
        spec = {
            "K":          tf.io.FixedLenFeature([], tf.int64),
            "patch_size": tf.io.FixedLenFeature([], tf.int64),
            "country":    tf.io.FixedLenFeature([], tf.string),
            "year":       tf.io.FixedLenFeature([], tf.int64),
            "cluster":    tf.io.FixedLenFeature([], tf.int64),
        }
        for b in TFRECORD_BAND_ORDER:
            spec[b] = tf.io.VarLenFeature(tf.float32)

        e = tf.io.parse_single_example(raw, spec)

        K = tf.cast(e["K"], tf.int32)
        H = tf.cast(e["patch_size"], tf.int32)
        tf.debugging.assert_equal(K, EXPECTED_K, message="Unexpected K in TFRecord")
        tf.debugging.assert_equal(H, EXPECTED_H, message="Unexpected patch_size in TFRecord")

        k = make_key(e["country"], e["year"], e["cluster"])
        y = tf.cast(label_table.lookup(k), tf.float32)
        y = tf.reshape(y, [1])

        imgs = []
        for b in bands:
            arr = tf.reshape(tf.sparse.to_dense(e[b]), [EXPECTED_K, EXPECTED_H, EXPECTED_H])

            if b in CLIP_NEGATIVE:
                arr = tf.maximum(arr, 0.0)

            arr = tf.where(tf.math.is_finite(arr), arr, tf.zeros_like(arr))

            if b == "NIGHTLIGHTS":
                arr = tf.math.log1p(arr)

            imgs.append(arr)

        X = tf.stack(imgs, axis=-1)
        X = tf.ensure_shape(X, [EXPECTED_K, EXPECTED_H, EXPECTED_H, C])

        if variant == "nl":

            X = X / (stds + 1e-8)
            X = tf.clip_by_value(X, 0.0, 50.0)
            X = tf.cast(X, tf.float32)
        else:
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

        if variant == "ms":
            def aug(x, y):
                if tf.random.uniform([]) > 0.5:
                    x = tf.reverse(x, axis=[2])
                if tf.random.uniform([]) > 0.5:
                    x = tf.reverse(x, axis=[1])

                x32 = tf.cast(x, tf.float32)
                noise = tf.random.uniform([], -0.2, 0.2, dtype=tf.float32)
                x32 = x32 + noise
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

@register_keras_serializable()
class AttentionPool(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        td, attn_w = inputs
        return tf.reduce_sum(td * attn_w, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

@register_keras_serializable()
class PosBiasAdd(layers.Layer):
    def __init__(self, k=9, init_zero=True, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)
        self.init_zero = bool(init_zero)

    def build(self, input_shape):
        init = np.zeros((self.k, 1), dtype=np.float32) if self.init_zero else np.random.normal(0, 0.01, (self.k, 1)).astype(np.float32)
        self.pos_bias = self.add_weight(
            name="pos_bias",
            shape=(self.k, 1),
            initializer=keras.initializers.Constant(init),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, z):
        b = tf.reshape(self.pos_bias, [1, self.k, 1])
        return z + b

    def get_config(self):
        return {**super().get_config(), "k": self.k, "init_zero": self.init_zero}

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
        input_shape=(EXPECTED_H, EXPECTED_H, 3),
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
            continue

        rw = rgb_layer.get_weights()
        lw = layer.get_weights()
        if len(rw) != len(lw) or any(a.shape != b.shape for a, b in zip(rw, lw)):
            continue
        layer.set_weights(rw)

    ms_blocks  = [l for l in backbone.layers if isinstance(l, PreActBlock)]
    rgb_blocks = [l for l in rgb_backbone.layers if isinstance(l, PreActBlock)]
    if len(ms_blocks) != len(rgb_blocks):
        raise ValueError(f"PreActBlock count mismatch: target={len(ms_blocks)} rgb={len(rgb_blocks)}")

    for b_tgt, b_rgb in zip(ms_blocks, rgb_blocks):
        b_tgt.set_weights(b_rgb.get_weights())

    if verbose:
        print("[PRETRAIN] Finished initializing backbone:", backbone.name)

def make_mil_model(variant, lr, l2):
    l2_val = float(l2) if l2 is not None else 0.0
    weight_reg = keras.regularizers.l2(l2_val) if (l2_val > 0) else None

    bands = len(BANDS_MS) if variant == "ms" else len(BANDS_NL)
    mil_inp = keras.Input((EXPECTED_K, EXPECTED_H, EXPECTED_H, bands))

    backbone = _make_resnet18_backbone_generic(
        input_shape=(EXPECTED_H, EXPECTED_H, bands),
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

    td = layers.TimeDistributed(backbone)(mil_inp)

    attn_logits = layers.TimeDistributed(
        layers.Dense(1, kernel_initializer="glorot_uniform"),
        name="attn_logits"
    )(td)

    if variant == "nl":
        attn_logits = PosBiasAdd(k=EXPECTED_K, init_zero=True, name="nl_pos_bias_add")(attn_logits)

    attn_w = layers.Softmax(axis=1, name="attn_w")(attn_logits)

    agg = AttentionPool(name="attn_pool")([td, attn_w])

    out = layers.Dense(1, kernel_regularizer=weight_reg, dtype="float32")(agg)
    model = keras.Model(mil_inp, out, name=f"mil_{variant}")

    opt = keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)

    loss_fn = tf.keras.losses.Huber(delta=1.0) if variant == "nl" else "mse"

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[keras.metrics.MeanSquaredError(name="mse"), "mae"],
        jit_compile=False
    )
    return model

print("\n--- STARTING GRID SEARCH TRAINING (WITH CKPT LOG REBUILD) ---")

RES_CSV = os.path.join(ROOT_OUT, "cnn_training_log.csv")
CKPT_GLOB = os.path.join(CKPT_DIR, "*.keras")

CKPT_RE = re.compile(
    r"^(?P<scheme>incountry|ooc)_(?P<fold>[A-Z])_(?P<variant>ms|nl)_lr(?P<lr>[\deE\+\-\.]+)_l2(?P<l2>[\deE\+\-\.]+)\.keras$"
)

def _parse_ckpt_filename(fname):
    m = CKPT_RE.match(fname)
    if not m:
        return None
    d = m.groupdict()
    try:
        d["lr"] = float(d["lr"])
        d["l2"] = float(d["l2"])
    except Exception:
        return None
    return d

def _append_row_csv(path, row_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_new = pd.DataFrame([row_dict])
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)

def rebuild_log_from_checkpoints():
    ckpts = sorted(glob.glob(CKPT_GLOB))
    print(f"[REBUILD] Found {len(ckpts)} checkpoint files in CKPT_DIR")

    rows = []
    bad = 0
    for ckpt_path in ckpts:
        fname = os.path.basename(ckpt_path)
        info = _parse_ckpt_filename(fname)
        if info is None:
            bad += 1
            continue

        rows.append({
            "scheme": info["scheme"],
            "fold": info["fold"],
            "variant": info["variant"],
            "lr": info["lr"],
            "l2": info["l2"],
            "ckpt_path": ckpt_path,
            "val_r2": np.nan,
            "test_r2": np.nan,
        })

    print(f"[REBUILD] Parsed {len(rows)} checkpoint(s). Skipped {bad} (name mismatch).")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["scheme", "fold", "variant", "lr", "l2"])
    df.to_csv(RES_CSV, index=False)
    print(f"[REBUILD] Wrote fresh global log: {RES_CSV}  (rows={len(df)})")
    return df

df_rebuilt = rebuild_log_from_checkpoints()
results = df_rebuilt.to_dict("records") if not df_rebuilt.empty else []

def _already_done(ckpt_path):
    return os.path.exists(ckpt_path)

def _grid_for_fold_variant(fold, variant):
    if fold in PARAMS_BY_FOLD and variant in PARAMS_BY_FOLD[fold]:
        return PARAMS_BY_FOLD[fold][variant]

    return [(3e-4, 0.0)] if variant == "ms" else [(1e-4, 3e-5)]

for scheme in SCHEMES_TO_RUN:
    folds_dict = INCOUNTRY_FOLDS if scheme == "incountry" else OOC_FOLDS
    epochs = EPOCHS_INCOUNTRY if scheme == "incountry" else EPOCHS_OOC
    print(f"\n>>> SCHEME: {scheme.upper()} <<<")

    for fold in FOLDS_TO_RUN:
        for variant in VARIANTS_TO_RUN:

            PER_SEARCH_CSV = os.path.join(LOGS_DIR, f"log_{scheme}_{fold}_{variant}.csv")

            ds, counts = {}, {}
            for split in ["train", "val", "test"]:
                idxs = np.asarray(folds_dict[fold][split], dtype=np.int64)
                ds[split], counts[split] = get_dataset_in_place(idxs, variant, split, return_country=False)

            if counts["train"] == 0:
                print(f"No train data for {scheme}/{fold}/{variant}. Skipping.")
                continue

            for lr, l2 in _grid_for_fold_variant(fold, variant):
                tag = f"{scheme}_{fold}_{variant}_lr{lr:.0e}_l2{l2:.0e}"
                ckpt = os.path.join(CKPT_DIR, f"{tag}.keras")

                if _already_done(ckpt):
                    print(f"Skipping {tag} (Checkpoint exists)")
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
                        validation_steps=max(1, math.ceil(counts["val"] / BATCH_SIZE)),
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

                    _append_row_csv(PER_SEARCH_CSV, row)

                    print("Saved row:", row)
                    print("  Global log:", RES_CSV)
                    print("  Per-search log:", PER_SEARCH_CSV)

                except Exception as e:
                    print(f"Error {tag}: {e}")
                    import traceback
                    traceback.print_exc()

print("\nTraining loop finished. Global log saved at:", RES_CSV)
print("Per-search logs in:", LOGS_DIR)

print("\n--- RIDGE FUSION ---")

df_retrain = pd.DataFrame(results)
if df_retrain.empty and os.path.exists(RES_CSV):
    df_retrain = pd.read_csv(RES_CSV)

fusion_out = os.path.join(ROOT_OUT, "ridge_fusion_scaled_predictions.csv")
RIDGE_ALPHAS = [10.0 ** p for p in range(-4, 5)]

def get_features(model, ds):
    feat_layer_name = "attn_pool"
    feat_model = keras.Model(model.input, model.get_layer(feat_layer_name).output)

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

        for fold in sorted(set(df_retrain[df_retrain["scheme"] == scheme]["fold"].unique())):
            sub = df_retrain[(df_retrain.scheme == scheme) & (df_retrain.fold == fold)]

            if not (("ms" in set(sub.variant)) and ("nl" in set(sub.variant))):
                continue

            ms_row = sub[sub.variant == "ms"].sort_values("val_r2", ascending=False).iloc[0]
            nl_row = sub[sub.variant == "nl"].sort_values("val_r2", ascending=False).iloc[0]

            print(f"\n>>> FUSION: {scheme}/{fold} <<<")
            print(f"  ms: val_r2={ms_row.val_r2}  ckpt={os.path.basename(ms_row.ckpt_path)}")
            print(f"  nl: val_r2={nl_row.val_r2}  ckpt={os.path.basename(nl_row.ckpt_path)}")

            tf.keras.backend.clear_session(); gc.collect()

            custom_objs = {
                "PreActBlock": PreActBlock,
                "AttentionPool": AttentionPool,
                "PosBiasAdd": PosBiasAdd,
            }

            try:
                m_ms = keras.models.load_model(
                    ms_row.ckpt_path,
                    custom_objects=custom_objs,
                    compile=False,
                    safe_mode=False,
                )
            except Exception as e:
                print(f"FAILED to load MS model: {e}")
                continue

            tf.keras.backend.clear_session(); gc.collect()

            try:
                m_nl = keras.models.load_model(
                    nl_row.ckpt_path,
                    custom_objects=custom_objs,
                    compile=False,
                    safe_mode=False,
                )
            except Exception as e:
                print(f"FAILED to load NL model: {e}")
                continue

            data = {}
            for split in ["train", "val", "test"]:
                idxs = np.asarray(folds_dict[fold][split], dtype=np.int64)

                ds_ms, _ = get_dataset_in_place(idxs, "ms", split, return_country=True)
                ds_nl, _ = get_dataset_in_place(idxs, "nl", split, return_country=True)

                Xms, y, c = get_features(m_ms, ds_ms)
                Xnl, _, _ = get_features(m_nl, ds_nl)

                if Xms.size == 0 or Xnl.size == 0:
                    print(f"Empty features in {scheme}/{fold}/{split} -> skipping fusion for this fold.")
                    data = None
                    break

                data[split] = (np.concatenate([Xms, Xnl], axis=1), y, c)

            if data is None:
                continue

            X_tr, y_tr, _     = data["train"]
            X_val, y_val, _   = data["val"]
            X_te,  y_te, c_te = data["test"]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s  = scaler.transform(X_tr)
            X_val_s = scaler.transform(X_val)

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
    df_fusion = None
    print("No fusion rows produced (missing ms/nl pairs OR model load failed).")

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
        v = str(dfc["variant"].iloc[0]).upper()
        plt.plot(dfc["frac_train"], dfc["test_r2"], marker="o", label=f"{v} (test R²)")
    plt.xlabel("% of data used in training")
    plt.ylabel("R² on test")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if title is None:
        title = "Learning curve: R² vs % of train data"
    plt.title(title)
    plt.show()

if df_fusion is not None and not df_fusion.empty:
    import geopandas as gpd
    import pycountry

    df_map = df_fusion.copy()
    df_map["cc2"] = [str(x).split("'")[1] if "b'" in str(x) else str(x) for x in df_map["country"]]
    df_map["cc2"] = df_map["cc2"].str.upper().str.slice(0, 2)

    stats = []

    for c, g in df_map.groupby("cc2"):
        if len(g) >= 10:
            stats.append({"cc2": c, "r2": r2_score(g.y_true, g.y_pred)})

    dmap = pd.DataFrame(stats)

    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    try:
        world = gpd.read_file(url)
        africa = world[world["CONTINENT"] == "Africa"]

        def get_iso3(cc2):
            try:
                return pycountry.countries.get(alpha_2=cc2).alpha_3
            except:
                return None

        dmap["iso3"] = dmap["cc2"].apply(get_iso3)

        merged = africa.merge(dmap, left_on="ISO_A3", right_on="iso3", how="left")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        merged.plot(
            column="r2",
            ax=ax,
            cmap="RdYlGn",
            legend=True,
            missing_kwds={"color": "lightgrey"},
            edgecolor="black",
            vmin=0.2,
            vmax=0.8
        )
        plt.title("West Africa Wealth Prediction R² (Folds A+C)")
        plt.show()

    except Exception as e:
        print(f"Map error: {e}")
        display(dmap)
else:
    print("No fusion results.")
