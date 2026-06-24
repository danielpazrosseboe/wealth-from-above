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

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    import psutil
except ImportError:
    import psutil

from google.colab import drive
if not os.path.exists("/content/drive"):
    drive.mount("/content/drive")

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

SEED = 123
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DRIVE_BASE    = "/content/drive/My Drive/TFRecords_MIL"
DRIVE_TF_ROOT = DRIVE_BASE
DRIVE_DD_BASE = os.path.join(DRIVE_BASE, "Data Distribution_MIL")

SSD_ROOT      = "/content/local_cache"
LOCAL_RAW_DIR = os.path.join(SSD_ROOT, "raw_tfrecords")
SSD_META_DIR  = os.path.join(SSD_ROOT, "metadata")
SSD_CKPT_DIR  = "/content/ckpts_tmp"

ROOT_OUT = os.path.join(DRIVE_DD_BASE, "cnn_results_mil_west_africa_grid_per_variant")
CKPT_DIR = os.path.join(ROOT_OUT, "checkpoints")

for d in [LOCAL_RAW_DIR, SSD_META_DIR, SSD_CKPT_DIR, ROOT_OUT, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

EXPECTED_K = 9
EXPECTED_H = 224

TFRECORD_BAND_ORDER = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "TEMP1", "NIGHTLIGHTS"]
BANDS_MS = ["RED", "GREEN", "BLUE", "SWIR1", "SWIR2", "TEMP1", "NIR"]
BANDS_NL = ["NIGHTLIGHTS"]

WEST_AFRICA_CODES = {"BJ","BF","CV","CI","GM","GH","GN","GW","LR","ML","MR","NE","NG","SN","SL","TG"}

BATCH_SIZE = 64
EPOCHS_INCOUNTRY = 150
EPOCHS_OOC       = 200
EARLY_STOPPING_PATIENCE = 30

FOLDS_TO_RUN   = ["A", "B", "C", "D", "E"]
SCHEMES_TO_RUN = ["incountry", "ooc"]
VARIANTS_TO_RUN = ["nl"]

PARAM_GRID_MS = [
    (1e-3, 0.0),
    (3e-4, 0.0),
]

PARAM_GRID_NL = [
    (1e-3, 0.01),
    (3e-4, 0.01),
]

CACHE_VAL_TEST_IN_RAM = True
CACHE_TRAIN_IN_RAM    = True
TRAIN_CACHE_SAFETY_FRAC = 0.60

PRETRAIN_DRIVE = "/content/drive/My Drive/TFRecords_MIL/resnet18_preact_imagenet_rgb_from_npz.weights.h5"
PRETRAIN_SSD_DIR = os.path.join(SSD_ROOT, "pretrained")
PRETRAIN_SSD = os.path.join(PRETRAIN_SSD_DIR, "resnet18_preact_imagenet_rgb_from_npz.weights.h5")
os.makedirs(PRETRAIN_SSD_DIR, exist_ok=True)

if not os.path.exists(PRETRAIN_SSD):
    if os.path.exists(PRETRAIN_DRIVE):
        print("[PRETRAIN] Copying resnet18_preact_imagenet_rgb_from_npz.weights.h5 to SSD...")
        shutil.copy(PRETRAIN_DRIVE, PRETRAIN_SSD)
    else:
        raise FileNotFoundError(f"Missing pretrained weights on Drive: {PRETRAIN_DRIVE}")

print("[PRETRAIN] SSD weights path:", PRETRAIN_SSD)

print("\n--- 1. METADATA & FILE SYNC ---")

for f in ["dhs_combined_df.csv", "dhs_incountry_folds.pkl", "dhs_ooc_folds.pkl"]:
    src = os.path.join(DRIVE_DD_BASE, f)
    dst = os.path.join(SSD_META_DIR, f)
    if not os.path.exists(dst) and os.path.exists(src):
        shutil.copy(src, dst)

stats_src = os.path.join(DRIVE_BASE, "Data Validation_MIL", "band_stats_summary.json")
stats_dst = os.path.join(SSD_META_DIR, "band_stats_summary.json")
if not os.path.exists(stats_dst) and os.path.exists(stats_src):
    shutil.copy(stats_src, stats_dst)

df_meta = pd.read_csv(os.path.join(SSD_META_DIR, "dhs_combined_df.csv"))

def _pick(df, cands):
    return next(c for c in cands if c in df.columns)

C_COUNTRY = _pick(df_meta, ["country", "CC"])
C_CLUSTER = _pick(df_meta, ["cluster", "clu"])
C_YEAR    = _pick(df_meta, ["year", "survey_year"])

df_meta[C_COUNTRY] = df_meta[C_COUNTRY].astype(str).str.upper().str.strip().str[:2]
df_meta = df_meta[df_meta[C_COUNTRY].isin(WEST_AFRICA_CODES)].reset_index(drop=True)
df_meta["_key"] = df_meta[C_COUNTRY] + "|" + df_meta[C_YEAR].astype(str) + "|" + df_meta[C_CLUSTER].astype(str)

KEY_TO_LABEL = dict(zip(df_meta["_key"], df_meta["wealthpooled"].astype(np.float32)))

with open(os.path.join(SSD_META_DIR, "dhs_incountry_folds.pkl"), "rb") as f:
    INCOUNTRY_FOLDS = pickle.load(f)
with open(os.path.join(SSD_META_DIR, "dhs_ooc_folds.pkl"), "rb") as f:
    OOC_FOLDS = pickle.load(f)

TEMP_SCALE  = 0.00341802
TEMP_OFFSET = 149.0

with open(stats_dst, "r") as f:
    band_stats = json.load(f)

example_band = TFRECORD_BAND_ORDER[0]
MEAN_KEY = next(k for k in band_stats[example_band].keys() if "mean" in k.lower())
STD_KEY  = next(k for k in band_stats[example_band].keys() if ("std" in k.lower() or "stdev" in k.lower()))

BAND_MEAN = {b: float(band_stats[b][MEAN_KEY]) for b in TFRECORD_BAND_ORDER}
BAND_STD  = {b: float(band_stats[b][STD_KEY])  for b in TFRECORD_BAND_ORDER}

print("[NORM] Loaded keys:", MEAN_KEY, STD_KEY)
print("[NORM] Example:", example_band, BAND_MEAN[example_band], BAND_STD[example_band])

FULL_IDX_TO_KEY = pd.read_csv(os.path.join(SSD_META_DIR, "dhs_combined_df.csv"))
FULL_IDX_TO_KEY[C_COUNTRY] = FULL_IDX_TO_KEY[C_COUNTRY].astype(str).str.upper().str.strip().str[:2]
FULL_IDX_TO_KEY["_key"] = FULL_IDX_TO_KEY[C_COUNTRY] + "|" + FULL_IDX_TO_KEY[C_YEAR].astype(str) + "|" + FULL_IDX_TO_KEY[C_CLUSTER].astype(str)
FULL_IDX_TO_KEY = FULL_IDX_TO_KEY["_key"].to_dict()

print("Checking local TFRecords...")
all_drive_files = sorted(glob.glob(os.path.join(DRIVE_TF_ROOT, "*.tfrecord")))
target_drive_paths = [f for f in all_drive_files if any(c in os.path.basename(f).upper() for c in WEST_AFRICA_CODES)]
target_filenames = {os.path.basename(p): p for p in target_drive_paths}

existing_local_files = set(os.listdir(LOCAL_RAW_DIR))
missing_files = [path for fname, path in target_filenames.items() if fname not in existing_local_files]

if missing_files:
    print(f"Downloading {len(missing_files)} missing files to SSD...")

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
            "year": tf.io.FixedLenFeature([], tf.int64),
            "cluster": tf.io.FixedLenFeature([], tf.int64),
        }
        try:
            for raw in tf.data.TFRecordDataset(path):
                ex = tf.io.parse_single_example(raw, idx_spec)
                key = f"{ex['country'].numpy().decode('utf-8').upper()[:2]}|{int(ex['year'])}|{int(ex['cluster'])}"
                local_idx[key] = (path, 0)
        except Exception:
            pass
        return local_idx

    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in tqdm(ex.map(index_worker, local_files), total=len(local_files)):
            tfrecord_index.update(res)

    with open(INDEX_CACHE, "wb") as f:
        pickle.dump(tfrecord_index, f)

print(f"Index Ready: {len(tfrecord_index)} clusters.")

def _estimate_cache_bytes(n_examples, variant):
    c = 7 if variant == "ms" else 1
    x_bytes = n_examples * EXPECTED_K * EXPECTED_H * EXPECTED_H * c * 2
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

    def make_key(country_str, year_i64, cluster_i64):
        c = tf.strings.upper(country_str)
        c = tf.strings.substr(c, 0, 2)
        return tf.strings.join([c, tf.as_string(year_i64), tf.as_string(cluster_i64)], separator="|")

    def is_valid_key(raw):
        s = {
            "country": tf.io.FixedLenFeature([], tf.string),
            "year": tf.io.FixedLenFeature([], tf.int64),
            "cluster": tf.io.FixedLenFeature([], tf.int64),
        }
        e = tf.io.parse_single_example(raw, s)
        k = make_key(e["country"], e["year"], e["cluster"])
        return valid_table.lookup(k) == 1

    def parse(raw):
        spec = {
            "K": tf.io.FixedLenFeature([], tf.int64),
            "patch_size": tf.io.FixedLenFeature([], tf.int64),
            "country": tf.io.FixedLenFeature([], tf.string),
            "year": tf.io.FixedLenFeature([], tf.int64),
            "cluster": tf.io.FixedLenFeature([], tf.int64),
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

            if b == "TEMP1":
                arr = arr * tf.cast(TEMP_SCALE, tf.float32) + tf.cast(TEMP_OFFSET, tf.float32)

            arr = tf.where(tf.math.is_finite(arr), arr, tf.zeros_like(arr))
            imgs.append(arr)

        X = tf.stack(imgs, axis=-1)
        X = tf.ensure_shape(X, [EXPECTED_K, EXPECTED_H, EXPECTED_H, C])

        X = (X - means) / (stds + 1e-8)

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

        def aug_like_code1(x, y):

            if tf.random.uniform([]) > 0.5:
                x = tf.reverse(x, axis=[2])
            if tf.random.uniform([]) > 0.5:
                x = tf.reverse(x, axis=[1])

            if variant == "ms":

                x32 = tf.cast(x, tf.float32)
                x32 = tf.image.random_brightness(x32, max_delta=0.5)
                x32 = tf.image.random_contrast(x32, lower=0.75, upper=1.25)
                x = tf.cast(x32, tf.float16)

            return x, y

        ds = ds.map(aug_like_code1, num_parallel_calls=tf.data.AUTOTUNE)
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
    agg = layers.GlobalAveragePooling1D(name="gap")(td)
    out = layers.Dense(1, kernel_regularizer=weight_reg, dtype="float32")(agg)

    model = keras.Model(mil_inp, out, name=f"mil_{variant}")
    opt = keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"], jit_compile=False)
    return model

print("\n--- 2. STARTING GRID SEARCH TRAINING ---")

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
                    keras.callbacks.ModelCheckpoint(tmp_best, monitor="val_loss", save_best_only=True, verbose=0),
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                        restore_best_weights=True, verbose=1
                    ),
                    keras.callbacks.TerminateOnNaN(),
                ]

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

                    def ev(d, n):
                        yt, yp = [], []
                        for x, y in d.take(math.ceil(n / BATCH_SIZE)):
                            yt.extend(y.numpy().reshape(-1).tolist())
                            yp.extend(model(x, training=False).numpy().reshape(-1).tolist())
                        yt = np.asarray(yt[:n], dtype=np.float32)
                        yp = np.asarray(yp[:n], dtype=np.float32)
                        return float(r2_score(yt, yp))

                    row = {
                        "scheme": scheme,
                        "fold": fold,
                        "variant": variant,
                        "lr": float(lr),
                        "l2": float(l2),
                        "ckpt_path": ckpt,
                        "val_r2": ev(ds["val"], counts["val"]),
                        "test_r2": ev(ds["test"], counts["test"]),
                    }
                    results.append(row)
                    pd.DataFrame(results).to_csv(RES_CSV, index=False)

                except Exception as e:
                    print(f"Error {tag}: {e}")
                    import traceback
                    traceback.print_exc()

print("\n--- 3. RIDGE FUSION ---")

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
    return np.concatenate(fl, axis=0), np.concatenate(yl, axis=0), np.concatenate(cl, axis=0).astype(str)

fusion_rows = []

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
    print("No fusion rows produced (missing ms/nl pairs).")
