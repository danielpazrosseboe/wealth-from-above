from google.colab import drive
drive.mount("/content/drive")

import os, glob, json, pickle, random, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

print("TF version:", tf.__version__)
print("Compute Policy:", policy.name)

SEED = 123
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

AUTOTUNE   = tf.data.AUTOTUNE
PATCH_SIZE = 255
CROP       = 224
BATCH_SIZE = 64

EPOCHS_INCOUNTRY = 150
EPOCHS_OOC       = 200
def get_epochs(scheme): return EPOCHS_INCOUNTRY if scheme == "incountry" else EPOCHS_OOC

TFRECORD_MERGED_DIR = "/content/drive/My Drive/TFRecords/merged_rounds"

DD_BASE       = "/content/drive/My Drive/Data Distribution"
COMBINED_CSV  = os.path.join(DD_BASE, "dhs_combined_df.csv")
INC_FOLDS_PKL = os.path.join(DD_BASE, "dhs_incountry_folds.pkl")
OOC_FOLDS_PKL = os.path.join(DD_BASE, "dhs_ooc_folds.pkl")

GLOBAL_STATS_JSON = "/content/drive/My Drive/Data Validation/Band Analysis/band_stats_summary.json"

HYPERSEARCH_CSV_CANDIDATES = [
    os.path.join(DD_BASE, "cnn_results_yeh_style_full_hp", "full_hypersearch_results_with_metrics.csv"),
    os.path.join(DD_BASE, "cnn_results_yeh_style_full_hp", "full_hypersearch_results.csv"),
]

ROOT_OUT     = os.path.join(DD_BASE, "cnn_results_merged_retrain_best_hp")
CKPT_DIR     = os.path.join(ROOT_OUT, "checkpoints")
LOG_DIR_ROOT = os.path.join(ROOT_OUT, "logs")
PLOTS_DIR    = os.path.join(ROOT_OUT, "plots")
for d in [ROOT_OUT, CKPT_DIR, LOG_DIR_ROOT, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

def savefig(path, dpi=200):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print("Saved figure:", path)

BANDS_MS  = ["RED","GREEN","BLUE","SWIR1","SWIR2","TEMP1","NIR"]
BANDS_NL  = ["NIGHTLIGHTS"]
BANDS_ALL = BANDS_MS + BANDS_NL

TEMP_SCALE  = 0.00341802
TEMP_OFFSET = 149.0

IMAGENET_RGB_H5 = "/content/drive/My Drive/resnet18_imagenet_rgb.h5"

SCHEMES = ["incountry","ooc"]
FOLDS   = ["A","B","C","D","E"]
SPLITS  = ["train","val","test"]

print("TFRECORD_MERGED_DIR:", TFRECORD_MERGED_DIR)

with open(GLOBAL_STATS_JSON, "r") as f:
    stats = json.load(f)

example_band = BANDS_ALL[0]
MEAN_KEY = next(k for k in stats[example_band].keys() if "mean" in k.lower())
STD_KEY  = next(k for k in stats[example_band].keys() if ("std" in k.lower() or "stdev" in k.lower()))

BAND_MEAN = {b: float(stats[b][MEAN_KEY]) for b in BANDS_ALL}
BAND_STD  = {b: float(stats[b][STD_KEY])  for b in BANDS_ALL}

print("Loaded normalization keys:", MEAN_KEY, STD_KEY)
print("Example mean/std:", BANDS_ALL[0], BAND_MEAN[BANDS_ALL[0]], BAND_STD[BANDS_ALL[0]])

if not tf.io.gfile.exists(COMBINED_CSV):
    raise FileNotFoundError(f"Missing COMBINED_CSV: {COMBINED_CSV}")
if not tf.io.gfile.exists(INC_FOLDS_PKL):
    raise FileNotFoundError(f"Missing folds file: {INC_FOLDS_PKL}")
if not tf.io.gfile.exists(OOC_FOLDS_PKL):
    raise FileNotFoundError(f"Missing folds file: {OOC_FOLDS_PKL}")

df_meta = pd.read_csv(COMBINED_CSV)

def _pick(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of {cands} in combined df columns: {list(df.columns)[:80]}")

C_COUNTRY  = _pick(df_meta, ["country","CC","iso2"])
C_CLUSTER  = _pick(df_meta, ["cluster_index","cluster","clu"])

C_TF_YEAR  = df_meta.columns[df_meta.columns.isin(["tfrecord_year"])].tolist()
C_TF_YEAR  = C_TF_YEAR[0] if len(C_TF_YEAR) else _pick(df_meta, ["year","survey_year"])

df_meta[C_COUNTRY] = df_meta[C_COUNTRY].astype(str).str.upper().str.strip().str[:2]
df_meta[C_TF_YEAR] = pd.to_numeric(df_meta[C_TF_YEAR], errors="coerce").astype("Int64")
df_meta[C_CLUSTER] = pd.to_numeric(df_meta[C_CLUSTER], errors="coerce").astype("Int64")

if df_meta[[C_COUNTRY, C_TF_YEAR, C_CLUSTER]].isna().any().any():
    bad = df_meta[df_meta[[C_COUNTRY, C_TF_YEAR, C_CLUSTER]].isna().any(axis=1)].head(10)
    raise ValueError(
        "combined df has NaNs in key columns. Example rows:\n"
        + bad[[C_COUNTRY, C_TF_YEAR, C_CLUSTER]].to_string(index=False)
    )

KEY_STR = (df_meta[C_COUNTRY].astype(str) + "|" +
           df_meta[C_TF_YEAR].astype(int).astype(str) + "|" +
           df_meta[C_CLUSTER].astype(int).astype(str)).to_numpy(dtype=str)

PAIR_STR = (df_meta[C_COUNTRY].astype(str) + "_" +
            df_meta[C_TF_YEAR].astype(int).astype(str)).to_numpy(dtype=str)

with tf.io.gfile.GFile(INC_FOLDS_PKL, "rb") as f:
    INCOUNTRY_FOLDS = pickle.load(f)
with tf.io.gfile.GFile(OOC_FOLDS_PKL, "rb") as f:
    OOC_FOLDS = pickle.load(f)

def get_folds_dict(scheme: str):
    return INCOUNTRY_FOLDS if scheme == "incountry" else OOC_FOLDS

print("Loaded df_meta rows:", len(df_meta))
print("Example KEY_STR:", KEY_STR[:3])

paths = sorted(glob.glob(os.path.join(TFRECORD_MERGED_DIR, "*.tfrecord*")))
if not paths:
    raise FileNotFoundError(f"No merged TFRecords found in: {TFRECORD_MERGED_DIR}")

pair_to_path = {}
for p in paths:
    base = os.path.basename(p)

    if base.count("_") >= 1:
        cc = base.split("_")[0].upper()
        yr = base.split("_")[1][:4]
        pair = f"{cc}_{yr}"
        pair_to_path[pair] = p

print("Merged TFRecord files found:", len(paths))
print("Unique CC_YYYY pairs mapped:", len(pair_to_path))
print("Example mapping:", list(pair_to_path.items())[:3])

pairs_in_df = set(PAIR_STR.tolist())
missing_pairs = sorted([p for p in pairs_in_df if p not in pair_to_path])
print("Pairs in df:", len(pairs_in_df))
print("Missing pair files:", len(missing_pairs))
if missing_pairs[:10]:
    print("First missing pairs:", missing_pairs[:10])

key_feature_description = {
    "country":       tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "year":          tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
    "cluster_index": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
    "cluster":       tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
}

feature_description = {
    **{b: tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32) for b in BANDS_ALL},

    "LON":          tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32, default_value=[0.0]*(PATCH_SIZE*PATCH_SIZE)),
    "LAT":          tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32, default_value=[0.0]*(PATCH_SIZE*PATCH_SIZE)),
    "wealthpooled": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    "country":      tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "year":         tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
    "cluster_index":tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
    "cluster":      tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
}

def _norm_country(cc_bytes):
    cc = tf.strings.upper(cc_bytes)
    cc = tf.strings.substr(cc, 0, 2)
    return cc

def _pick_cluster(ci, c):

    return tf.where(ci >= 0.0, ci, c)

def key_from_serialized(serialized):
    f = tf.io.parse_single_example(serialized, key_feature_description)
    cc = _norm_country(f["country"])
    yr = tf.cast(tf.round(f["year"]), tf.int32)
    cl = tf.cast(tf.round(_pick_cluster(f["cluster_index"], f["cluster"])), tf.int32)
    key = tf.strings.join([cc, tf.strings.as_string(yr), tf.strings.as_string(cl)], separator="|")
    return key

def center_crop_2d(img, crop_size=CROP):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    pad_h = tf.maximum(crop_size - h, 0)
    pad_w = tf.maximum(crop_size - w, 0)
    if (pad_h > 0) or (pad_w > 0):
        return tf.image.resize_with_pad(img, crop_size, crop_size)
    off_h = (h - crop_size) // 2
    off_w = (w - crop_size) // 2
    return tf.image.crop_to_bounding_box(img, off_h, off_w, crop_size, crop_size)

def normalize_bands(img, bands):
    chans = []
    for i, b in enumerate(bands):
        ch = img[..., i]
        ch = (ch - tf.cast(BAND_MEAN[b], tf.float32)) / (tf.cast(BAND_STD[b], tf.float32) + 1e-8)
        chans.append(ch)
    return tf.stack(chans, axis=-1)

def parse_example_full(serialized_example, variant):
    f = tf.io.parse_single_example(serialized_example, feature_description)

    def band_to_img(band_name):
        flat = f[band_name]
        img = tf.reshape(flat, [PATCH_SIZE, PATCH_SIZE])
        if band_name == "TEMP1":
            img = img * tf.cast(TEMP_SCALE, tf.float32) + tf.cast(TEMP_OFFSET, tf.float32)
        return img

    if variant == "ms":
        bands = BANDS_MS
    elif variant == "nl":
        bands = BANDS_NL
    else:
        raise ValueError(f"Unknown variant: {variant}")

    imgs = [band_to_img(b) for b in bands]
    img = tf.stack(imgs, axis=-1)
    img = center_crop_2d(img, CROP)
    img = normalize_bands(img, bands)

    y = f["wealthpooled"]
    return img, y

def make_augment_fn(variant):
    def augment(img, y):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        if variant == "ms":
            img = tf.image.random_brightness(img, max_delta=0.5)
            img = tf.image.random_contrast(img, lower=0.75, upper=1.25)
        return img, y
    return augment

def _files_for_indices(indices):
    indices = np.asarray(indices, dtype=np.int64)
    pairs = np.unique(PAIR_STR[indices])
    files = []
    missing = []
    for p in pairs:
        if p in pair_to_path:
            files.append(pair_to_path[p])
        else:
            missing.append(p)
    if missing:
        print(f"[WARN] Missing {len(missing)} CC_YYYY files (first 10):", missing[:10])
    return sorted(files)

def _make_tfrecord_ds_from_files(files):
    files = list(files)
    if not files:
        raise ValueError("No TFRecord files selected.")

    gz = [f for f in files if f.endswith(".gz")]
    plain = [f for f in files if not f.endswith(".gz")]

    ds_parts = []
    if plain:
        ds_parts.append(
            tf.data.Dataset.from_tensor_slices(plain).interleave(
                lambda fn: tf.data.TFRecordDataset(fn),
                cycle_length=min(16, len(plain)),
                num_parallel_calls=AUTOTUNE,
                deterministic=False,
            )
        )
    if gz:
        ds_parts.append(
            tf.data.Dataset.from_tensor_slices(gz).interleave(
                lambda fn: tf.data.TFRecordDataset(fn, compression_type="GZIP"),
                cycle_length=min(16, len(gz)),
                num_parallel_calls=AUTOTUNE,
                deterministic=False,
            )
        )

    ds = ds_parts[0]
    for part in ds_parts[1:]:
        ds = ds.concatenate(part)
    return ds

def build_dataset_from_indices(indices, variant, shuffle=True, cache=False):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("Empty index array.")

    keys_np = KEY_STR[indices]
    keys_tf = tf.constant(keys_np, dtype=tf.string)

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys_tf, tf.ones_like(keys_tf, dtype=tf.int32)),
        default_value=0
    )

    files = _files_for_indices(indices)
    ds = _make_tfrecord_ds_from_files(files)

    ds = ds.filter(lambda s: table.lookup(key_from_serialized(s)) > 0)

    ds = ds.map(lambda s: parse_example_full(s, variant=variant), num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
        ds = ds.map(make_augment_fn(variant), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def make_fold_datasets(variant, scheme, fold_letter, cache=False):
    folds_dict = get_folds_dict(scheme)
    idx_train = np.asarray(folds_dict[fold_letter]["train"], dtype=np.int64)
    idx_val   = np.asarray(folds_dict[fold_letter]["val"],   dtype=np.int64)
    idx_test  = np.asarray(folds_dict[fold_letter]["test"],  dtype=np.int64)

    print(f"{scheme}/{fold_letter}/{variant} -> Train:{len(idx_train)}, Val:{len(idx_val)}, Test:{len(idx_test)}")

    train_ds = build_dataset_from_indices(idx_train, variant, shuffle=True,  cache=cache)
    val_ds   = build_dataset_from_indices(idx_val,   variant, shuffle=False, cache=cache)
    test_ds  = build_dataset_from_indices(idx_test,  variant, shuffle=False, cache=cache)
    return train_ds, val_ds, test_ds

_ = make_fold_datasets("ms", "incountry", "A", cache=False)
print("Dataset smoke test OK.")

from tensorflow.keras.utils import register_keras_serializable

def conv3x3(filters, stride=1, weight_reg=None):
    return layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                         kernel_initializer="he_normal", kernel_regularizer=weight_reg)

def conv1x1(filters, stride=1, weight_reg=None):
    return layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False,
                         kernel_initializer="he_normal", kernel_regularizer=weight_reg)

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

    def build(self, input_shape):

        self.bn1.build(input_shape)

        self.conv1.build(input_shape)
        s1 = self.conv1.compute_output_shape(input_shape)

        self.bn2.build(s1)
        self.conv2.build(s1)

        if self.shortcut is not None:
            self.shortcut.build(input_shape)

        super().build(input_shape)

    def call(self, x, training=False):
        shortcut = x

        out = self.bn1(x, training=training)
        out = tf.nn.relu(out)

        out = tf.cast(out, self.compute_dtype)

        if self.shortcut is not None:
            shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)
        out = tf.cast(out, self.compute_dtype)
        out = self.conv2(out)

        if shortcut.dtype != out.dtype:
            shortcut = tf.cast(shortcut, out.dtype)

        return out + shortcut

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "stride": self.stride,
            "weight_reg": self.weight_reg,
        })
        return config

def _make_resnet18_backbone_generic(input_shape, l2_weight=0.0, name="backbone"):

    weight_reg = keras.regularizers.l2(l2_weight) if l2_weight > 0 else None

    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False,
                      kernel_initializer="he_normal", kernel_regularizer=weight_reg, name="conv1")(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    def make_block(x, filters, num_blocks, stride_first):
        x = PreActBlock(filters, stride=stride_first, weight_reg=weight_reg)(x)
        for _ in range(1, num_blocks):
            x = PreActBlock(filters, stride=1, weight_reg=weight_reg)(x)
        return x

    x = make_block(x, 64, 2, 1)
    x = make_block(x, 128, 2, 2)
    x = make_block(x, 256, 2, 2)
    x = make_block(x, 512, 2, 2)

    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU()(x)
    feats = layers.GlobalAveragePooling2D(name="gap")(x)

    return keras.Model(inputs, feats, name=name)

def make_hs_conv1_weights_from_rgb(rgb_weights, target_in, hs_weight_init="samescaled"):
    if target_in == 3:
        return rgb_weights
    if target_in < 3:
        rgb_mean = rgb_weights.mean(axis=2, keepdims=True)
        return np.tile(rgb_mean, (1, 1, target_in, 1))

    num_hs = target_in - 3
    rgb_mean = rgb_weights.mean(axis=2, keepdims=True)
    hs_weights = np.tile(rgb_mean, (1, 1, num_hs, 1))

    if hs_weight_init == "samescaled":
        scale = 3.0 / target_in
        return np.concatenate([rgb_weights * scale, hs_weights * scale], axis=2)

    return np.concatenate([rgb_weights, hs_weights], axis=2)

def init_backbone_from_imagenet(backbone, h5_path, hs_init="samescaled"):
    rgb_backbone = _make_resnet18_backbone_generic((224,224,3), name="tmp_rgb")
    rgb_backbone.load_weights(h5_path)

    for layer in backbone.layers:
        if not layer.weights or isinstance(layer, PreActBlock):
            continue
        if layer.name == "conv1":
            rgb_w = rgb_backbone.get_layer("conv1").get_weights()[0]
            target_in = layer.get_weights()[0].shape[2]
            new_w = make_hs_conv1_weights_from_rgb(rgb_w, target_in, hs_init)
            layer.set_weights([new_w])
            continue
        try:
            rgb_layer = rgb_backbone.get_layer(layer.name)
            layer.set_weights(rgb_layer.get_weights())
        except Exception:
            pass

    ms_blocks  = [l for l in backbone.layers if isinstance(l, PreActBlock)]
    rgb_blocks = [l for l in rgb_backbone.layers if isinstance(l, PreActBlock)]
    for m, r in zip(ms_blocks, rgb_blocks):
        m.set_weights(r.get_weights())

def make_resnet18_regressor(variant, l2_weight=0.0):
    if variant == "ms":
        input_shape = (CROP, CROP, len(BANDS_MS))
        backbone = _make_resnet18_backbone_generic(input_shape, l2_weight, name="ms_backbone")
        init_backbone_from_imagenet(backbone, IMAGENET_RGB_H5)
    elif variant == "nl":
        input_shape = (CROP, CROP, len(BANDS_NL))
        backbone = _make_resnet18_backbone_generic(input_shape, l2_weight, name="nl_backbone")

    else:
        raise ValueError("variant must be 'ms' or 'nl'")

    out = layers.Dense(
        1,
        kernel_regularizer=keras.regularizers.l2(l2_weight) if l2_weight > 0 else None,
        dtype="float32"
    )(backbone.output)

    return keras.Model(backbone.input, out)

def make_lr_schedule(initial_lr=1e-4, decay_factor=0.96):
    return lambda epoch, lr: float(initial_lr * (decay_factor ** epoch))

LOCAL_CKPT_DIR = "/content/ckpts_tmp"
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

def _drive_ckpt_path(run_tag): return os.path.join(CKPT_DIR, f"{run_tag}.keras")
def _local_ckpt_path(run_tag): return os.path.join(LOCAL_CKPT_DIR, f"{run_tag}.keras")

def _is_valid_checkpoint(path, min_bytes=50_000):
    try:
        return tf.io.gfile.exists(path) and tf.io.gfile.stat(path).length >= min_bytes
    except Exception:
        return False

def _cleanup_bad_checkpoint(path):
    try:
        if tf.io.gfile.exists(path):
            tf.io.gfile.remove(path)
            print(f"[CLEANUP] Removed invalid checkpoint: {path}")
    except Exception as e:
        print(f"[WARN] Could not remove {path}: {e}")

def eval_model_on_dataset(model, ds):
    y_true = []
    y_pred = []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0).reshape(-1)
        y_true.append(yb.numpy().reshape(-1))
        y_pred.append(p.astype(np.float32))
    y_true = np.concatenate(y_true).astype(np.float32)
    y_pred = np.concatenate(y_pred).astype(np.float32)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "n": int(len(y_true)),
    }

def train_one_model_besthp(variant, train_ds, val_ds, scheme, fold, lr, l2, seed, run_tag):
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

    drive_ckpt = _drive_ckpt_path(run_tag)
    local_ckpt = _local_ckpt_path(run_tag)

    if _is_valid_checkpoint(drive_ckpt):
        print(f"[SKIP] Valid checkpoint exists: {drive_ckpt}")
        return drive_ckpt

    if tf.io.gfile.exists(drive_ckpt) and (not _is_valid_checkpoint(drive_ckpt)):
        _cleanup_bad_checkpoint(drive_ckpt)
    if tf.io.gfile.exists(local_ckpt) and (not _is_valid_checkpoint(local_ckpt)):
        _cleanup_bad_checkpoint(local_ckpt)

    model = make_resnet18_regressor(variant, l2_weight=l2)

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=["mae"],
        jit_compile=True,
    )

    ckpt_cb = keras.callbacks.ModelCheckpoint(local_ckpt, monitor="val_loss", mode="min", save_best_only=True, verbose=0)
    lr_cb   = keras.callbacks.LearningRateScheduler(make_lr_schedule(lr), verbose=0)
    tb_cb   = keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR_ROOT, run_tag))

    n_epochs = get_epochs(scheme)
    print(f"Training {run_tag} for FULL {n_epochs} epochs (EarlyStopping DISABLED)...")

    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=[lr_cb, ckpt_cb, tb_cb],
        verbose=2
    )

    if _is_valid_checkpoint(local_ckpt):
        tf.io.gfile.copy(local_ckpt, drive_ckpt, overwrite=True)
        print(f"Saved best model to {drive_ckpt}")
    else:
        print("[WARN] No valid local checkpoint; saving current model to Drive.")
        model.save(drive_ckpt)

    tf.keras.backend.clear_session()
    return drive_ckpt

import itertools
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras

RETRAIN_SEED = 123
CACHE_DATASETS = True
SCHEME = "ooc"
VARIANT = "ms"

CKPT_DIR = "/content/drive/My Drive/Data Distribution/cnn_results_merged_retrain_best_hp/checkpoints"
CKPT_EXT = ".keras"

GRID_LR = [1e-2, 1e-3, 1e-4, 1e-5]
GRID_L2 = [1e-0, 1e-1, 1e-2, 1e-3]
FOLDS   = ["A", "B", "C", "D", "E"]

output_csv = os.path.join(ROOT_OUT, "grid_ooc_ms.csv")

if os.path.exists(output_csv):
    print(f"Loading existing CSV from: {output_csv}")
    rows = pd.read_csv(output_csv).to_dict('records')
else:
    rows = []

print(f"Starting Full Grid Search for {SCHEME}/{VARIANT}...")
print(f"Checking for existing files in: {CKPT_DIR}")

for fold in FOLDS:

    missing_combos = []
    print(f"\n>>> Checking status for Fold {fold}...")

    for lr, l2 in itertools.product(GRID_LR, GRID_L2):

        run_tag = f"GRID_{SCHEME}_{fold}_{VARIANT}_lr{lr:.0e}_l2{l2:.0e}_s{RETRAIN_SEED}"
        expected_file = os.path.join(CKPT_DIR, run_tag + CKPT_EXT)

        if not os.path.exists(expected_file):
            missing_combos.append((lr, l2, run_tag))

    if not missing_combos:
        print(f"  All checkpoints for Fold {fold} exist. Skipping dataset load.")
        continue

    print(f"  Found {len(missing_combos)} missing runs. Loading Datasets...")
    train_ds, val_ds, test_ds = make_fold_datasets(VARIANT, SCHEME, fold, cache=CACHE_DATASETS)

    for lr, l2, run_tag in missing_combos:
        print(f"\n--- Training Missing: {run_tag} ---")

        ckpt_path = train_one_model_besthp(
            variant=VARIANT, train_ds=train_ds, val_ds=val_ds,
            scheme=SCHEME, fold=fold, lr=lr, l2=l2, seed=RETRAIN_SEED, run_tag=run_tag
        )

        model = keras.models.load_model(ckpt_path, custom_objects={"PreActBlock": PreActBlock})
        metrics = eval_model_on_dataset(model, test_ds)
        tf.keras.backend.clear_session()

        rows.append({
            "scheme": SCHEME, "fold": fold, "variant": VARIANT,
            "lr": lr, "l2": l2, "seed": RETRAIN_SEED,
            "ckpt_path": ckpt_path,
            **{f"test_{k}": v for k, v in metrics.items()},
        })

        pd.DataFrame(rows).to_csv(output_csv, index=False)

print(f"\nCompleted {SCHEME}/{VARIANT}. Results saved to: {output_csv}")
