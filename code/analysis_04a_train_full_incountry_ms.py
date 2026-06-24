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
SCHEME = "incountry"
VARIANT = "ms"

CKPT_DIR = "/content/drive/My Drive/Data Distribution/cnn_results_merged_retrain_best_hp/checkpoints"
CKPT_EXT = ".keras"

GRID_LR = [1e-2, 1e-3, 1e-4, 1e-5]
GRID_L2 = [1e-0, 1e-1, 1e-2, 1e-3]
FOLDS   = ["B"]

output_csv = os.path.join(ROOT_OUT, "grid_incountry_ms.csv")

if os.path.exists(output_csv):
    rows = pd.read_csv(output_csv).to_dict('records')
else:
    rows = []

print(f"Checking for missing files in: {CKPT_DIR}")

for fold in FOLDS:

    print(f"\n>>> Checking Fold {fold}...")

    missing_combos = []
    for lr, l2 in itertools.product(GRID_LR, GRID_L2):

        run_tag = f"GRID_{SCHEME}_{fold}_{VARIANT}_lr{lr:.0e}_l2{l2:.0e}_s{RETRAIN_SEED}"
        expected_file = os.path.join(CKPT_DIR, run_tag + CKPT_EXT)

        if not os.path.exists(expected_file):
            missing_combos.append((lr, l2, run_tag))
        else:
            print(f"  [FOUND] {run_tag}")

    if not missing_combos:
        print(f"  All checkpoints for Fold {fold} exist. Skipping.")
        continue

    print(f"  Found {len(missing_combos)} missing checkpoints. Loading Data...")
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

print(f"\nProcess Complete. Results saved to: {output_csv}")

import re
import pandas as pd
import tensorflow as tf
from tensorflow import keras

fname_pattern = re.compile(r"GRID_([a-z]+)_([A-E])_([a-z]+)_lr([\d\.e-]+)_l2([\d\.e-]+)_s(\d+)")

ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.keras")))
print(f"Found {len(ckpt_files)} checkpoints in {CKPT_DIR}")

tasks = []
for fpath in ckpt_files:
    fname = os.path.basename(fpath)
    match = fname_pattern.search(fname)
    if match:
        scheme, fold, variant, lr_str, l2_str, seed_str = match.groups()
        tasks.append({
            "path": fpath,
            "filename": fname,
            "scheme": scheme,
            "fold": fold,
            "variant": variant,
            "lr": float(lr_str),
            "l2": float(l2_str),
            "seed": int(seed_str)
        })

if not tasks:
    raise ValueError("No checkpoints matching the GRID pattern were found.")

df_tasks = pd.DataFrame(tasks)

results = []
unique_configs = df_tasks[["scheme", "fold", "variant"]].drop_duplicates().values

print(f"\nRe-evaluating models across {len(unique_configs)} data configurations...")

for scheme, fold, variant in unique_configs:
    print(f"\n>>> Loading Data for: Scheme={scheme}, Fold={fold}, Variant={variant}")

    _, val_ds, _ = make_fold_datasets(variant, scheme, fold, cache=True)

    group_df = df_tasks[
        (df_tasks["scheme"] == scheme) &
        (df_tasks["fold"] == fold) &
        (df_tasks["variant"] == variant)
    ]

    for _, row in tqdm(group_df.iterrows(), total=len(group_df), desc="Eval Models"):
        try:

            model = keras.models.load_model(row["path"], custom_objects={"PreActBlock": PreActBlock})

            metrics = eval_model_on_dataset(model, val_ds)

            res_entry = row.to_dict()
            res_entry["val_r2"] = metrics["r2"]
            res_entry["val_mae"] = metrics["mae"]
            res_entry["val_mse"] = metrics["mse"]
            results.append(res_entry)

            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"[ERROR] Failed to eval {row['filename']}: {e}")

if results:
    df_results = pd.DataFrame(results)

    df_results = df_results.sort_values("val_r2", ascending=False)

    recovered_csv = os.path.join(ROOT_OUT, "recovered_hypersearch_results.csv")
    df_results.to_csv(recovered_csv, index=False)

    print("-" * 50)
    print("RECOVERY COMPLETE")
    print(f"Results saved to: {recovered_csv}")
    print("-" * 50)

    print("\nTOP 5 MODELS BY VAL R2:")
    cols_to_show = ["scheme", "fold", "variant", "lr", "l2", "val_r2", "val_mae"]
    display(df_results[cols_to_show].head(5))

    best_row = df_results.iloc[0]
    print(f"\nBEST MODEL FOUND: {best_row['filename']}")
    print(f"LR: {best_row['lr']}, L2: {best_row['l2']} -> Val R2: {best_row['val_r2']:.4f}")

else:
    print("No results could be recovered.")

def parse_example_full_with_country(serialized_example, variant):
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
    country = _norm_country(f["country"])
    return img, y, country

def build_dataset_with_country(indices, variant, batch_size=BATCH_SIZE):
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

    ds = ds.map(lambda s: parse_example_full_with_country(s, variant=variant), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
from tqdm.auto import tqdm

SCHEME = "incountry"
FOLDS  = ["B"]

fusion_out_csv_scaled = os.path.join(
    ROOT_OUT,
    f"ridge_fusion_scaled_{SCHEME}_foldB.csv"
)
print(f"Scaled fusion results will be saved to: {fusion_out_csv_scaled}")

RETRAIN_CSV = os.path.join(ROOT_OUT, "recovered_hypersearch_results.csv")
if not os.path.exists(RETRAIN_CSV):
    raise FileNotFoundError(f"Expected recovered_hypersearch_results.csv at: {RETRAIN_CSV}")

df_retrain = pd.read_csv(RETRAIN_CSV)

if "path" in df_retrain.columns and "ckpt_path" not in df_retrain.columns:
    print("Renaming column 'path' to 'ckpt_path' in loaded dataframe.")
    df_retrain = df_retrain.rename(columns={"path": "ckpt_path"})

if "ckpt_path" not in df_retrain.columns:
    raise KeyError(
        f"{RETRAIN_CSV} is missing 'ckpt_path'. "
        f"Available columns: {list(df_retrain.columns)}"
    )

if "val_r2" not in df_retrain.columns:
    raise KeyError(
        f"{RETRAIN_CSV} is missing 'val_r2'. "
        f"Columns: {list(df_retrain.columns)}"
    )

RIDGE_ALPHAS = [10**p for p in range(-4, 5)]
fusion_results_scaled = []

def get_features_and_meta(model, ds):
    gap_layer = model.get_layer("gap")
    feature_model = keras.Model(model.input, gap_layer.output)

    feats_list, y_list, c_list = [], [], []

    for img_b, y_b, c_b in tqdm(ds, leave=False, desc="Extracting features"):
        f = feature_model.predict(img_b, verbose=0)
        feats_list.append(f)
        y_list.append(y_b.numpy())
        c_list.append(c_b.numpy())

    if not feats_list:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(feats_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(c_list, axis=0).astype(str),
    )

for fold in FOLDS:
    print(f"\n=== RIDGE FUSION (SCALED): scheme={SCHEME} / fold={fold} ===")

    df_ms = df_retrain[
        (df_retrain["scheme"] == SCHEME) &
        (df_retrain["fold"]   == fold) &
        (df_retrain["variant"] == "ms")
    ]
    df_nl = df_retrain[
        (df_retrain["scheme"] == SCHEME) &
        (df_retrain["fold"]   == fold) &
        (df_retrain["variant"] == "nl")
    ]

    if df_ms.empty or df_nl.empty:
        print(f"[SKIP] Missing MS or NL entries in recovered CSV for {SCHEME}/{fold}")
        continue

    best_ms = df_ms.sort_values("val_r2", ascending=False).iloc[0]
    best_nl = df_nl.sort_values("val_r2", ascending=False).iloc[0]

    path_ms = best_ms["ckpt_path"]
    path_nl = best_nl["ckpt_path"]

    print(f"Best MS for {SCHEME}/{fold}: lr={best_ms['lr']}, l2={best_ms['l2']}, val_r2={best_ms['val_r2']:.4f}")
    print(f"  -> checkpoint: {path_ms}")
    print(f"Best NL for {SCHEME}/{fold}: lr={best_nl['lr']}, l2={best_nl['l2']}, val_r2={best_nl['val_r2']:.4f}")
    print(f"  -> checkpoint: {path_nl}")

    folds_dict = get_folds_dict(SCHEME)
    idx_train = np.asarray(folds_dict[fold]["train"], dtype=np.int64)
    idx_val   = np.asarray(folds_dict[fold]["val"],   dtype=np.int64)
    idx_test  = np.asarray(folds_dict[fold]["test"],  dtype=np.int64)

    print("-> Building datasets with country (no shuffle)...")
    ds_tr_ms   = build_dataset_with_country(idx_train, "ms")
    ds_val_ms  = build_dataset_with_country(idx_val,   "ms")
    ds_test_ms = build_dataset_with_country(idx_test,  "ms")

    ds_tr_nl   = build_dataset_with_country(idx_train, "nl")
    ds_val_nl  = build_dataset_with_country(idx_val,   "nl")
    ds_test_nl = build_dataset_with_country(idx_test,  "nl")

    print("-> Extracting MS features...")
    model_ms = keras.models.load_model(path_ms, custom_objects={"PreActBlock": PreActBlock})
    X_tr_ms, y_tr, c_tr       = get_features_and_meta(model_ms, ds_tr_ms)
    X_val_ms, y_val, c_val    = get_features_and_meta(model_ms, ds_val_ms)
    X_test_ms, y_test, c_test = get_features_and_meta(model_ms, ds_test_ms)
    del model_ms
    tf.keras.backend.clear_session()

    print("-> Extracting NL features...")
    model_nl = keras.models.load_model(path_nl, custom_objects={"PreActBlock": PreActBlock})
    X_tr_nl, _, _   = get_features_and_meta(model_nl, ds_tr_nl)
    X_val_nl, _, _  = get_features_and_meta(model_nl, ds_val_nl)
    X_test_nl, _, _ = get_features_and_meta(model_nl, ds_test_nl)
    del model_nl
    tf.keras.backend.clear_session()

    print("Shapes:")
    print("  X_tr_ms:",   X_tr_ms.shape,   "X_tr_nl:",   X_tr_nl.shape)
    print("  X_val_ms:",  X_val_ms.shape,  "X_val_nl:",  X_val_nl.shape)
    print("  X_test_ms:", X_test_ms.shape, "X_test_nl:", X_test_nl.shape)

    X_tr_fused   = np.concatenate([X_tr_ms,   X_tr_nl],  axis=1)
    X_val_fused  = np.concatenate([X_val_ms,  X_val_nl], axis=1)
    X_test_fused = np.concatenate([X_test_ms, X_test_nl], axis=1)

    scaler = StandardScaler()
    X_tr_scaled   = scaler.fit_transform(X_tr_fused)
    X_val_scaled  = scaler.transform(X_val_fused)
    X_test_scaled = scaler.transform(X_test_fused)

    best_alpha  = None
    best_val_r2 = -np.inf

    for alpha in RIDGE_ALPHAS:
        clf = Ridge(alpha=alpha)
        clf.fit(X_tr_scaled, y_tr)
        val_pred = clf.predict(X_val_scaled)
        r2 = r2_score(y_val, val_pred)

        if r2 > best_val_r2:
            best_val_r2 = r2
            best_alpha  = alpha

    print(f"-> Best alpha (scaled) on VAL: {best_alpha} (Val R²: {best_val_r2:.4f})")

    X_full_fused   = np.concatenate([X_tr_fused,  X_val_fused], axis=0)
    y_full         = np.concatenate([y_tr,        y_val],       axis=0)
    X_full_scaled  = scaler.transform(X_full_fused)

    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_full_scaled, y_full)

    preds = final_model.predict(X_test_scaled)

    r2_test  = r2_score(y_test, preds)
    mae_test = mean_absolute_error(y_test, preds)
    print(f"-> TEST R² (scaled fusion, {SCHEME}/{fold}): {r2_test:.4f}, MAE: {mae_test:.4f}")

    df_fold_preds = pd.DataFrame({
        "country":    c_test,
        "y_true":     y_test,
        "y_pred":     preds,
        "scheme":     SCHEME,
        "fold":       fold,
        "best_alpha": best_alpha,
        "ms_lr":      best_ms["lr"],
        "ms_l2":      best_ms["l2"],
        "nl_lr":      best_nl["lr"],
        "nl_l2":      best_nl["l2"],
    })
    fusion_results_scaled.append(df_fold_preds)

if fusion_results_scaled:
    df_fusion_scaled_all = pd.concat(fusion_results_scaled, ignore_index=True)
    df_fusion_scaled_all.to_csv(fusion_out_csv_scaled, index=False)
    print(f"\nSaved SCALED fusion predictions to: {fusion_out_csv_scaled}")
else:
    print("No scaled fusion results generated for Fold B.")

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
from tqdm.auto import tqdm

fusion_out_csv = os.path.join(ROOT_OUT, "ridge_fusion_predictions.csv")

if os.path.exists(fusion_out_csv):
    print(f" Found existing fusion results at: {fusion_out_csv}")
    print("Loading data directly...")
    df_fusion_all = pd.read_csv(fusion_out_csv)

else:
    print(" No existing results found. Starting Ridge Fusion process...")

    RETRAIN_CSV = os.path.join(ROOT_OUT, "recovered_hypersearch_results.csv")
    if not os.path.exists(RETRAIN_CSV):

        RETRAIN_CSV = os.path.join(ROOT_OUT, "retrain_best_hp_test_metrics.csv")

    if not os.path.exists(RETRAIN_CSV):
        raise FileNotFoundError(f"Could not find results CSV at: {RETRAIN_CSV}")

    df_retrain = pd.read_csv(RETRAIN_CSV)

    if "path" in df_retrain.columns and "ckpt_path" not in df_retrain.columns:
        print(f" Renaming column 'path' to 'ckpt_path' in loaded dataframe.")
        df_retrain = df_retrain.rename(columns={"path": "ckpt_path"})

    if "ckpt_path" not in df_retrain.columns:
        raise KeyError(f"The CSV file {RETRAIN_CSV} is missing the 'ckpt_path' column. Available columns: {list(df_retrain.columns)}")

    RIDGE_ALPHAS = [10**p for p in range(-4, 5)]
    fusion_results = []

    def get_features_and_meta(model, ds):
        gap_layer = model.get_layer("gap")
        feature_model = keras.Model(model.input, gap_layer.output)

        feats_list, y_list, c_list = [], [], []

        for img_b, y_b, c_b in tqdm(ds, leave=False, desc="Extracting features"):
            f = feature_model.predict(img_b, verbose=0)
            feats_list.append(f)
            y_list.append(y_b.numpy())
            c_list.append(c_b.numpy())

        if not feats_list:
            return np.array([]), np.array([]), np.array([])

        return (
            np.concatenate(feats_list, axis=0),
            np.concatenate(y_list, axis=0),
            np.concatenate(c_list, axis=0).astype(str)
        )

    for scheme in SCHEMES:
        for fold in FOLDS:
            print(f"\n=== RIDGE FUSION: {scheme} / Fold {fold} ===")

            row_ms = df_retrain[(df_retrain["scheme"]==scheme) & (df_retrain["fold"]==fold) & (df_retrain["variant"]=="ms")]
            row_nl = df_retrain[(df_retrain["scheme"]==scheme) & (df_retrain["fold"]==fold) & (df_retrain["variant"]=="nl")]

            if row_ms.empty or row_nl.empty:
                print(f"[SKIP] Missing MS or NL model for {scheme}/{fold}")
                continue

            path_ms = row_ms.iloc[0]["ckpt_path"]
            path_nl = row_nl.iloc[0]["ckpt_path"]

            folds_dict = get_folds_dict(scheme)
            idx_train = np.asarray(folds_dict[fold]["train"], dtype=np.int64)
            idx_val   = np.asarray(folds_dict[fold]["val"],   dtype=np.int64)
            idx_test  = np.asarray(folds_dict[fold]["test"],  dtype=np.int64)

            print("-> Processing MS (Train/Val/Test)...")
            ds_tr_ms   = build_dataset_with_country(idx_train, "ms")
            ds_val_ms  = build_dataset_with_country(idx_val,   "ms")
            ds_test_ms = build_dataset_with_country(idx_test,  "ms")

            model_ms = keras.models.load_model(path_ms, custom_objects={"PreActBlock": PreActBlock})
            X_tr_ms, y_tr, c_tr       = get_features_and_meta(model_ms, ds_tr_ms)
            X_val_ms, y_val, c_val    = get_features_and_meta(model_ms, ds_val_ms)
            X_test_ms, y_test, c_test = get_features_and_meta(model_ms, ds_test_ms)
            del model_ms; tf.keras.backend.clear_session()

            print("-> Processing NL (Train/Val/Test)...")
            ds_tr_nl   = build_dataset_with_country(idx_train, "nl")
            ds_val_nl  = build_dataset_with_country(idx_val,   "nl")
            ds_test_nl = build_dataset_with_country(idx_test,  "nl")

            model_nl = keras.models.load_model(path_nl, custom_objects={"PreActBlock": PreActBlock})
            X_tr_nl, _, _   = get_features_and_meta(model_nl, ds_tr_nl)
            X_val_nl, _, _  = get_features_and_meta(model_nl, ds_val_nl)
            X_test_nl, _, _ = get_features_and_meta(model_nl, ds_test_nl)
            del model_nl; tf.keras.backend.clear_session()

            X_tr_fused   = np.concatenate([X_tr_ms,   X_tr_nl],  axis=1)
            X_val_fused  = np.concatenate([X_val_ms,  X_val_nl], axis=1)
            X_test_fused = np.concatenate([X_test_ms, X_test_nl], axis=1)

            best_alpha = None
            best_val_r2 = -np.inf

            for alpha in RIDGE_ALPHAS:
                clf = Ridge(alpha=alpha)
                clf.fit(X_tr_fused, y_tr)
                val_pred = clf.predict(X_val_fused)
                r2 = r2_score(y_val, val_pred)

                if r2 > best_val_r2:
                    best_val_r2 = r2
                    best_alpha = alpha

            print(f"-> Best Alpha found on Validation: {best_alpha} (Val R2: {best_val_r2:.4f})")

            X_full = np.concatenate([X_tr_fused, X_val_fused], axis=0)
            y_full = np.concatenate([y_tr, y_val], axis=0)

            final_model = Ridge(alpha=best_alpha)
            final_model.fit(X_full, y_full)

            preds = final_model.predict(X_test_fused)

            df_fold_preds = pd.DataFrame({
                "country": c_test,
                "y_true": y_test,
                "y_pred": preds,
                "scheme": scheme,
                "fold": fold,
                "best_alpha": best_alpha
            })
            fusion_results.append(df_fold_preds)

            r2_test = r2_score(y_test, preds)
            print(f"-> Final Ridge Test R2: {r2_test:.4f}")

    if fusion_results:
        df_fusion_all = pd.concat(fusion_results, ignore_index=True)
        df_fusion_all.to_csv(fusion_out_csv, index=False)
        print(f"\nSaved all fusion predictions to: {fusion_out_csv}")
    else:
        print("No fusion results generated.")
        df_fusion_all = pd.DataFrame()

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

if 'df_fusion_all' not in globals() or df_fusion_all.empty:
    print("No fusion predictions found. Run Cell 16 first.")
else:

    report_rows = []

    for scheme in SCHEMES:
        df_s = df_fusion_all[df_fusion_all["scheme"] == scheme]
        countries = df_s["country"].unique()

        print(f"\n--- Scheme: {scheme.upper()} (R² per Country) ---")
        print(f"{'Country':<10} | {'N':<5} | {'R²':<7} | {'MAE':<7}")
        print("-" * 35)

        scheme_metrics = []

        for c in sorted(countries):
            sub = df_s[df_s["country"] == c]
            n = len(sub)
            if n < 2:
                r2 = np.nan
                mae = np.nan
            else:
                r2 = r2_score(sub["y_true"], sub["y_pred"])
                mae = mean_absolute_error(sub["y_true"], sub["y_pred"])

            print(f"{c:<10} | {n:<5} | {r2:6.3f}  | {mae:6.3f}")

            scheme_metrics.append({
                "scheme": scheme,
                "country": c,
                "n": n,
                "r2": r2,
                "mae": mae
            })

        report_rows.extend(scheme_metrics)

        g_r2 = r2_score(df_s["y_true"], df_s["y_pred"])
        print("-" * 35)
        print(f"{'GLOBAL':<10} | {len(df_s):<5} | {g_r2:6.3f}")

    df_country_report = pd.DataFrame(report_rows)
    report_csv = os.path.join(ROOT_OUT, "ridge_fusion_per_country_r2.csv")
    df_country_report.to_csv(report_csv, index=False)
    print(f"\nSaved per-country report to: {report_csv}")

    PLOTS_DIR = os.path.join(ROOT_OUT, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_data = df_country_report[df_country_report["n"] > 10].copy()

    pivot_df = plot_data.pivot(index="country", columns="scheme", values="r2")

    if "incountry" in pivot_df.columns:
        sorted_countries = pivot_df.sort_values("incountry", ascending=False).index.tolist()
    else:

        sorted_countries = pivot_df.mean(axis=1).sort_values(ascending=False).index.tolist()

    plt.figure(figsize=(12, 14))

    ax = sns.barplot(
        data=plot_data,
        y="country",
        x="r2",
        hue="scheme",
        order=sorted_countries,
        palette={"incountry": "#3498db", "ooc": "#e74c3c"},
        edgecolor="white",
        linewidth=0.5
    )

    plt.title("Ridge Fusion Performance: Incountry vs. Out-of-Country (OOC)", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("$R^2$ Score (Wealth Prediction)", fontsize=12)
    plt.ylabel("Country", fontsize=12)

    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.legend(loc="lower right", frameon=True)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.2f',
            label_type='edge',
            padding=3,
            fontsize=8,
            color='black'
        )

    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "ridge_fusion_grouped_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Grouped bar chart saved to: {out_path}")
    plt.show()

import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

r2_csv_path = os.path.join(ROOT_OUT, "ridge_fusion_per_country_r2.csv")
if not os.path.exists(r2_csv_path):
    raise FileNotFoundError("R² report not found. Please run Cell 17 first.")

df_r2 = pd.read_csv(r2_csv_path)

target_scheme = "incountry"
df_plot = df_r2[df_r2["scheme"] == target_scheme].copy()

if df_plot.empty:
    print(f"No data found for scheme '{target_scheme}'. Switching to first available scheme.")
    target_scheme = df_r2["scheme"].unique()[0]
    df_plot = df_r2[df_r2["scheme"] == target_scheme].copy()

NE110_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(NE110_URL)
africa = world[world["CONTINENT"] == "Africa"].to_crs("EPSG:4326")

africa["merge_code"] = africa["ISO_A2"]

africa_map = africa.merge(df_plot, left_on="merge_code", right_on="country", how="left")

fig, ax = plt.subplots(figsize=(14, 12))
ax.set_facecolor("#a6bddb")

africa.plot(ax=ax, color="#f0f0f0", edgecolor="white", linewidth=0.5)

africa_data = africa_map.dropna(subset=["r2"])

if not africa_data.empty:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    plot = africa_data.plot(
        column="r2",
        ax=ax,
        legend=True,
        cax=cax,
        cmap="RdYlGn",
        edgecolor="black",
        linewidth=0.5,
        vmin=0.0,
        vmax=0.8,
        legend_kwds={'label': f"$R^2$ Score ({target_scheme})"}
    )

if 'LOCS' in globals():
    ax.scatter(LOCS[:, 1], LOCS[:, 0], s=2, c='black', alpha=0.15, label='Clusters')

ax.set_xlim(-25, 55)
ax.set_ylim(-36, 38)
ax.set_xticks(np.arange(-20, 61, 10))
ax.set_yticks(np.arange(-30, 41, 10))
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Model Performance: $R^2$ by Country ({target_scheme.upper()})", fontsize=16)

out_path = os.path.join(DD_DIR, "africa_r2_map.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Map saved to: {out_path}")
plt.show()
