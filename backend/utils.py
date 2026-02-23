import os
import pickle
import numpy as np
import h5py

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence oneDNN info messages

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, Add
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")

# Prefer the patched model (in git repo), fall back to original (local only)
_patched = os.path.join(MODEL_DIR, "best_model_patched.h5")
_original = os.path.join(MODEL_DIR, "best_model.h5")
MODEL_PATH  = _patched if os.path.exists(_patched) else _original

TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# ── Architecture constants (read from the saved h5 config) ───────────────
VOCAB_SIZE  = 8485
MAX_LENGTH  = 35
EMBED_DIM   = 256
UNITS       = 256
FEATURE_DIM = 4096   # VGG16 fc2 output

# ── Globals (loaded once at startup) ────────────────────────────────────
_caption_model  = None
_tokenizer      = None
_vgg_model      = None
_reverse_lookup = None  # index → word dict, built once


def _build_caption_model() -> Model:
    """
    Rebuild the caption model and load weights directly from h5 via h5py.

    Verified original topology from model_config in best_model.h5:
        input_layer_1 (4096) → dropout  → Dense(256, relu)  [image branch]
        input_layer_2 (35)   → Embedding → dropout_1 → LSTM(256)  [text branch]
        Add([img_out, lstm]) → Dense(256, relu) → Dense(8485, softmax)
    """
    # Image feature branch
    img_in  = Input(shape=(FEATURE_DIM,), name="input_layer_1")
    img_d   = Dropout(0.4, name="dropout")(img_in)
    img_out = Dense(UNITS, activation="relu", name="img_dense")(img_d)

    # Text / sequence branch
    seq_in  = Input(shape=(MAX_LENGTH,), name="input_layer_2")
    emb     = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=False, name="embedding")(seq_in)
    seq_d   = Dropout(0.4, name="dropout_1")(emb)
    lstm    = LSTM(UNITS, name="lstm")(seq_d)

    # Merge & decode
    merged  = Add(name="add")([img_out, lstm])
    dec     = Dense(UNITS, activation="relu", name="dense_1")(merged)
    out     = Dense(VOCAB_SIZE, activation="softmax", name="dense_2")(dec)

    model = Model(inputs=[img_in, seq_in], outputs=out)

    # ── Load weights directly from h5 via h5py ───────────────────────────
    with h5py.File(MODEL_PATH, "r") as f:
        mw = f["model_weights"]

        emb_w = mw["embedding"]["embedding"]["embeddings"][:]
        model.get_layer("embedding").set_weights([emb_w])
        print(f"[INFO] embedding  {emb_w.shape}")

        lc    = mw["lstm"]["lstm"]["lstm_cell"]
        model.get_layer("lstm").set_weights([lc["kernel"][:], lc["recurrent_kernel"][:], lc["bias"][:]])
        print(f"[INFO] lstm       kernel={lc['kernel'].shape}")

        dg    = mw["dense"]["dense"]
        model.get_layer("img_dense").set_weights([dg["kernel"][:], dg["bias"][:]])
        print(f"[INFO] img_dense  {dg['kernel'].shape}")

        d1    = mw["dense_1"]["dense_1"]
        model.get_layer("dense_1").set_weights([d1["kernel"][:], d1["bias"][:]])
        print(f"[INFO] dense_1    {d1['kernel'].shape}")

        d2    = mw["dense_2"]["dense_2"]
        model.get_layer("dense_2").set_weights([d2["kernel"][:], d2["bias"][:]])
        print(f"[INFO] dense_2    {d2['kernel'].shape}")

    print("[INFO] All weights loaded from best_model.h5")
    return model


def _get_vgg_model() -> Model:
    """Return a VGG16 model truncated at the fc2 layer (4096-d features)."""
    global _vgg_model
    if _vgg_model is None:
        base       = VGG16(weights="imagenet")
        _vgg_model = Model(inputs=base.input, outputs=base.layers[-2].output)
        print("[INFO] VGG16 feature extractor ready.")
    return _vgg_model


def load_model_and_tokenizer():
    """Load the caption model and tokenizer (cached globally after first call)."""
    global _caption_model, _tokenizer, _reverse_lookup

    if _caption_model is None:
        _caption_model = _build_caption_model()

    if _tokenizer is None:
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
        # Build fast O(1) reverse lookup once
        _reverse_lookup = {idx: word for word, idx in _tokenizer.word_index.items()}
        print(f"[INFO] Tokenizer loaded — vocab size: {len(_tokenizer.word_index) + 1}")

    return _caption_model, _tokenizer


def extract_features(image_path: str) -> np.ndarray:
    """Extract a 4096-d feature vector from an image file using VGG16."""
    vgg   = _get_vgg_model()
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return vgg.predict(image, verbose=0)   # shape (1, 4096)


def generate_caption(
    model,
    tokenizer,
    photo_features: np.ndarray,
    max_length: int = MAX_LENGTH,
) -> str:
    """
    Greedy-decode a caption given VGG16 photo features.

    Improvements:
    - Skips 'endseq' at step 0 (picks next-best real word instead).
    - Prevents immediate word repetition (stops 'the the the' loops).
    - Uses an O(1) reverse index->word lookup built at load time.
    """
    global _reverse_lookup
    if _reverse_lookup is None:
        _reverse_lookup = {idx: word for word, idx in tokenizer.word_index.items()}

    in_text   = "startseq"
    last_word = None

    for step in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Input order matches Model(inputs=[img_in, seq_in])
        yhat = model.predict([photo_features, sequence], verbose=0)[0]

        # Walk sorted predictions, skip invalid candidates
        for idx in np.argsort(yhat)[::-1]:
            word = _reverse_lookup.get(int(idx))
            if word is None:
                continue
            if step == 0 and word == "endseq":   # never start with endseq
                continue
            if word == last_word:                 # no immediate repetition
                continue
            break
        else:
            break   # nothing valid found

        if word == "endseq":
            break

        in_text   += " " + word
        last_word  = word

    return in_text.replace("startseq", "").replace("endseq", "").strip()
