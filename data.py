"""MovieLens-1M loading and sequence construction.

Item ids are remapped to 1..num_items; 0 is reserved for padding.
Split is leave-one-out (the standard protocol for SASRec/BERT4Rec):
    last item  -> test
    2nd last   -> validation
    remainder  -> training
"""

import os
import subprocess
import zipfile
import urllib.request

import numpy as np

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_ml1m(root="data"):
    """Download and unzip MovieLens-1M if it is not already present."""
    os.makedirs(root, exist_ok=True)
    target_dir = os.path.join(root, "ml-1m")
    ratings = os.path.join(target_dir, "ratings.dat")
    if os.path.exists(ratings):
        return target_dir

    zip_path = os.path.join(root, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens-1M from {ML1M_URL} ...")
        try:
            urllib.request.urlretrieve(ML1M_URL, zip_path)
        except Exception as exc:
            # Some Windows Python installs have no CA bundle configured, so
            # urllib raises SSLCertVerificationError. curl ships with Windows
            # 10+ and uses the system trust store, so fall back to it.
            print(f"urllib failed ({exc.__class__.__name__}); falling back to curl ...")
            rc = subprocess.call(["curl", "-sSL", "-o", zip_path, ML1M_URL])
            if rc != 0 or not os.path.exists(zip_path):
                raise RuntimeError(
                    f"Could not download {ML1M_URL}. Download it manually and "
                    f"place it at {zip_path}."
                ) from exc
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return target_dir


def load_sequences(root="data", min_len=5):
    """Return (sequences, item_titles, num_items).

    sequences: list of lists of item ids, ordered by timestamp, one per user.
    """
    target_dir = download_ml1m(root)

    # ratings.dat: UserID::MovieID::Rating::Timestamp
    raw = {}
    with open(os.path.join(target_dir, "ratings.dat"), encoding="latin-1") as f:
        for line in f:
            user, item, _rating, ts = line.strip().split("::")
            raw.setdefault(int(user), []).append((int(ts), int(item)))

    # Keep only users with enough history to form train/valid/test.
    raw = {u: v for u, v in raw.items() if len(v) >= min_len}

    # Remap item ids to a dense 1..N range (0 = padding).
    all_items = sorted({item for events in raw.values() for _, item in events})
    item2idx = {item: i + 1 for i, item in enumerate(all_items)}

    sequences = []
    for user in sorted(raw):
        events = sorted(raw[user])  # chronological
        sequences.append([item2idx[item] for _, item in events])

    # movies.dat: MovieID::Title::Genres  (nice-to-have for readable demo output)
    titles = {}
    movies_path = os.path.join(target_dir, "movies.dat")
    if os.path.exists(movies_path):
        with open(movies_path, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 2 and int(parts[0]) in item2idx:
                    titles[item2idx[int(parts[0])]] = parts[1]

    return sequences, titles, len(all_items)


def split(sequences):
    """Leave-one-out split. Returns (train, valid_targets, test_targets)."""
    train, valid, test = [], [], []
    for seq in sequences:
        train.append(seq[:-2])
        valid.append(seq[-2])
        test.append(seq[-1])
    return train, valid, test


def make_training_pairs(train_seqs, max_len=20):
    """Expand each user history into (context, next_item) pairs.

    [A]       -> B
    [A,B]     -> C
    [A,B,C]   -> D
    Contexts are left-padded with 0 to max_len.
    """
    contexts, targets = [], []
    for seq in train_seqs:
        for t in range(1, len(seq)):
            ctx = seq[max(0, t - max_len):t]
            padded = [0] * (max_len - len(ctx)) + ctx
            contexts.append(padded)
            targets.append(seq[t])
    return np.array(contexts, dtype=np.int64), np.array(targets, dtype=np.int64)


def make_eval_contexts(train_seqs, max_len=20, extra=None):
    """Build the context used to predict the held-out item for each user.

    extra: optional list of items to append first (used so that the test
    context legitimately includes the validation item).
    """
    contexts = []
    for i, seq in enumerate(train_seqs):
        full = seq + ([extra[i]] if extra is not None else [])
        ctx = full[-max_len:]
        contexts.append([0] * (max_len - len(ctx)) + ctx)
    return np.array(contexts, dtype=np.int64)
