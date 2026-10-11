"""Shared-encoder multi-task learning model (v2: checkpoint + memory-safe full-batch inference)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder

from model_utils import leak_free_features

SEED = 20231010
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data/models"


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class MTLConfig:
    hidden_dims: tuple = (256, 128)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 40
    task_weights: dict = field(default_factory=lambda: {"category": 1.0, "price_tier": 1.0, "sentiment": 1.0})
    max_features: int = 5000
    seed: int = SEED


class SharedEncoderMTL(nn.Module):
    def __init__(self, in_dim: int, n_cat: int, n_tier: int, n_sent: int,
                 hidden_dims=(256, 128), dropout: float = 0.2):
        super().__init__()
        enc = []
        d = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.encoder = nn.Sequential(*enc)
        self.head_cat = nn.Linear(d, n_cat)
        self.head_tier = nn.Linear(d, n_tier)
        self.head_sent = nn.Linear(d, n_sent)

    def forward(self, x):
        h = self.encoder(x)
        return self.head_cat(h), self.head_tier(h), self.head_sent(h)


def _dense(X, max_rows: int = 200_000):
    """Convert sparse matrix or ndarray to a dense float32 numpy array."""
    if hasattr(X, "toarray"):
        if X.shape[0] > max_rows:
            return np.vstack([X[a:b].toarray() for a, b in zip(range(0, X.shape[0], max_rows),
                                                               range(max_rows, X.shape[0] + max_rows, max_rows))])
        return X.toarray()
    return np.asarray(X)


def _encode_labels(le, values, fallback=None):
    classes = set(le.classes_)
    if fallback is None:
        return le.transform(values)
    return le.transform([v if v in classes else fallback for v in values])


class MTLResult:
    def __init__(self):
        self.history: list[dict] = []
        self.test_metrics: dict = {}
        self.val_metrics_best: dict = {}
        self.train_seconds: float = 0.0
        self.model_size_bytes: int = 0
        self.confusions: dict = {}
        self.labels: dict = {}
        self.config: dict = {}


def train_mtl(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
              cfg: MTLConfig | None = None, device: str | None = None,
              save_checkpoint: bool = True) -> MTLResult:
    """Train shared-encoder MTL; select best epoch on val; evaluate once on test."""
    cfg = cfg or MTLConfig()
    seed_everything(cfg.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tr_txt, _ = leak_free_features(train, "category")
    va_txt, _ = leak_free_features(val, "category")
    te_txt, _ = leak_free_features(test, "category")

    vec = TfidfVectorizer(max_features=cfg.max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = _dense(vec.fit_transform(tr_txt)).astype(np.float32)
    Xva = _dense(vec.transform(va_txt)).astype(np.float32)
    Xte = _dense(vec.transform(te_txt)).astype(np.float32)

    le_cat = LabelEncoder().fit(train.category.astype(str))
    le_tier = LabelEncoder().fit(["low", "medium", "high"])
    le_sent = LabelEncoder().fit(["negative", "neutral", "positive"])

    ycat_tr = _encode_labels(le_cat, train.category.astype(str).values, le_cat.classes_[0])
    ycat_va = _encode_labels(le_cat, val.category.astype(str).values, le_cat.classes_[0])
    ycat_te = _encode_labels(le_cat, test.category.astype(str).values, le_cat.classes_[0])
    ytier_tr = _encode_labels(le_tier, train.price_tier.astype(str).values, "medium")
    ytier_va = _encode_labels(le_tier, val.price_tier.astype(str).values, "medium")
    ytier_te = _encode_labels(le_tier, test.price_tier.astype(str).values, "medium")
    ysent_tr = _encode_labels(le_sent, train.sentiment_label.astype(str).values, "positive")
    ysent_va = _encode_labels(le_sent, val.sentiment_label.astype(str).values, "positive")
    ysent_te = _encode_labels(le_sent, test.sentiment_label.astype(str).values, "positive")

    model = SharedEncoderMTL(Xtr.shape[1], len(le_cat.classes_), len(le_tier.classes_),
                             len(le_sent.classes_), cfg.hidden_dims, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    ws = cfg.task_weights

    def macro_f1s(logits, *pairs):
        preds = [lg.argmax(1).cpu().numpy() for lg in logits]
        return tuple(f1_score(y, p, average="macro", zero_division=0) for (y, p) in zip(pairs, preds))

    Xtr_t = torch.tensor(Xtr, device=device)
    Ytr = [torch.tensor(y, device=device) for y in (ycat_tr, ytier_tr, ysent_tr)]
    Xva_t = torch.tensor(Xva, device=device)
    n = Xtr_t.shape[0]
    if device.type == "cuda" and n > 4000:
        # with only ~3.4k rows and 5k TF-IDF features, single GPU tensors are fine
        pass

    res = MTLResult()
    res.config = {**vars(cfg), "device": str(device), "in_dim": Xtr.shape[1],
                  "n_cat": len(le_cat.classes_), "n_tier": len(le_tier.classes_),
                  "n_sent": len(le_sent.classes_)}
    res.labels = {"category": list(le_cat.classes_), "price_tier": list(le_tier.classes_),
                  "sentiment": list(le_sent.classes_)}

    best_val, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + epoch)
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, cfg.batch_size):
            bidx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            outs = model(Xtr_t[bidx])
            loss = sum(w * crit(o, y[bidx]) for w, o, y in
                       zip((ws["category"], ws["price_tier"], ws["sentiment"]), outs, Ytr))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            outs_va = model(Xva_t)
        f1s = [f1_score(yv, lg.argmax(1).cpu().numpy(), average="macro", zero_division=0)
               for yv, lg in zip((ycat_va, ytier_va, ysent_va), outs_va)]
        res.history.append({"epoch": epoch + 1, "f1_category": f1s[0], "f1_price_tier": f1s[1],
                            "f1_sentiment": f1s[2], "mean_f1": float(np.mean(f1s))})
        if np.mean(f1s) > best_val:
            best_val = float(np.mean(f1s))
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    res.train_seconds = time.time() - t0
    res.val_metrics_best = {"mean_f1": best_val}

    model.load_state_dict(best_state)
    model.eval()

    # single-shot test inference (test set is small)
    Xte_t = torch.tensor(Xte, device=device)
    with torch.no_grad():
        outs_te = model(Xte_t)
    preds = [lg.argmax(1).cpu().numpy() for lg in outs_te]

    for (name, ytrue, ypred, lg) in zip(
        ("category", "price_tier", "sentiment"),
        (ycat_te, ytier_te, ysent_te), preds, outs_te):
        res.test_metrics[name] = {
            "accuracy": accuracy_score(ytrue, ypred),
            "precision_macro": precision_score(ytrue, ypred, average="macro", zero_division=0),
            "recall_macro": recall_score(ytrue, ypred, average="macro", zero_division=0),
            "f1_macro": f1_score(ytrue, ypred, average="macro", zero_division=0),
        }
        res.confusions[name] = confusion_matrix(ytrue, ypred).tolist()

    res.model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    if save_checkpoint:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = CHECKPOINT_DIR / "mtl_best.pt"
        torch.save(best_state, ckpt_path)
        cfg_path = CHECKPOINT_DIR / "mtl_config.json"
        cfg_path.write_text(json.dumps(res.config, indent=1, default=str))
    return res
