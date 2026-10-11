"""Traditional-ML baselines: per-task TF-IDF + LogisticRegression / DecisionTree.

Leakage-safe: receives the split dataframes and task targets; text features are
built from title/description/review text only (no ratings/prices/categories).
TF-IDF is fit on the training portion of each task's text ONLY.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from model_utils import leak_free_features, prepare_targets


@dataclass
class BaselineResult:
    task: str
    model: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    train_seconds: float
    n_train: int
    n_test: int
    confusion: list = field(default_factory=list)
    labels: list = field(default_factory=list)


def _evaluate(task, model_name, pipe, Xtr, ytr, Xte, yte, t0, labels_order=None):
    pipe.fit(Xtr, ytr)
    secs = time.time() - t0
    pred = pipe.predict(Xte)
    labels = sorted(set(ytr) | set(yte))
    r = BaselineResult(
        task=task, model=model_name,
        accuracy=accuracy_score(yte, pred),
        precision_macro=precision_score(yte, pred, labels=labels, average="macro", zero_division=0),
        recall_macro=recall_score(yte, pred, labels=labels, average="macro", zero_division=0),
        f1_macro=f1_score(yte, pred, labels=labels, average="macro", zero_division=0),
        train_seconds=secs, n_train=len(Xtr), n_test=len(Xte),
        confusion=confusion_matrix(yte, pred, labels=labels).tolist(),
        labels=labels,
    )
    if labels_order:
        r.labels = labels_order
    return r


def run_baselines(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                  tasks=("category", "price_tier", "sentiment"), seed: int = 20231010):
    """Returns dict of BaselineResult keyed by (task, model_name).

    Protocol per task:
      1. selection: TF-IDF fit on train text; LogReg fit on train; score on val
      2. final: LogReg re-fit on train+val; score on test (TF-IDF still train-fit)
      3. DecisionTree baseline: TF-IDF + scaled SVD/no SVD; per task, fitted on train, scored on val/test
    """
    results: dict[tuple[str, str], BaselineResult] = {}
    target_cols = {"category": "category", "price_tier": "price_tier", "sentiment": "sentiment_label"}

    for task in tasks:
        ytr = train[target_cols[task]].astype(str)
        yva = val[target_cols[task]].astype(str)
        yte = test[target_cols[task]].astype(str)

        tr_txt_all, _ = leak_free_features(train, task)
        va_txt_all, _ = leak_free_features(val, task)
        te_txt_all, _ = leak_free_features(test, task)

        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(tr_txt_all)
        Xva = vec.transform(va_txt_all)
        Xte = vec.transform(te_txt_all)

        # LogReg: fit on train (selection on val)
        t0 = time.time()
        lr_sel = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        lr_sel.fit(Xtr, ytr)
        secs_sel = time.time() - t0
        labels = sorted(set(ytr) | set(yva) | set(yte))
        pred_va = lr_sel.predict(Xva)
        results[(task, "logreg_val")] = BaselineResult(
            task=task, model="tfidf_logreg",
            accuracy=accuracy_score(yva, pred_va),
            precision_macro=precision_score(yva, pred_va, labels=labels, average="macro", zero_division=0),
            recall_macro=recall_score(yva, pred_va, labels=labels, average="macro", zero_division=0),
            f1_macro=f1_score(yva, pred_va, labels=labels, average="macro", zero_division=0),
            train_seconds=secs_sel, n_train=Xtr.shape[0], n_test=Xva.shape[0],
            confusion=confusion_matrix(yva, pred_va, labels=labels).tolist(),
            labels=labels)

        # LogReg: re-fit on train+val; scored on test
        from scipy.sparse import vstack
        t0 = time.time()
        joint_X = vstack([Xtr, Xva])
        joint_y = pd.concat([ytr, yva], axis=0)
        lr_test = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        lr_test.fit(joint_X, joint_y)
        secs_test = time.time() - t0
        pred_te = lr_test.predict(Xte)
        results[(task, "logreg_test")] = BaselineResult(
            task=task, model="tfidf_logreg",
            accuracy=accuracy_score(yte, pred_te),
            precision_macro=precision_score(yte, pred_te, labels=labels, average="macro", zero_division=0),
            recall_macro=recall_score(yte, pred_te, labels=labels, average="macro", zero_division=0),
            f1_macro=f1_score(yte, pred_te, labels=labels, average="macro", zero_division=0),
            train_seconds=secs_sel + secs_test, n_train=joint_X.shape[0], n_test=Xte.shape[0],
            confusion=confusion_matrix(yte, pred_te, labels=labels).tolist(),
            labels=labels)

        # Decision Tree on SVD of TF-IDF (train-fit)
        svd = TruncatedSVD(n_components=100, random_state=seed)
        Ztr = svd.fit_transform(Xtr)
        Zva = svd.transform(Xva)
        Zte = svd.transform(Xte)
        t0 = time.time()
        dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed)
        dt.fit(Ztr, ytr)
        secs_dt = time.time() - t0
        pred_dt = dt.predict(Zte)
        results[(task, "decision_tree_test")] = BaselineResult(
            task=task, model="tfidf_svd100_decisiontree",
            accuracy=accuracy_score(yte, pred_dt),
            precision_macro=precision_score(yte, pred_dt, labels=labels, average="macro", zero_division=0),
            recall_macro=recall_score(yte, pred_dt, labels=labels, average="macro", zero_division=0),
            f1_macro=f1_score(yte, pred_dt, labels=labels, average="macro", zero_division=0),
            train_seconds=secs_dt, n_train=Ztr.shape[0], n_test=Zte.shape[0],
            confusion=confusion_matrix(yte, pred_dt, labels=labels).tolist(),
            labels=labels)

    return results


def results_frame(results: dict) -> "pd.DataFrame":
    rows = []
    for (task, model), r in results.items():
        rows.append({"task": task, "model": model, "accuracy": round(r.accuracy, 4),
                     "precision_macro": round(r.precision_macro, 4),
                     "recall_macro": round(r.recall_macro, 4),
                     "f1_macro": round(r.f1_macro, 4),
                     "train_seconds": round(r.train_seconds, 2),
                     "n_train": r.n_train, "n_test": r.n_test})
    import pandas as pd
    return pd.DataFrame(rows).sort_values(["task", "model"]).reset_index(drop=True)
