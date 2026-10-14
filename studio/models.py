"""Model supervised, clustering, dan deep learning (PyTorch)."""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REG_MODELS = ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"]
CLF_MODELS = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
CLUSTER_ALGOS = ["K-Means", "Agglomerative", "DBSCAN"]

MAX_AGGLOMERATIVE_ROWS = 5000


def _estimator(kind: str, name: str, rs: int):
    if kind == "regression":
        return {
            "Linear Regression": lambda: LinearRegression(),
            "Ridge": lambda: Ridge(alpha=1.0, random_state=rs),
            "Lasso": lambda: Lasso(alpha=0.1, random_state=rs),
            "Decision Tree": lambda: DecisionTreeRegressor(random_state=rs),
            "Random Forest": lambda: RandomForestRegressor(random_state=rs, n_jobs=-1),
        }[name]()
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=rs),
        "KNN": lambda: KNeighborsClassifier(),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=rs),
        "Random Forest": lambda: RandomForestClassifier(random_state=rs, n_jobs=-1),
        "SVM": lambda: SVC(random_state=rs),
    }[name]()


def _xy(df: pd.DataFrame):
    return df.drop(columns=["TARGET"]), df["TARGET"]


def _split(X, y, kind: str, test_size: float, rs: int):
    strat = y if kind == "classification" else None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=rs, stratify=strat)
    except ValueError:  # kelas terlalu sedikit untuk stratifikasi
        return train_test_split(X, y, test_size=test_size, random_state=rs)


def _reg_eval(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R² Score": r2_score(y_true, y_pred),
    }


def _clf_eval(y_true, y_pred, names: list[str]):
    labels = list(range(len(names)))
    metrics = {
        "Akurasi": accuracy_score(y_true, y_pred),
        "Presisi": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Skor F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, report, cm


def _importances(pipe: Pipeline, features: list[str]):
    est = pipe.named_steps["model"]
    vals = None
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = np.abs(np.asarray(est.coef_))
        vals = coef.mean(axis=0) if coef.ndim > 1 else coef
    if vals is None or len(vals) != len(features):
        return None
    return pd.Series(vals, index=features).sort_values(ascending=False)


def _base_result(mode: str, dataset: str, model: str, kind: str) -> dict:
    return {
        "mode": mode,
        "dataset": dataset,
        "model": model,
        "kind": kind,
        "trained_at": datetime.now().strftime("%H:%M:%S"),
    }


# ------------------------------------------------------------------
# Regresi dan klasifikasi klasik
# ------------------------------------------------------------------
def train_supervised(df, kind, names, model_name, dataset, test_size, rs) -> dict:
    mode = "Regresi" if kind == "regression" else "Klasifikasi"
    X, y = _xy(df)
    X_tr, X_te, y_tr, y_te = _split(X, y, kind, test_size, rs)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", _estimator(kind, model_name, rs))]
    )
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    seconds = time.perf_counter() - t0

    res = _base_result(mode, dataset, model_name, kind)
    res.update(
        y_test=y_te.to_numpy(),
        y_pred=np.asarray(y_pred),
        labels=names,
        importances=_importances(pipe, list(X.columns)),
        seconds=seconds,
        n_train=len(X_tr),
        n_test=len(X_te),
        history=None,
        report=None,
        cm=None,
    )
    if kind == "regression":
        res["metrics"] = _reg_eval(y_te, y_pred)
    else:
        res["metrics"], res["report"], res["cm"] = _clf_eval(y_te, y_pred, names)
    return res


# ------------------------------------------------------------------
# Deep learning ringan (PyTorch MLP)
# ------------------------------------------------------------------
def train_deep_learning(df, kind, names, cfg: dict, dataset, test_size, rs, progress=None):
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch belum terpasang. Jalankan: pip install torch") from exc

    torch.manual_seed(rs)
    np.random.seed(rs)
    X, y = _xy(df)
    X_tr, X_te, y_tr, y_te = _split(X, y, kind, test_size, rs)
    scaler = StandardScaler().fit(X_tr)
    xtr = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
    xte = torch.tensor(scaler.transform(X_te), dtype=torch.float32)

    mu, sd = 0.0, 1.0
    if kind == "classification":
        n_out = len(names)
        ytr = torch.tensor(y_tr.to_numpy(), dtype=torch.long)
        yte = torch.tensor(y_te.to_numpy(), dtype=torch.long)
        crit = nn.CrossEntropyLoss()
    else:
        n_out = 1
        mu = float(y_tr.mean())
        sd = float(y_tr.std()) or 1.0
        ytr = torch.tensor((y_tr.to_numpy() - mu) / sd, dtype=torch.float32).unsqueeze(1)
        yte = torch.tensor((y_te.to_numpy() - mu) / sd, dtype=torch.float32).unsqueeze(1)
        crit = nn.MSELoss()

    layers: list = []
    d = xtr.shape[1]
    for _ in range(cfg["layers"]):
        layers += [nn.Linear(d, cfg["neurons"]), nn.ReLU(), nn.Dropout(cfg["dropout"])]
        d = cfg["neurons"]
    layers.append(nn.Linear(d, n_out))
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=1e-5)

    n, bs, epochs = len(xtr), cfg["batch"], cfg["epochs"]
    history = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = crit(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            out = net(xte)
            val_loss = crit(out, yte).item()
        row = {"epoch": ep + 1, "train_loss": total / n, "val_loss": val_loss}
        if kind == "classification":
            row["val_acc"] = (out.argmax(1) == yte).float().mean().item()
        history.append(row)
        if progress:
            progress((ep + 1) / epochs, row)
    seconds = time.perf_counter() - t0

    with torch.no_grad():
        out = net(xte)
    if kind == "classification":
        y_pred = out.argmax(1).numpy()
    else:
        y_pred = out.squeeze(1).numpy() * sd + mu

    arch = f"MLP {cfg['layers']}×{cfg['neurons']}"
    res = _base_result("Deep Learning", dataset, arch, kind)
    res.update(
        y_test=y_te.to_numpy(),
        y_pred=np.asarray(y_pred),
        labels=names,
        importances=None,
        seconds=seconds,
        n_train=len(X_tr),
        n_test=len(X_te),
        history=pd.DataFrame(history),
        n_params=int(sum(p.numel() for p in net.parameters())),
        cfg=dict(cfg),
        report=None,
        cm=None,
    )
    if kind == "regression":
        res["metrics"] = _reg_eval(y_te, y_pred)
    else:
        res["metrics"], res["report"], res["cm"] = _clf_eval(y_te, y_pred, names)
    return res


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------
def _cluster_name(label: int) -> str:
    return "Noise" if label == -1 else f"Cluster {label + 1}"


def _safe_scores(Xs: np.ndarray, labels: np.ndarray, rs: int) -> dict:
    mask = labels != -1
    uniq = np.unique(labels[mask])
    out = {"Silhouette": None, "Davies-Bouldin": None, "Calinski-Harabasz": None}
    if len(uniq) < 2 or mask.sum() <= len(uniq):
        return out
    Xm, lm = Xs[mask], labels[mask]
    out["Silhouette"] = float(
        silhouette_score(Xm, lm, sample_size=min(len(lm), 3000), random_state=rs)
    )
    out["Davies-Bouldin"] = float(davies_bouldin_score(Xm, lm))
    out["Calinski-Harabasz"] = float(calinski_harabasz_score(Xm, lm))
    return out


def run_clustering(df, kind, names, algo, k, eps, min_samples, dataset, rs) -> dict:
    X, y = _xy(df)
    sampled = False
    if algo == "Agglomerative" and len(X) > MAX_AGGLOMERATIVE_ROWS:
        idx = np.random.RandomState(rs).choice(len(X), MAX_AGGLOMERATIVE_ROWS, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        sampled = True

    Xs = StandardScaler().fit_transform(X)
    t0 = time.perf_counter()
    if algo == "K-Means":
        labels = KMeans(n_clusters=k, n_init=10, random_state=rs).fit_predict(Xs)
    elif algo == "Agglomerative":
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    seconds = time.perf_counter() - t0

    pca = PCA(n_components=2, random_state=rs)
    coords = pca.fit_transform(Xs)

    n_clusters = len(set(labels) - {-1})
    metrics: dict = {"Jumlah Cluster": n_clusters}
    metrics.update(_safe_scores(Xs, labels, rs))
    if algo == "DBSCAN":
        metrics["Noise"] = int((labels == -1).sum())
    if kind == "classification":
        metrics["ARI vs label asli"] = float(adjusted_rand_score(y, labels))

    cluster_names = np.array([_cluster_name(int(lb)) for lb in labels])
    order = sorted(set(cluster_names), key=lambda s: (s == "Noise", s))
    sizes = pd.Series(cluster_names).value_counts().reindex(order)

    prof = X.copy()
    prof["cluster"] = cluster_names
    profile = prof.groupby("cluster").mean().reindex(order)
    z = (profile - X.mean()) / X.std().replace(0, 1)

    crosstab = None
    if kind == "classification" and names:
        crosstab = pd.crosstab(
            pd.Series(cluster_names, name="Cluster"),
            pd.Series([names[int(v)] for v in y], name="Label asli"),
        ).reindex(order)

    elbow = None
    if algo == "K-Means":
        rows = []
        for kk in range(2, min(10, len(Xs) - 1) + 1):
            km = KMeans(n_clusters=kk, n_init=5, random_state=rs).fit(Xs)
            sil = silhouette_score(
                Xs, km.labels_, sample_size=min(len(Xs), 2000), random_state=rs
            )
            rows.append({"k": kk, "inertia": float(km.inertia_), "silhouette": float(sil)})
        elbow = pd.DataFrame(rows)

    res = _base_result("Clustering", dataset, algo, kind)
    res.update(
        metrics=metrics,
        coords=coords,
        labels_arr=cluster_names,
        var_ratio=pca.explained_variance_ratio_,
        sizes=sizes,
        profile=profile,
        profile_z=z,
        crosstab=crosstab,
        elbow=elbow,
        k=k,
        seconds=seconds,
        n_rows=len(X),
        n_features=X.shape[1],
        sampled=sampled,
    )
    return res
