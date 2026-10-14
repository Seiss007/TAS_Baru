"""Katalog dataset: scikit-learn, UCI Machine Learning Repository, dan API real-time.

Semua loader mengembalikan tuple ``(df, target_names, info)`` dengan:
- ``df``           : DataFrame numerik, kolom target bernama ``TARGET``
- ``target_names`` : daftar nama kelas (klasifikasi) atau ``None`` (regresi)
- ``info``         : dict metadata (sumber, waktu pengambilan data, dsb.)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn import datasets as skd

CITIES = {
    "Salatiga": (-7.33, 110.50),
    "Jakarta": (-6.21, 106.85),
    "Bandung": (-6.91, 107.61),
    "Yogyakarta": (-7.80, 110.36),
    "Surabaya": (-7.25, 112.75),
    "Denpasar": (-8.65, 115.22),
    "Medan": (3.59, 98.67),
    "Tokyo": (35.68, 139.69),
}

QUAKE_CLASSES = ["Mikro (<1)", "Kecil (1–2.5)", "Sedang (2.5–4.5)", "Kuat (≥4.5)"]

_SKLEARN = {
    "Diabetes": dict(
        kind="regression",
        loader=skd.load_diabetes,
        desc="442 pasien, 10 fitur klinis. Target: progres penyakit setelah satu tahun.",
    ),
    "California Housing": dict(
        kind="regression",
        loader=skd.fetch_california_housing,
        desc="20.640 distrik di California. Target: median harga rumah.",
    ),
    "Iris": dict(
        kind="classification",
        loader=skd.load_iris,
        desc="150 bunga iris, 3 spesies, 4 ukuran kelopak dan mahkota.",
    ),
    "Breast Cancer": dict(
        kind="classification",
        loader=skd.load_breast_cancer,
        desc="569 pasien, 30 fitur citra sel. Target: ganas atau jinak.",
    ),
    "Wine": dict(
        kind="classification",
        loader=skd.load_wine,
        desc="178 sampel anggur, 13 fitur kimia, 3 kultivar.",
    ),
}

_UCI = {
    "Abalone": dict(
        id=1,
        kind="regression",
        desc="4.177 siput abalone. Target: jumlah cincin (perkiraan usia).",
    ),
    "Auto MPG": dict(
        id=9,
        kind="regression",
        desc="Konsumsi bahan bakar mobil (mpg) dari spesifikasi mesin dan bobot.",
    ),
    "Wine Quality": dict(
        id=186,
        kind="regression",
        desc="6.497 anggur vinho verde, 11 fitur fisikokimia. Target: skor kualitas.",
    ),
    "Heart Disease": dict(
        id=45,
        kind="classification",
        binarize=True,
        names=["Tanpa penyakit", "Penyakit jantung"],
        desc="Data klinis pasien (Cleveland). Target: ada atau tidaknya penyakit jantung.",
    ),
    "Seeds": dict(
        id=236,
        kind="classification",
        desc="210 biji gandum, 7 ukuran geometris, 3 varietas.",
    ),
    "Banknote Authentication": dict(
        id=267,
        kind="classification",
        desc="1.372 citra uang kertas, 4 fitur wavelet. Target: asli atau palsu.",
    ),
}

DATASETS: dict[str, dict] = {}
for _n, _m in _SKLEARN.items():
    DATASETS[_n] = dict(kind=_m["kind"], source="scikit-learn", loader="sklearn", desc=_m["desc"])
for _n, _m in _UCI.items():
    DATASETS[_n] = dict(kind=_m["kind"], source="UCI", loader="uci", desc=_m["desc"])

DATASETS["Cuaca · Open-Meteo"] = dict(
    kind="regression",
    source="Real-time",
    loader="weather",
    needs_city=True,
    desc="30 hari data cuaca per jam dari Open-Meteo. Target: suhu udara.",
)
DATASETS["Gempa · USGS"] = dict(
    kind="regression",
    source="Real-time",
    loader="quake",
    desc="Gempa bumi 30 hari terakhir seluruh dunia (USGS). Target: magnitudo.",
)
DATASETS["Gempa · USGS (Kelas Kekuatan)"] = dict(
    kind="classification",
    source="Real-time",
    loader="quake",
    desc="Gempa bumi 30 hari terakhir (USGS). Target: kelas kekuatan gempa.",
)


def datasets_for(mode: str) -> list[str]:
    """Daftar dataset yang cocok untuk sebuah mode."""
    if mode == "Regresi":
        return [n for n, m in DATASETS.items() if m["kind"] == "regression"]
    if mode == "Klasifikasi":
        return [n for n, m in DATASETS.items() if m["kind"] == "classification"]
    return list(DATASETS)


def count_realtime() -> int:
    return sum(1 for m in DATASETS.values() if m["source"] == "Real-time")


# ------------------------------------------------------------------
# Pembersihan umum
# ------------------------------------------------------------------
def _finalize(X: pd.DataFrame, y, kind: str, class_names: list[str] | None = None):
    """Ubah fitur menjadi numerik, isi nilai hilang, dan encode target."""
    X = pd.get_dummies(X.copy(), drop_first=True, dtype=float)
    X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y) if not isinstance(y, pd.Series) else y.to_numpy())

    if kind == "regression":
        y = pd.to_numeric(y, errors="coerce")
        names = None
    else:
        names = None
    keep = y.notna().to_numpy()
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    if kind == "classification":
        codes, uniques = pd.factorize(y, sort=True)
        y = pd.Series(codes)
        try:
            names = [str(class_names[int(u)]) for u in uniques] if class_names else None
        except (IndexError, ValueError, TypeError):
            names = None
        if names is None:
            names = [f"Kelas {u}" for u in uniques]

    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]
    X.columns = [str(c) for c in X.columns]
    df = X.copy()
    df["TARGET"] = y.to_numpy()
    return df, names


# ------------------------------------------------------------------
# Loader per sumber
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_sklearn(name: str):
    meta = _SKLEARN[name]
    data = meta["loader"]()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    names = [str(n) for n in data.target_names] if meta["kind"] == "classification" else None
    df, names = _finalize(X, data.target, meta["kind"], names)
    return df, names, {"source": "scikit-learn", "fetched_at": None}


@st.cache_data(show_spinner=False)
def _load_uci(name: str):
    from ucimlrepo import fetch_ucirepo

    meta = _UCI[name]
    last_exc: Exception | None = None
    ds = None
    for _ in range(2):
        try:
            ds = fetch_ucirepo(id=meta["id"])
            break
        except Exception as exc:  # jaringan atau server UCI sedang bermasalah
            last_exc = exc
    if ds is None:
        raise RuntimeError(f"Tidak dapat mengunduh dataset dari UCI ({last_exc}).")

    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].copy()
    if meta.get("binarize"):
        y = (pd.to_numeric(y, errors="coerce") > 0).astype(int)
    df, names = _finalize(X, y, meta["kind"], meta.get("names"))
    return df, names, {"source": f"UCI ML Repository (id {meta['id']})", "fetched_at": None}


def _now_label() -> str:
    wib = dt.timezone(dt.timedelta(hours=7))
    return dt.datetime.now(wib).strftime("%d %b %Y %H:%M WIB")


@st.cache_data(ttl=300, show_spinner=False)
def _load_weather(city: str):
    lat, lon = CITIES[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,"
        "cloud_cover,wind_speed_10m,precipitation",
        "past_days": 30,
        "forecast_days": 1,
        "timezone": "Asia/Jakarta",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    resp.raise_for_status()
    raw = pd.DataFrame(resp.json()["hourly"])
    raw["time"] = pd.to_datetime(raw["time"])
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    raw = raw[raw["time"] <= now].dropna().reset_index(drop=True)  # buang jam prakiraan
    hour = raw["time"].dt.hour
    X = raw.drop(columns=["time", "temperature_2m"]).copy()
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df, _ = _finalize(X, raw["temperature_2m"], "regression")
    return df, None, {"source": f"Open-Meteo · {city}", "fetched_at": _now_label()}


@st.cache_data(ttl=300, show_spinner=False)
def _load_quake(kind: str):
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = []
    for feat in resp.json()["features"]:
        p = feat["properties"]
        lon, lat, depth = feat["geometry"]["coordinates"][:3]
        rows.append(
            {
                "latitude": lat,
                "longitude": lon,
                "depth_km": depth,
                "nst": p.get("nst"),
                "gap": p.get("gap"),
                "dmin": p.get("dmin"),
                "rms": p.get("rms"),
                "mag": p.get("mag"),
            }
        )
    raw = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["mag"]).reset_index(drop=True)
    X = raw.drop(columns=["mag"])
    if kind == "regression":
        df, names = _finalize(X, raw["mag"], "regression")
    else:
        y = np.digitize(raw["mag"].to_numpy(), [1, 2.5, 4.5])
        df, names = _finalize(X, y, "classification", QUAKE_CLASSES)
    return df, names, {"source": "USGS Earthquake Feed (30 hari)", "fetched_at": _now_label()}


def load_dataset(name: str, city: str = "Salatiga"):
    """Titik masuk tunggal untuk memuat dataset mana pun."""
    meta = DATASETS[name]
    loader = meta["loader"]
    if loader == "sklearn":
        return _load_sklearn(name)
    if loader == "uci":
        return _load_uci(name)
    if loader == "weather":
        return _load_weather(city)
    return _load_quake(meta["kind"])


def refresh_realtime() -> None:
    """Buang cache data real-time agar diunduh ulang."""
    _load_weather.clear()
    _load_quake.clear()
