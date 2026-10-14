"""ML Studio: aplikasi Streamlit untuk regresi, klasifikasi, clustering, deep learning,
komentar publik (MongoDB), dan analisis LLM (Groq)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from studio import catalog, charts, llm, models, storage
from studio.compat import STRETCH
from studio.styles import CSS, hero_html, poster_grid_html, section_title

st.set_page_config(page_title="Machine Learning", page_icon="💀", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

WIB = timezone(timedelta(hours=7))
MODES = ["Regresi", "Klasifikasi", "Clustering", "Deep Learning"]
COMMENT_COOLDOWN = 15  # detik antar komentar per sesi

for key, default in {"result": None, "train_error": None, "summary": None,
                     "last_post": 0.0, "insights": {}}.items():
    st.session_state.setdefault(key, default)


# ======================================================================
# Helper tampilan
# ======================================================================
def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{v:,.4f}"


def ago(ts: datetime) -> str:
    secs = (datetime.now(timezone.utc) - ts).total_seconds()
    if secs < 60:
        return "baru saja"
    if secs < 3600:
        return f"{int(secs // 60)} menit lalu"
    if secs < 86400:
        return f"{int(secs // 3600)} jam lalu"
    return ts.astimezone(WIB).strftime("%d %b %Y, %H:%M")


def result_label(res: dict | None) -> str:
    return f"{res['dataset']} · {res['model']}" if res else ""


def describe_result(res: dict) -> str:
    lines = [f"Mode: {res['mode']}", f"Dataset: {res['dataset']}", f"Model: {res['model']}"]
    for k, v in res["metrics"].items():
        lines.append(f"{k}: {fmt(v)}")
    if res["mode"] == "Clustering":
        lines.append(f"Jumlah data: {res['n_rows']}, fitur: {res['n_features']}")
        lines.append("Ukuran cluster: " + ", ".join(f"{k}={v}" for k, v in res["sizes"].items()))
    else:
        lines.append(f"Data latih: {res['n_train']}, data uji: {res['n_test']}")
    if res.get("history") is not None:
        h = res["history"]
        lines.append(f"Epoch: {len(h)}, train loss akhir: {h['train_loss'].iloc[-1]:.4f}, "
                     f"val loss akhir: {h['val_loss'].iloc[-1]:.4f}, parameter: {res['n_params']}")
    return "\n".join(lines)


# ======================================================================
# Sidebar
# ======================================================================
with st.sidebar:
    st.markdown('<div class="brand">Machine<span>Learning</span></div>'
                '<div class="brand-sub">672023230 </div>',
                unsafe_allow_html=True)
    mode = st.radio("Mode", MODES)
    dataset_name = st.selectbox(
        "Dataset", catalog.datasets_for(mode),
        format_func=lambda n: f"{n}  ·  {catalog.DATASETS[n]['source']}",
    )
    meta = catalog.DATASETS[dataset_name]
    st.caption(meta["desc"])

    city = "Salatiga"
    if meta.get("needs_city"):
        city = st.selectbox("Lokasi cuaca", list(catalog.CITIES))
    if meta["source"] == "Real-time":
        st.markdown('<span class="live"><i></i>LIVE DATA</span>', unsafe_allow_html=True)
        if st.button("Perbarui data real-time"):
            catalog.refresh_realtime()

    st.divider()
    params: dict = {}
    if mode == "Regresi":
        params["model"] = st.selectbox("Algoritma", models.REG_MODELS)
    elif mode == "Klasifikasi":
        params["model"] = st.selectbox("Algoritma", models.CLF_MODELS)
    elif mode == "Clustering":
        params["algo"] = st.selectbox("Algoritma", models.CLUSTER_ALGOS)
        if params["algo"] == "DBSCAN":
            params["eps"] = st.slider("Epsilon (eps)", 0.1, 5.0, 1.0, 0.1)
            params["min_samples"] = st.slider("Min samples", 2, 30, 5)
        else:
            params["k"] = st.slider("Jumlah cluster (k)", 2, 10, 3)
    else:
        params["layers"] = st.slider("Hidden layer", 1, 4, 2)
        params["neurons"] = st.select_slider("Neuron per layer", [16, 32, 64, 128], value=64)
        params["dropout"] = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
        params["epochs"] = st.slider("Epoch", 10, 200, 60, 10)
        params["lr"] = st.select_slider("Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01], value=0.001)
        params["batch"] = st.select_slider("Batch size", [32, 64, 128, 256], value=64)

    test_size = 0.2
    if mode != "Clustering":
        test_size = st.slider("Ukuran data uji", 0.1, 0.5, 0.2, 0.05, format="%.2f")
    random_state = int(st.number_input("Random state", value=42, step=1))

    st.divider()
    btn_run = st.button("Jalankan clustering" if mode == "Clustering" else "Latih model",
                        **STRETCH)

# ======================================================================
# Muat dataset
# ======================================================================
df = target_names = info = None
data_error = None
try:
    with st.spinner("Memuat dataset…"):
        df, target_names, info = catalog.load_dataset(dataset_name, city)
except Exception as exc:  # jaringan, server UCI/API tidak tersedia, dsb.
    data_error = f"{type(exc).__name__}: {exc}"

# ======================================================================
# Training
# ======================================================================
if btn_run and df is not None:
    st.session_state.train_error = None
    kind = meta["kind"]
    try:
        if mode in ("Regresi", "Klasifikasi"):
            with st.spinner("Melatih model…"):
                res = models.train_supervised(df, kind, target_names, params["model"],
                                              dataset_name, test_size, random_state)
        elif mode == "Clustering":
            with st.spinner("Menjalankan clustering…"):
                res = models.run_clustering(df, kind, target_names, params["algo"],
                                            params.get("k", 3), params.get("eps", 1.0),
                                            params.get("min_samples", 5), dataset_name, random_state)
        else:
            bar = st.progress(0.0, text="Melatih jaringan saraf…")

            def on_epoch(frac: float, row: dict) -> None:
                bar.progress(frac, text=f"Epoch {int(row['epoch'])} · val loss {row['val_loss']:.4f}")

            res = models.train_deep_learning(df, kind, target_names, params, dataset_name,
                                             test_size, random_state, on_epoch)
            bar.empty()
        st.session_state.result = res
    except Exception as exc:
        st.session_state.train_error = f"{type(exc).__name__}: {exc}"


# ======================================================================
# Komponen halaman
# ======================================================================
def render_result(res: dict) -> None:
    live = '<span class="live"><i></i>LIVE</span>' if catalog.DATASETS[res["dataset"]]["source"] == "Real-time" else ""
    st.markdown(
        f'<div class="now-playing"><span class="np-label">Now playing · {escape(res["mode"])}</span>'
        f'<span class="np-title">{escape(res["dataset"])} — {escape(res["model"])}</span>{live}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(res["metrics"]))
    for col, (name, val) in zip(cols, res["metrics"].items()):
        col.metric(name, fmt(val))

    tab_vis, tab_eval, tab_ai = st.tabs(["Visualisasi", "Evaluasi", "Insight AI"])
    mode = res["mode"]

    # ---------------- Visualisasi ----------------
    with tab_vis:
        if mode == "Clustering":
            if res["sampled"]:
                st.caption(f"Agglomerative memakai sampel acak {res['n_rows']:,} baris agar hemat memori.")
            left, right = st.columns([3, 2])
            left.plotly_chart(charts.clusters_scatter(res["coords"], res["labels_arr"], res["var_ratio"]),
                              **STRETCH)
            if res["elbow"] is not None:
                right.plotly_chart(charts.elbow(res["elbow"], res["k"]), **STRETCH)
            else:
                right.plotly_chart(charts.sizes_bar(res["sizes"]), **STRETCH)
            st.plotly_chart(charts.heat(res["profile_z"], "Profil cluster (z-score rata-rata fitur)",
                                        xlab="Fitur", ylab="Cluster"), **STRETCH)
        else:
            if res.get("history") is not None:
                st.plotly_chart(charts.loss_curve(res["history"]), **STRETCH)
                st.caption(f"{res['model']} · {res['n_params']:,} parameter · "
                           f"{res['seconds']:.1f} detik pelatihan")
            if res["kind"] == "regression":
                left, right = st.columns([2, 1])
                left.plotly_chart(charts.actual_vs_pred(res["y_test"], res["y_pred"]), **STRETCH)
                right.markdown("**Contoh prediksi**")
                right.dataframe(pd.DataFrame({"Aktual": res["y_test"], "Prediksi": res["y_pred"]}).head(15).round(3),
                                **STRETCH, hide_index=True)
            else:
                left, right = st.columns(2)
                left.plotly_chart(charts.confusion(res["cm"], res["labels"]), **STRETCH)
                if res.get("importances") is not None:
                    right.plotly_chart(charts.importance(res["importances"]), **STRETCH)
                else:
                    right.info("Model ini tidak menyediakan skor pentingnya fitur.")
            if res["kind"] == "regression" and res.get("importances") is not None:
                st.plotly_chart(charts.importance(res["importances"]), **STRETCH)

    # ---------------- Evaluasi ----------------
    with tab_eval:
        if mode == "Clustering":
            a, b = st.columns([1, 2])
            a.markdown("**Ukuran cluster**")
            a.dataframe(res["sizes"].rename("Jumlah"), **STRETCH)
            b.markdown("**Rata-rata fitur per cluster (satuan asli)**")
            b.dataframe(res["profile"].round(3), **STRETCH)
            if res["crosstab"] is not None:
                st.plotly_chart(charts.heat(res["crosstab"], "Cluster vs label asli", diverging=False,
                                            xlab="Label asli", ylab="Cluster"), **STRETCH)
            st.caption("Silhouette mendekati 1 berarti cluster padat dan terpisah baik; "
                       "Davies-Bouldin makin kecil makin baik; ARI 1 berarti sama persis dengan label asli.")
        else:
            st.caption(f"{res['n_train']:,} data latih · {res['n_test']:,} data uji · "
                       f"{res['seconds']:.2f} detik")
            if res["kind"] == "classification":
                st.markdown("**Laporan klasifikasi lengkap**")
                st.code(res["report"], language="text")
            else:
                resid = pd.Series(res["y_test"] - res["y_pred"], name="Residual")
                st.markdown("**Ringkasan residual (aktual − prediksi)**")
                st.dataframe(resid.describe().round(4).to_frame().T, **STRETCH, hide_index=True)
            if res.get("history") is not None:
                with st.expander("Riwayat per epoch"):
                    st.dataframe(res["history"].round(5), **STRETCH, hide_index=True)

    # ---------------- Insight AI ----------------
    with tab_ai:
        key = f"{res['dataset']}|{res['model']}|{res['trained_at']}"
        if not llm.is_configured():
            st.info("Tambahkan `GROQ_API_KEY` di `.streamlit/secrets.toml` untuk mengaktifkan penjelasan "
                    "otomatis oleh LLM Groq.")
        else:
            if st.button("Jelaskan hasil dengan Groq", key="btn_insight"):
                try:
                    with st.spinner("Groq sedang menganalisis…"):
                        st.session_state.insights[key] = llm.explain_results(describe_result(res))
                except Exception as exc:
                    st.error(f"Groq gagal merespons ({type(exc).__name__}).")
            if key in st.session_state.insights:
                st.markdown(st.session_state.insights[key])


def render_empty() -> None:
    st.markdown(section_title("Mulai dari sini"), unsafe_allow_html=True)
    st.markdown(poster_grid_html(), unsafe_allow_html=True)
    with st.expander("Cara menggunakan", expanded=True):
        st.markdown(
            "1. Pilih **mode** dan **dataset** di panel kiri, termasuk data UCI dan data real-time.\n"
            "2. Atur algoritma dan parameter, lalu klik tombol merah.\n"
            "3. Lihat visualisasi dan evaluasi, minta penjelasan dari Groq.\n"
            "4. Tulis komentar di tab **Komunitas**. Komentar tersimpan di MongoDB dan dianalisis sentimennya."
        )


def render_data_tab() -> None:
    if data_error:
        st.error(f"Dataset tidak dapat dimuat. {data_error}")
        st.caption("Data UCI dan real-time membutuhkan koneksi internet.")
        return
    n_feat = df.shape[1] - 1
    st.markdown(section_title(dataset_name), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", fmt(len(df)))
    c2.metric("Fitur", fmt(n_feat))
    c3.metric("Jenis", "Regresi" if meta["kind"] == "regression" else f"{len(target_names)} kelas")
    c4.metric("Diperbarui", info["fetched_at"] or "Statis")
    st.caption(f"Sumber: {info['source']}. {meta['desc']}")
    st.markdown("**Pratinjau data**")
    st.dataframe(df.head(100), **STRETCH)
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.correlation(df), **STRETCH)
    right.markdown("**Statistik deskriptif**")
    right.dataframe(df.describe().T.round(3), **STRETCH, height=520)
    st.download_button("Unduh CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{dataset_name.replace(' ', '_')}.csv", mime="text/csv")


def comment_html(c: dict) -> str:
    s = c.get("sentiment") or {}
    label = s.get("label")
    badge = ""
    if label:
        cls = {"positif": "pos", "netral": "neu", "negatif": "neg"}[label]
        badge = f'<span class="badge {cls}">{escape(label)}</span>'
    name = escape(c["name"])
    body = escape(c["text"]).replace("\n", "<br>")
    ctx = f" · {escape(c['context'])}" if c.get("context") else ""
    return (f'<div class="comment"><div class="c-head"><div class="avatar">{escape(c["name"][:1].upper() or "A")}</div>'
            f'<div><div class="c-name">{name}</div><div class="c-meta">{ago(c["created_at"])}{ctx}</div></div>{badge}</div>'
            f'<div class="c-body">{body}</div></div>')


def render_community() -> None:
    kind, _, detail = storage.backend()
    groq_ok = llm.is_configured()

    st.markdown(section_title("Komentar publik"), unsafe_allow_html=True)
    with st.form("comment_form", clear_on_submit=True):
        name = st.text_input("Nama", max_chars=40, placeholder="Nama atau inisial")
        text = st.text_area("Komentar", max_chars=500, height=110,
                            placeholder="Bagikan pendapatmu tentang hasil model…")
        sent = st.form_submit_button("Kirim komentar")
    if sent:
        wait = COMMENT_COOLDOWN - (time.time() - st.session_state.last_post)
        if len(text.strip()) < 3:
            st.warning("Komentar terlalu pendek.")
        elif wait > 0:
            st.warning(f"Tunggu {int(wait) + 1} detik sebelum mengirim lagi.")
        else:
            sentiment = None
            if groq_ok:
                try:
                    with st.spinner("Menganalisis sentimen dengan Groq…"):
                        sentiment = llm.analyze_sentiment(text.strip())
                except Exception as exc:
                    st.warning(f"Sentimen tidak dapat dianalisis ({type(exc).__name__}). Komentar tetap disimpan.")
            try:
                storage.add_comment(name.strip() or "Anonim", text.strip(),
                                    result_label(st.session_state.result), sentiment)
                st.session_state.last_post = time.time()
                st.toast("Komentar terkirim.")
            except Exception as exc:
                st.error(f"Komentar gagal disimpan ({type(exc).__name__}).")

    comments = storage.list_comments(50)
    feed, panel = st.columns([3, 2], gap="large")

    with feed:
        if not comments:
            st.info("Belum ada komentar. Jadilah yang pertama.")
        for c in comments[:30]:
            st.markdown(comment_html(c), unsafe_allow_html=True)

    with panel:
        st.markdown(section_title("Analisis AI"), unsafe_allow_html=True)
        dot = {"mongodb": "", "local": "warn"}[kind]
        st.markdown(f'<div class="status"><span class="dot {dot}"></span>{escape(detail)}</div>'
                    f'<div class="status"><span class="dot {"" if groq_ok else "off"}"></span>'
                    f'{"Groq aktif" if groq_ok else "Groq belum dikonfigurasi (GROQ_API_KEY)"}</div>',
                    unsafe_allow_html=True)
        rated = [c for c in comments if c.get("sentiment")]
        if rated:
            counts = {l: sum(1 for c in rated if c["sentiment"]["label"] == l) for l in llm.LABELS}
            st.plotly_chart(charts.sentiment_donut(counts), **STRETCH)
            avg = float(np.mean([c["sentiment"].get("score", 0) for c in rated]))
            st.metric("Skor sentimen rata-rata", f"{avg:+.2f}")
        elif groq_ok:
            st.caption("Belum ada komentar yang dinilai sentimennya.")

        if groq_ok:
            pending = [c for c in comments if not c.get("sentiment")]
            if pending and st.button(f"Nilai {min(len(pending), 10)} komentar tanpa sentimen"):
                done = 0
                with st.spinner("Menganalisis…"):
                    for c in pending[:10]:
                        try:
                            storage.update_sentiment(c["id"], llm.analyze_sentiment(c["text"]))
                            done += 1
                        except Exception:
                            break
                st.toast(f"{done} komentar dinilai.")
                st.rerun()
            if comments and st.button("Ringkas semua komentar"):
                try:
                    with st.spinner("Groq sedang merangkum…"):
                        st.session_state.summary = llm.summarize_comments(comments)
                except Exception as exc:
                    st.error(f"Ringkasan gagal ({type(exc).__name__}).")
            if st.session_state.summary:
                st.markdown("**Auto-summary**")
                st.markdown(st.session_state.summary)
        else:
            st.info("Isi `GROQ_API_KEY` di `.streamlit/secrets.toml` untuk analisis sentimen "
                    "dan ringkasan otomatis.")


# ======================================================================
# Layout utama
# ======================================================================
st.markdown(hero_html(len(catalog.DATASETS), catalog.count_realtime()), unsafe_allow_html=True)

tab_studio, tab_data, tab_comm = st.tabs(["Studio", "Eksplorasi data", "Komunitas"])

with tab_studio:
    if data_error:
        st.error(f"Dataset tidak dapat dimuat. {data_error}")
    if st.session_state.train_error:
        st.error(f"Pelatihan gagal. {st.session_state.train_error}")
    if st.session_state.result is None:
        render_empty()
    else:
        render_result(st.session_state.result)

with tab_data:
    render_data_tab()

with tab_comm:
    render_community()
