"""Grafik Plotly bertema gelap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RED, GOLD, TEAL, VIOLET = "#E50914", "#F5C518", "#2EC4B6", "#8E7DFF"
MUTED, GRID, SURFACE = "#9A9AB0", "#2A2A38", "#14141C"
COLORWAY = [RED, GOLD, TEAL, VIOLET, "#FF8A5B", "#4CC9F0", "#B5E48C", "#F15BB5", "#C8C8D6"]
DIVERGING = [[0, TEAL], [0.5, SURFACE], [1, RED]]
SEQ_RED = [[0, SURFACE], [1, RED]]
MAX_POINTS = 4000


def style(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E8E8EE", size=13),
        colorway=COLORWAY,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text=title, font=dict(family="Bebas Neue, Inter", size=22), x=0.01)
        if title
        else None,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def _sample(n: int, rs: int = 0) -> np.ndarray:
    if n <= MAX_POINTS:
        return np.arange(n)
    return np.sort(np.random.RandomState(rs).choice(n, MAX_POINTS, replace=False))


def actual_vs_pred(y_true, y_pred) -> go.Figure:
    idx = _sample(len(y_true))
    yt, yp = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=yt, y=yp, mode="markers", name="Prediksi",
            marker=dict(color=RED, size=7, opacity=0.65, line=dict(color="#0B0B10", width=0.5)),
        )
    )
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                   line=dict(color=GOLD, dash="dash", width=2))
    )
    fig.update_xaxes(title="Nilai aktual")
    fig.update_yaxes(title="Nilai prediksi")
    return style(fig, 440, "Aktual vs Prediksi")


def confusion(cm, labels) -> go.Figure:
    fig = px.imshow(
        cm, x=labels, y=labels, text_auto=True, aspect="auto",
        color_continuous_scale=SEQ_RED,
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
    )
    fig.update_layout(coloraxis_showscale=False)
    return style(fig, 440, "Confusion Matrix")


def importance(series: pd.Series, top: int = 15) -> go.Figure:
    s = series.head(top)[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=GOLD))
    fig.update_xaxes(title="Pengaruh relatif")
    return style(fig, max(260, 28 * len(s) + 80), "Fitur paling berpengaruh")


def loss_curve(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"], name="Train loss",
                             line=dict(color=RED, width=3)))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_loss"], name="Validation loss",
                             line=dict(color=GOLD, width=3)))
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss")
    return style(fig, 380, "Kurva pelatihan")


def clusters_scatter(coords, labels, var_ratio) -> go.Figure:
    idx = _sample(len(labels))
    d = pd.DataFrame({"PC1": coords[idx, 0], "PC2": coords[idx, 1], "Cluster": labels[idx]})
    order = sorted(d["Cluster"].unique(), key=lambda s: (s == "Noise", s))
    cmap = {c: COLORWAY[i % len(COLORWAY)] for i, c in enumerate([o for o in order if o != "Noise"])}
    cmap["Noise"] = "#5A5A6E"
    fig = px.scatter(d, x="PC1", y="PC2", color="Cluster", category_orders={"Cluster": order},
                     color_discrete_map=cmap, opacity=0.8)
    fig.update_traces(marker=dict(size=7, line=dict(color="#0B0B10", width=0.5)))
    fig.update_xaxes(title=f"PC1 ({var_ratio[0]:.0%} varians)")
    fig.update_yaxes(title=f"PC2 ({var_ratio[1]:.0%} varians)")
    return style(fig, 460, "Peta cluster (proyeksi PCA)")


def elbow(table: pd.DataFrame, chosen_k: int) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=table["k"], y=table["inertia"], name="Inertia",
                             mode="lines+markers", line=dict(color=RED, width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=table["k"], y=table["silhouette"], name="Silhouette",
                             mode="lines+markers", line=dict(color=GOLD, width=3)), secondary_y=True)
    fig.add_vline(x=chosen_k, line_dash="dot", line_color=TEAL)
    fig.update_xaxes(title="Jumlah cluster (k)", dtick=1)
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette", secondary_y=True, showgrid=False)
    return style(fig, 460, "Elbow & Silhouette")


def sizes_bar(sizes: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=sizes.index, y=sizes.values, marker_color=COLORWAY[: len(sizes)]))
    fig.update_yaxes(title="Jumlah data")
    return style(fig, 460, "Ukuran cluster")


def heat(df: pd.DataFrame, title: str, diverging: bool = True, xlab: str = "", ylab: str = "") -> go.Figure:
    kwargs = dict(color_continuous_midpoint=0) if diverging else {}
    fig = px.imshow(
        df.round(2), x=list(df.columns), y=list(df.index), text_auto=".2f" if diverging else True,
        aspect="auto", color_continuous_scale=DIVERGING if diverging else SEQ_RED,
        labels=dict(x=xlab, y=ylab), **kwargs,
    )
    fig.update_layout(coloraxis_showscale=False)
    return style(fig, max(300, 44 * len(df) + 140), title)


def correlation(df: pd.DataFrame, max_cols: int = 18) -> go.Figure:
    cols = list(df.columns[:max_cols])
    if "TARGET" in df.columns and "TARGET" not in cols:
        cols[-1] = "TARGET"
    corr = df[cols].corr()
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale=DIVERGING, aspect="auto")
    return style(fig, 560, "Korelasi antar fitur")


def sentiment_donut(counts: dict[str, int]) -> go.Figure:
    colors = {"positif": TEAL, "netral": "#8A8AA0", "negatif": RED}
    labels = [k for k in ("positif", "netral", "negatif") if counts.get(k)]
    fig = go.Figure(
        go.Pie(
            labels=[l.capitalize() for l in labels],
            values=[counts[l] for l in labels],
            hole=0.62,
            marker=dict(colors=[colors[l] for l in labels], line=dict(color="#0B0B10", width=3)),
            textinfo="percent",
        )
    )
    fig.update_layout(showlegend=True)
    return style(fig, 260)
