"""Plotly charts for the local application."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
from fuzzy_core import RayleighFuzzy

TEAL = "#087F8C"
BLUE = "#4169B1"
ORANGE = "#D47B35"
PALETTE = [TEAL, BLUE, ORANGE, "#9273B9", "#BE5870", "#658745", "#5A899B", "#A7804F"]
NOISE_COLOR = "#8995A3"


def cluster_name(label):
    return "Шум" if label == 0 else f"Кластер {label}"


def cluster_color(label):
    return NOISE_COLOR if label == 0 else PALETTE[(label - 1) % len(PALETTE)]


def base_figure(height=440, x_title="Значение x", y_title="Принадлежность μ(x)"):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height,
                      font=dict(family="Arial, sans-serif", size=12, color="#202124"),
                      margin=dict(l=12, r=16, t=28, b=30),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hovermode="closest", legend=dict(orientation="h", y=1.12, x=0),
                      xaxis_title=x_title, yaxis_title=y_title)
    fig.update_xaxes(gridcolor="#EBECED", zerolinecolor="#BEC2C6", automargin=True)
    fig.update_yaxes(gridcolor="#EBECED", zerolinecolor="#BEC2C6", automargin=True)
    return fig


def curve_trace(value, config, name, color, fill=False):
    cuts = value.to_alpha_cuts(config)
    if isinstance(value, RayleighFuzzy):
        x = np.linspace(0, cuts.right[0] * 1.06, 700)
        y = value.membership(x)
    else:
        x = np.r_[cuts.left[0], cuts.left, cuts.right[::-1], cuts.right[0]]
        y = np.r_[0, cuts.alpha, cuts.alpha[::-1], 0]
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=3),
                      fill="tozeroy" if fill else None,
                      fillcolor="rgba(8,127,140,0.08)" if fill else None,
                      hovertemplate="x = %{x:.5g}<br>μ = %{y:.4f}<extra>" + name + "</extra>")


def model_figure(value, config, alpha):
    fig = base_figure(height=460)
    fig.add_trace(curve_trace(value, config, "Нормированная кривая", TEAL, True))
    lo, hi = value.alpha_cut(alpha)
    fig.add_hline(y=alpha, line_dash="dot", line_color="#A0ADB9")
    fig.add_trace(go.Scatter(x=[lo, hi], y=[alpha, alpha], mode="lines+markers",
                             name=f"α-срез · {alpha:.3f}",
                             line=dict(color=ORANGE, width=5), marker=dict(size=9),
                             hovertemplate="Граница x = %{x:.6g}<extra>α-срез</extra>"))
    fig.add_vrect(x0=lo, x1=hi, fillcolor=ORANGE, opacity=0.055, line_width=0)
    fig.add_annotation(x=value.sigma, y=1, text="вершина · x = σ", showarrow=True,
                       arrowcolor=TEAL, ay=-30, ax=60, font=dict(size=12, color=TEAL))
    fig.update_yaxes(range=[-0.025, 1.16])
    return fig


def operation_figure(a, b, result, config, alpha, full_tails=False):
    fig = base_figure(height=450)
    for value, name, color in [(a, "A", BLUE), (b, "B", ORANGE)]:
        fig.add_trace(curve_trace(value, config, name, color))
    if result is not None:
        fig.add_trace(curve_trace(result, config, "Результат", TEAL, True))
        lo, hi = result.alpha_cut(alpha)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[alpha, alpha], mode="lines+markers",
                                 line=dict(color=TEAL, width=5), name="Срез результата"))
    if not full_tails:
        values = [a, b] + ([] if result is None else [result])
        bounds = [v.alpha_cut(alpha) for v in values]
        lo, hi = min(v[0] for v in bounds), max(v[1] for v in bounds)
        span = max(hi - lo, abs(hi) * .15, 1e-6)
        fig.update_xaxes(range=[lo - span * .18, hi + span * .18])
    fig.update_yaxes(range=[-0.025, 1.15])
    return fig


def cluster_strip(items, labels):
    fig = base_figure(height=300, x_title="Параметр σ · положение вершины", y_title="Кластер")
    groups = sorted(set(labels))
    # y is solely a categorical cluster lane, never an invented second feature.
    for lab in groups:
        idx = [j for j, l in enumerate(labels) if l == lab]
        fig.add_trace(go.Scatter(x=[items[j].value.sigma for j in idx], y=[str(lab)] * len(idx),
                                 text=[items[j].id for j in idx], mode="markers",
                                 name=cluster_name(lab), marker=dict(size=9, opacity=0.65,
                                 color=cluster_color(lab)),
                                 hovertemplate="%{text}<br>σ = %{x:.5f}<extra></extra>"))
    fig.update_layout(showlegend=len(groups) <= 10)
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=[str(g) for g in groups])
    return fig


def generation_figure(items, generation, cluster_labels=None):
    """Aligned source/prediction lanes; y is a category, not another feature."""
    source_labels = [generation["origin_labels"][item.id] for item in items]
    comparison = cluster_labels is not None
    fig = make_subplots(rows=2 if comparison else 1, cols=1, shared_xaxes=comparison,
                        vertical_spacing=.22 if comparison else 0,
                        subplot_titles=["Как создано: источники генерации", "Как найдено: кластеры"] if comparison
                        else ["Источники генерации · алгоритм этих меток не видит"])
    for row, labels in enumerate([source_labels] + ([cluster_labels] if comparison else []), 1):
        for lab in sorted(set(int(v) for v in labels)):
            idx = [j for j, label in enumerate(labels) if label == lab]
            color = "#9BA5AF" if lab == 0 else PALETTE[(lab-1) % len(PALETTE)]
            name = ("Выбросы" if lab == 0 else f"Источник {lab}") if row == 1 else cluster_name(lab)
            fig.add_trace(go.Scatter(x=[items[j].value.sigma for j in idx], y=[name]*len(idx),
                                    text=[items[j].id for j in idx], mode="markers", name=name,
                                    showlegend=False, marker=dict(size=8, color=color, opacity=.65),
                                    hovertemplate="%{text}<br>σ = %{x:.5f}<extra>"+name+"</extra>"), row=row, col=1)
        fig.update_yaxes(type="category", categoryorder="trace", row=row, col=1)
    fig.update_layout(template="plotly_white", height=490 if comparison else 280,
                      font=dict(family="Arial, sans-serif", color="#202124", size=12),
                      margin=dict(l=20, r=20, t=36, b=28), paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Параметр σ", row=2 if comparison else 1, col=1)
    return fig


def comparison_figure(items, complete, density, generation=None):
    """Both partitions on a shared sigma axis, optionally with source labels."""
    lanes = [("Полная связь", complete.labels), ("HDBSCAN · серый цвет означает шум", density.labels)]
    if generation:
        lanes.insert(0, ("Источники генерации · скрыты от алгоритмов",
                         [generation["origin_labels"][i.id] for i in items]))
    fig = make_subplots(rows=len(lanes), cols=1, shared_xaxes=True,
                        vertical_spacing=.13, subplot_titles=[name for name, _ in lanes])
    low, high = min(i.value.sigma for i in items), max(i.value.sigma for i in items)
    pad = max((high - low) * .04, abs(low) * .04, .01)
    for row, (_, labels) in enumerate(lanes, 1):
        is_source = bool(generation) and row == 1
        for lab in sorted(set(int(v) for v in labels)):
            idx = [j for j, value in enumerate(labels) if value == lab]
            name = ("Выбросы" if lab == 0 else f"Источник {lab}") if is_source else cluster_name(lab)
            fig.add_trace(go.Scatter(x=[items[j].value.sigma for j in idx], y=[name] * len(idx),
                mode="markers", text=[items[j].id for j in idx], name=name, showlegend=False,
                marker=dict(size=8, color=cluster_color(lab), opacity=.72,
                            symbol="x" if lab == 0 else "circle"),
                hovertemplate="%{text}<br>σ = %{x:.6g}<extra>" + name + "</extra>"), row=row, col=1)
        fig.update_yaxes(type="category", categoryorder="trace", row=row, col=1)
        fig.update_xaxes(range=[low-pad, high+pad], gridcolor="#E9EDF1", row=row, col=1)
    fig.update_xaxes(title_text="Параметр σ · одни и те же величины", row=len(lanes), col=1)
    fig.update_layout(template="plotly_white", height=235 * len(lanes),
                      font=dict(family="Arial, sans-serif", size=12, color="#202124"),
                      margin=dict(l=12, r=16, t=40, b=35), paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_annotations(font_size=13)
    return fig


def tree_figure(result):
    fig = base_figure(height=380, x_title="Листья дерева · объекты или объединённые ветви",
                      y_title="Расстояние объединения")
    if len(result.linkage) == 0:
        fig.add_annotation(text="В наборе одна величина", showarrow=False)
        return fig
    tree = dendrogram(result.linkage, no_plot=True, truncate_mode="lastp", p=25)
    # The tree is intentionally neutral: categorical cluster colors live on the strip.
    for xs, ys in zip(tree["icoord"], tree["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=TEAL, width=1.7),
                                 showlegend=False, hoverinfo="y"))
    fig.update_xaxes(tickvals=list(range(5, 10 * len(tree["ivl"]), 10)), ticktext=tree["ivl"], tickfont_size=10)
    if result.mode == "threshold":
        fig.add_hline(y=result.threshold, line_dash="dash", line_color=ORANGE,
                      annotation_text=f"T = {result.threshold:g}")
    return fig
