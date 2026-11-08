"""Browser-side rendering of a fixed partition and its threshold."""
from pathlib import Path

import streamlit.components.v2 as components

from fuzzy_core import RayleighFuzzy
from lab_support import cluster_color, cluster_name

ASSETS = Path(__file__).resolve().parent / "ui"
_component = components.component(
    "rayleigh_live_clusters",
    html=(ASSETS / "live_clusters.html").read_text(encoding="utf-8"),
    css=(ASSETS / "live_clusters.css").read_text(encoding="utf-8"),
    js=(ASSETS / "live_clusters.mjs").read_text(encoding="utf-8"),
)


def _method(name, profile):
    return {
        "name": name,
        "labels": profile.labels.tolist(),
        "scores": profile.to_dict()["scores"],
        "groups": [
            {"id": group, "name": cluster_name(group), "color": cluster_color(group),
             "center": profile.centers[group].left[-1] if group else None}
            for group in sorted(set(profile.labels.tolist()))
        ],
    }


def live_clusters(items, config, primary, density, alpha, revision, on_threshold_change):
    methods = [_method("Полная связь", primary)]
    if density is not None:
        methods.append(_method("HDBSCAN", density))
    return _component(
        key="live_clusters",
        data={"alpha": alpha, "revision": revision, "methods": methods,
              "items": [{"id": i.id, "sigma": i.value.sigma} for i in items],
              "tail": RayleighFuzzy(1).alpha_cut(config.epsilon)[1] * 1.06},
        default={"threshold": {"alpha": alpha, "revision": revision}},
        on_threshold_change=on_threshold_change,
    )
