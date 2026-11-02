"""Live browser drawing with final controls synchronized to the Python session."""
from pathlib import Path
import streamlit.components.v2 as components

ASSETS = Path(__file__).resolve().parent / "ui"
_component = components.component(
    "rayleigh_live_model",
    html=(ASSETS / "live_model.html").read_text(encoding="utf-8"),
    css=(ASSETS / "live_model.css").read_text(encoding="utf-8"),
    js=(ASSETS / "live_model.mjs").read_text(encoding="utf-8"),
)


def live_model(sigma, alpha, epsilon, can_add, on_controls_change, on_add_change):
    return _component(
        key="live_model",
        data={"sigma": sigma, "alpha": alpha, "epsilon": epsilon, "can_add": can_add},
        default={"controls": {"sigma": sigma, "alpha": alpha}},
        on_controls_change=on_controls_change,
        on_add_change=on_add_change,
    )
