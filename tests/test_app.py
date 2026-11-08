from pathlib import Path
from streamlit.testing.v1 import AppTest
from fuzzy_core import SessionData, generate, ComputationConfig

APP = Path(__file__).resolve().parents[1] / "app.py"


def app():
    return AppTest.from_file(str(APP), default_timeout=30).run()


def test_four_working_pages_without_theory_or_journal():
    at = app()
    assert not at.exception
    assert at.radio(key="section").options == ["Модель", "Операции", "Набор", "Кластеры"]
    assert "audit" not in at.session_state
    at.run()
    for section in ["Операции", "Набор величин", "Кластеры", "Модель"]:
        at.radio(key="section").set_value(section).run()
        assert not at.exception, section


def test_model_add_ignores_cleared_trigger_and_replayed_request():
    # Exercise the actual callback with the component state before rendering,
    # including the transient reset Streamlit sends after a browser event.
    prefix = ('import streamlit as st\n'
              'if "test_payload" in st.session_state:\n'
              '    st.session_state["live_model"] = {"add": st.session_state.test_payload}\n')
    script = prefix + APP.read_text() + '\nmodel_add_requested()\nmodel_add_requested()\n'
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception and len(at.session_state["items"]) == 150
    at.session_state.test_payload = {"sigma": 3.0, "alpha": 0.7, "request_id": "one-click"}
    at.run()
    assert not at.exception and len(at.session_state["items"]) == 151
    assert at.session_state["items"][-1].value.sigma == 3.0
    at.run()
    assert not at.exception and len(at.session_state["items"]) == 151
    at.session_state.test_payload = None
    at.run()
    assert not at.exception and len(at.session_state["items"]) == 151


def test_operation_saved_once_and_can_be_reused():
    at = app()
    at.radio(key="section").set_value("Операции").run()
    at.button(key="save_operation").click().run()
    assert not at.exception and len(at.session_state.results) == 1
    at.run()
    assert len(at.session_state.results) == 1
    at.selectbox(key="source_a").set_value("C001").run()
    assert not at.exception
    at.selectbox(key="operation").set_value("Вычитание").run()
    assert not at.exception


def test_cluster_count_and_threshold_ui():
    at = app()
    at.radio(key="section").set_value("Кластеры").run()
    at.selectbox(key="cluster_mode").set_value("Ровно K кластеров").run()
    at.button(key="run_clustering").click().run()
    assert not at.exception
    assert len(set(at.session_state.cluster.labels)) == 3
    at.selectbox(key="cluster_mode").set_value("По порогу T").run()
    at.number_input(key="threshold").set_value(10.0).run()
    at.button(key="run_clustering").click().run()
    assert not at.exception and len(set(at.session_state.cluster.labels)) == 1


def test_auto_ui_needs_no_cluster_parameter_and_restores_settings():
    from fuzzy_clustering import cluster_auto
    at = app()
    at.radio(key="section").set_value("Кластеры").run()
    assert at.selectbox(key="cluster_mode").value == "Авто"
    assert not any(w.key in ("k", "threshold") for w in at.number_input)
    at.button(key="run_clustering").click().run()
    assert not at.exception and at.session_state.cluster.k == 3
    assert at.session_state.cluster.mode == "auto"
    assert at.session_state.density_cluster.k == 3
    items = generate(100, seed=42, mode="uniform")
    result = cluster_auto(items)
    at.session_state.pending_session = SessionData(items, clustering=result.to_dict())
    at.run()
    assert not at.exception and at.session_state.cluster.k == 1
    assert at.selectbox(key="cluster_mode").value == "Авто"


def test_new_dataset_invalidates_clusters():
    at = app()
    at.radio(key="section").set_value("Кластеры").run()
    at.button(key="run_clustering").click().run()
    at.radio(key="section").set_value("Набор величин").run()
    at.number_input(key="samples").set_value(100)
    at.button(key="generate_dataset").click().run()
    assert not at.exception and len(at.session_state["items"]) == 100
    assert at.session_state.cluster is None
    assert at.session_state.density_cluster is None


def test_new_scenarios_and_provenance_survive_navigation():
    at = app()
    at.radio(key="section").set_value("Набор величин").run()
    at.selectbox(key="generator_scenario").set_value("unequal").run()
    at.button(key="generate_dataset").click().run()
    assert not at.exception
    assert at.session_state.generation["scenario"] == "unequal"
    at.radio(key="section").set_value("Кластеры").run()
    at.button(key="run_clustering").click().run()
    assert not at.exception and at.session_state.cluster.k == 4
    assert any("Проверить по исходным группам" == e.label for e in at.expander)
    at.radio(key="section").set_value("Набор величин").run()
    assert at.selectbox(key="generator_scenario").value == "unequal"
    at.selectbox(key="generator_scenario").set_value("custom").run()
    at.number_input(key="gen_groups").set_value(6)
    at.slider(key="gen_outliers").set_value(10)
    at.button(key="generate_dataset").click().run()
    assert not at.exception
    assert len(at.session_state.generation["components"]) == 6
    assert at.session_state.generation["outliers"] == 15


def test_session_restore_before_widget_creation():
    at = app()
    at.session_state.pending_session = SessionData(generate(100), config=ComputationConfig(),
                                                  controls={"model_sigma": 3.0, "k": 2})
    at.run()
    assert not at.exception
    assert len(at.session_state["items"]) == 100
    assert at.session_state.model_sigma == 3.0


def test_zero_division_error_and_recovery():
    at = app()
    at.radio(key="section").set_value("Операции").run()
    at.number_input(key="op_b_sigma").set_value(1.0).run()
    at.selectbox(key="operation").set_value("Вычитание").run()
    at.button(key="save_operation").click().run()
    at.selectbox(key="source_b").set_value("C001").run()
    at.selectbox(key="operation").set_value("Деление").run()
    assert not at.exception and at.error
    at.run()
    assert not at.exception and at.error
    at.selectbox(key="operation").set_value("Сложение").run()
    at.selectbox(key="operation").set_value("Деление").run()
    assert not at.exception and at.error


def test_restore_minimal_single_item_session():
    from fuzzy_core import FuzzyItem, RayleighFuzzy
    at = app()
    at.session_state.pending_session = SessionData([FuzzyItem("R001", RayleighFuzzy(2))])
    at.run()
    assert not at.exception
    assert len(at.session_state["items"]) == 1 and at.session_state.k == 1


def test_hdbscan_comparison_ui_controls_noise_and_restore():
    from fuzzy_core import generate_scenario
    from fuzzy_clustering import cluster_auto, cluster_hdbscan
    at = app()
    batch = generate_scenario(scenario="outliers")
    at.session_state.pending_session = SessionData(batch.items, generation=batch.provenance)
    at.radio(key="section").set_value("Кластеры").run()
    at.button(key="run_clustering").click().run()
    assert not at.exception and at.session_state.cluster.k == 6
    assert at.session_state.density_cluster.k == 3
    assert sum(at.session_state.density_cluster.labels == 0) == 8
    assert at.session_state.density_profile.to_dict()["scores"].count(None) == 8
    at.number_input(key="hb_min_cluster_size").set_value(300)
    at.button(key="run_clustering").click().run()
    assert not at.exception and at.session_state.density_cluster.k == 0
    assert any("весь набор шумом" in info.value for info in at.info)
    at.checkbox(key="compare_hdbscan").uncheck().run()
    at.button(key="run_clustering").click().run()
    assert not at.exception and at.session_state.density_cluster is None
    d = cluster_hdbscan(batch.items, min_cluster_size=12, min_samples=7, allow_single_cluster=False)
    at.session_state.pending_session = SessionData(batch.items, clustering=cluster_auto(batch.items).to_dict(),
        comparison=d.to_dict(), generation=batch.provenance)
    at.run()
    assert not at.exception and at.session_state.density_cluster.to_dict() == d.to_dict()
    assert at.number_input(key="hb_min_cluster_size").value == 12
    assert at.number_input(key="hb_min_samples").value == 7
    at.radio(key="section").set_value("Модель").run()
    at.radio(key="section").set_value("Кластеры").run()
    assert not at.exception and at.number_input(key="hb_min_samples").value == 7


def test_utility_layout_keeps_operation_sources_and_pending_settings_between_pages():
    at = app()
    at.radio(key="section").set_value("Операции").run()
    at.button(key="save_operation").click().run()
    at.selectbox(key="source_a").set_value("C001").run()
    at.selectbox(key="operation").set_value("Вычитание").run()
    at.checkbox(key="full_tails").check().run()
    at.radio(key="section").set_value("Кластеры").run()
    at.selectbox(key="cluster_mode").set_value("Ровно K кластеров").run()
    at.number_input(key="k").set_value(4).run()
    at.number_input(key="hb_min_samples").set_value(12).run()
    at.radio(key="section").set_value("Набор величин").run()
    at.number_input(key="seed").set_value(99).run()
    assert at.session_state.generation["seed"] == 42
    at.radio(key="section").set_value("Операции").run()
    assert not at.exception
    assert at.selectbox(key="source_a").value == "C001"
    assert at.selectbox(key="operation").value == "Вычитание"
    assert at.checkbox(key="full_tails").value
    at.radio(key="section").set_value("Кластеры").run()
    assert at.selectbox(key="cluster_mode").value == "Ровно K кластеров"
    assert at.number_input(key="k").value == 4
    assert at.number_input(key="hb_min_samples").value == 12
    at.button(key="run_clustering").click().run()
    assert not at.exception and at.session_state.cluster.k == 4



def test_cluster_filter_import_navigation_and_recalculation():
    from fuzzy_clustering import cluster_auto, cluster_hdbscan, membership_mask

    at = app()
    items = generate()
    primary, density = cluster_auto(items), cluster_hdbscan(items)
    at.session_state.pending_session = SessionData(
        items, clustering=primary.to_dict(), comparison=density.to_dict(),
        controls={"cluster_alpha": .43})
    at.radio(key="section").set_value("Кластеры").run()
    assert not at.exception and at.session_state.cluster_alpha == .43
    for key in ("cluster_profile", "density_profile"):
        assert membership_mask(at.session_state[key], .43).sum() == 113
    at.radio(key="section").set_value("Модель").run()
    assert at.session_state.alpha == .5
    at.radio(key="section").set_value("Кластеры").run()
    assert at.session_state.cluster_alpha == .43
    at.button(key="run_clustering").click().run()
    assert not at.exception and at.session_state.cluster_alpha == 0
