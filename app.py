"""Local educational UI. Run: python -m streamlit run app.py"""
import math
import numpy as np
import pandas as pd
import streamlit as st
from fuzzy_core import (ComputationConfig, RayleighFuzzy, FuzzyItem, arithmetic, compare,
                        generate_scenario, SCENARIOS, SessionData, dumps_session, loads_session, dataset_csv)
from fuzzy_clustering import (cluster_by_count, cluster_by_threshold, cluster_auto, cluster_hdbscan,
                              adjusted_rand_index, cluster_membership)
from live_model_component import live_model
from live_clusters_component import live_clusters
from lab_support import (TEAL, PALETTE,
                         operation_figure, cluster_strip, tree_figure, base_figure, curve_trace, generation_figure,
                         cluster_name)

from ui_style import install_style, workspace, values_table

st.set_page_config(page_title="Рэлей", page_icon="∿", layout="wide", initial_sidebar_state="auto")
install_style()

DEFAULTS = dict(model_sigma=2.0, alpha=0.5, op_a_sigma=1.0, op_b_sigma=2.0,
                samples=150, seed=42, k=3, threshold=1.0, cluster_alpha=0.)
GEN_DEFAULTS = dict(generator_scenario="demo", gen_groups=4, gen_separation=7.0,
                    gen_imbalance=50, gen_outliers=0, gen_sigma_min=.5, gen_sigma_max=6.0)
DENSITY_DEFAULTS = dict(compare_hdbscan=True, hb_min_cluster_size=10, hb_min_samples=10,
                        hb_allow_single=True)


def set_analysis(primary=None, density=None):
    """Install a partition and its fixed profiles together."""
    state = st.session_state
    profile = cluster_membership(state["items"], primary, state.cfg) if primary is not None else None
    density_profile = cluster_membership(state["items"], density, state.cfg) if density is not None else None
    state.update(cluster=primary, density_cluster=density, cluster_profile=profile,
                 density_profile=density_profile, cluster_alpha=0.,
                 cluster_revision=state.get("cluster_revision", 0) + 1)


def cluster_alpha_changed():
    state = st.session_state
    payload = state["live_clusters"].get("threshold")
    if not isinstance(payload, dict) or payload.get("revision") != state.cluster_revision:
        return
    alpha = payload.get("alpha")
    if type(alpha) in (int, float) and 0 <= alpha <= 1:
        state.cluster_alpha = float(alpha)


def apply_model_controls(payload):
    if not isinstance(payload, dict):
        return
    sigma, alpha = float(payload["sigma"]), float(payload["alpha"])
    if not 0.000001 <= sigma <= 1_000_000 or not st.session_state.cfg.epsilon <= alpha <= 1:
        raise ValueError("Недопустимые параметры модели.")
    st.session_state.model_sigma = sigma
    st.session_state.alpha = alpha


def model_controls_changed():
    apply_model_controls(st.session_state["live_model"].get("controls"))


def model_add_requested():
    payload = st.session_state["live_model"].get("add")
    # Streamlit also clears transient component values on later reruns.
    if not isinstance(payload, dict) or not payload.get("request_id"):
        return
    if payload["request_id"] == st.session_state.get("last_model_add_request"):
        return
    apply_model_controls(payload)
    st.session_state.last_model_add_request = payload["request_id"]
    items = st.session_state["items"]
    if len(items) >= 300:
        return
    taken = {i.id for i in items + st.session_state.results}
    j = 1
    while f"R{j:03d}" in taken:
        j += 1
    name = f"R{j:03d}"
    items.append(FuzzyItem(name, RayleighFuzzy(st.session_state.model_sigma, name)))
    set_analysis()
    st.session_state.generation = None
    st.session_state.notice = f"Величина {name} добавлена в набор."


def clustering(items, cfg, mode, k=None, threshold=None, auto=None):
    if mode == "auto":
        settings = {key: auto[key] for key in ("max_clusters", "reference_samples", "seed")} if auto else {}
        return cluster_auto(items, cfg, **settings)
    return (cluster_by_count(items, int(k), cfg) if mode == "count"
            else cluster_by_threshold(items, float(threshold), cfg))


@st.cache_data(show_spinner=False, max_entries=12)
def cached_cluster(sigmas, epsilon, levels, mode, parameter):
    items = [FuzzyItem(f"R{i + 1:03d}", RayleighFuzzy(s)) for i, s in enumerate(sigmas)]
    return clustering(items, ComputationConfig(epsilon, levels), mode, parameter, parameter)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_density(sigmas, epsilon, levels, min_cluster_size, min_samples, allow_single_cluster):
    values = [RayleighFuzzy(s) for s in sigmas]
    return cluster_hdbscan(values, ComputationConfig(epsilon, levels),
        min_cluster_size=min_cluster_size, min_samples=min_samples, allow_single_cluster=allow_single_cluster)


def restore_controls(controls, cfg, n):
    output = DEFAULTS.copy()
    output["k"] = min(output["k"], n)
    output["alpha"] = max(output["alpha"], cfg.epsilon)
    for key in DEFAULTS:
        value = controls.get(key, output[key])
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Неверное значение настройки {key}.")
        output[key] = value
    for key in ("model_sigma", "op_a_sigma", "op_b_sigma"):
        if not 0.000001 <= output[key] <= 1_000_000:
            raise ValueError("Параметры интерфейса σ должны быть от 0,000001 до 1000000.")
        output[key] = float(output[key])
    if not cfg.epsilon <= output["alpha"] <= 1:
        raise ValueError("Выбранный α ниже порога сессии или выше 1.")
    if not 0 <= output["cluster_alpha"] <= 1:
        raise ValueError("Порог соответствия α должен быть числом от 0 до 1.")
    for key, low, high in [("samples", 100, 300), ("seed", 0, 2**32 - 1), ("k", 1, n)]:
        if int(output[key]) != output[key] or not low <= output[key] <= high:
            raise ValueError(f"Недопустимая настройка {key}.")
        output[key] = int(output[key])
    if output["threshold"] < 0:
        raise ValueError("Порог расстояния не может быть отрицательным.")
    output["alpha"] = float(output["alpha"])
    output["threshold"] = float(output["threshold"])
    scenario = controls.get("generator_scenario", "demo")
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise ValueError("Неизвестный сценарий в настройках интерфейса.")
    output["generator_scenario"] = scenario
    for key, low, high, integer in [("gen_groups", 2, 8, True), ("gen_separation", 1, 12, False),
                                    ("gen_imbalance", 0, 100, True), ("gen_outliers", 0, 20, True),
                                    ("gen_sigma_min", .000001, 1_000_000, False),
                                    ("gen_sigma_max", .000001, 1_000_000, False)]:
        value = controls.get(key, GEN_DEFAULTS[key])
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not low <= value <= high or (integer and int(value) != value)):
            raise ValueError(f"Неверная настройка генерации {key}.")
        output[key] = int(value) if integer else float(value)
    for key, default in DENSITY_DEFAULTS.items():
        value = controls.get(key, default)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"Неверная настройка HDBSCAN: {key}.")
        elif type(value) is not int or not (2 if key == "hb_min_cluster_size" else 1) <= value <= 300:
            raise ValueError(f"Неверная настройка HDBSCAN: {key}.")
        output[key] = value
    return output


if "items" not in st.session_state:
    initial = generate_scenario(scenario="demo")
    st.session_state["items"] = initial.items
    st.session_state.generation = initial.provenance
    st.session_state.results = []
    st.session_state.cfg = ComputationConfig()
    set_analysis()
    st.session_state.update({**DEFAULTS, **GEN_DEFAULTS, **DENSITY_DEFAULTS})

if "pending_session" in st.session_state:
    loaded = st.session_state.pop("pending_session")
    incoming_controls = loaded.controls.copy()
    if loaded.comparison:
        for key, param in [("hb_min_cluster_size", "min_cluster_size"), ("hb_min_samples", "min_samples"),
                           ("hb_allow_single", "allow_single_cluster")]:
            incoming_controls.setdefault(key, min(300, loaded.comparison["density"][param])
                                         if param != "allow_single_cluster" else loaded.comparison["density"][param])
    controls = restore_controls(incoming_controls, loaded.config, len(loaded.items))
    result = None
    if loaded.clustering:
        c = loaded.clustering
        result = clustering(loaded.items, loaded.config, c["mode"], c.get("k"), c.get("threshold"), c.get("auto"))
    st.session_state["items"] = loaded.items
    st.session_state.results = loaded.results
    st.session_state.cfg = loaded.config
    st.session_state.generation = loaded.generation
    d = loaded.comparison
    density = (cluster_hdbscan(loaded.items, loaded.config,
        **{key: d["density"][key] for key in ("min_cluster_size", "min_samples", "allow_single_cluster")}) if d else None)
    set_analysis(result, density)
    st.session_state.update(controls)
    for key in ["source_a", "source_b", "dataset_editor", "cluster_mode"]:
        st.session_state.pop(key, None)
    if loaded.clustering:
        st.session_state.cluster_mode = {"auto": "Авто", "count": "Ровно K кластеров",
                                         "threshold": "По порогу T"}[loaded.clustering["mode"]]
    st.session_state.notice = "Сессия восстановлена. Набор, результаты и настройки загружены."

# Sessions already open during a source reload need a profile for their partition.
if "cluster_profile" not in st.session_state:
    set_analysis(st.session_state.cluster, st.session_state.get("density_cluster"))

# Detach shared controls from Streamlit's per-page widget cleanup.
for control_key, default in {**DEFAULTS, **GEN_DEFAULTS, **DENSITY_DEFAULTS}.items():
    st.session_state[control_key] = st.session_state.get(control_key, default)

cfg = st.session_state.cfg
items = st.session_state["items"]
sections = ["Модель", "Операции", "Набор величин", "Кластеры"]
if st.session_state.get("section") not in sections:
    st.session_state.section = "Модель"


def snapshot():
    controls = {key: st.session_state.get(key, val) for key, val in {**DEFAULTS, **GEN_DEFAULTS, **DENSITY_DEFAULTS}.items()}
    controls["k"] = min(int(controls["k"]), len(st.session_state["items"]))
    def exported(result, profile):
        return {**result.to_dict(), "membership": profile.to_dict()} if result is not None else None
    return SessionData(st.session_state["items"], st.session_state.results, cfg,
                       exported(st.session_state.cluster, st.session_state.cluster_profile),
                       controls, st.session_state.get("generation"),
                       exported(st.session_state.density_cluster, st.session_state.density_profile))


# This section is assembled into app.py; computational helpers above it are unchanged.
for key in ("source_a", "source_b", "operation", "full_tails", "cluster_mode"):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]

with st.sidebar:
    st.markdown('<p class="utility-brand">Рэлей</p>', unsafe_allow_html=True)
    page = st.radio("Раздел", sections, key="section", label_visibility="collapsed",
                    format_func=lambda name: "Набор" if name == "Набор величин" else name)
    st.divider()
    st.markdown(f'<p class="utility-session">Величин: {len(items)}<br>Результатов: {len(st.session_state.results)}<br>'
                f'ε = {cfg.epsilon:g}<br>Уровней α: {cfg.levels}</p>', unsafe_allow_html=True)
    st.download_button("Скачать сессию JSON", dumps_session(snapshot()), "rayleigh-session.json",
                       "application/json", width="stretch", key="session_download")

if "notice" in st.session_state:
    st.success(st.session_state.pop("notice"))


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 2}, "scrollZoom": False})


def dataset_caption():
    generation = st.session_state.get("generation")
    text = f"{len(items)} величин"
    if generation:
        text += f" · {SCENARIOS[generation['scenario']][0]} · seed {generation['seed']}"
    return text


def csv_download(label="Скачать CSV", key="dataset_download"):
    c, d = st.session_state.cluster_profile, st.session_state.density_profile
    st.download_button(label, "\ufeff" + dataset_csv(items, c.labels if c else None, d.labels if d else None,
                       scores=c.to_dict()["scores"] if c else None,
                       hdbscan_scores=d.to_dict()["scores"] if d else None,
                       cluster_alpha=st.session_state.cluster_alpha),
                       "rayleigh-clusters.csv", "text/csv", width="stretch", key=key)


def import_session():
    uploaded = st.file_uploader("Файл сессии JSON", type="json", key="session_upload")
    if st.button("Загрузить сессию", disabled=uploaded is None, width="stretch", key="load_session"):
        try:
            loaded = loads_session(uploaded.getvalue())
            restore_controls(loaded.controls, loaded.config, len(loaded.items))
            st.session_state.pending_session = loaded
            st.rerun()
        except (ValueError, UnicodeDecodeError) as error:
            st.error(str(error))


def show_gap(result):
    if result.mode != "auto":
        return
    with st.expander("Как выбрано число кластеров"):
        reason = result.auto["reason"]
        if reason == "identical":
            st.write("Одна группа: все величины имеют одинаковый σ.")
        elif reason == "small_sample":
            st.write("Для оценки числа групп нужно хотя бы три величины. Оставлена одна группа.")
        st.write("Сравниваем компактность разбиений со случайными равномерными наборами σ. "
                 "Находим лучший Gap и берём наименьшее K, которое уступает ему не больше чем на одну погрешность.")
        st.caption(f"Полная связь · до {result.auto['max_clusters']} групп · "
                   f"{result.auto['reference_samples']} контрольных наборов · seed {result.auto['seed']}")
        rows = result.auto["candidates"]
        if rows:
            go = __import__("plotly.graph_objects", fromlist=["Scatter"])
            fig = base_figure(height=300, x_title="Число кластеров K", y_title="Gap statistic")
            fig.add_trace(go.Scatter(x=[r["k"] for r in rows], y=[r["gap"] for r in rows],
                error_y=dict(type="data", array=[r["se"] for r in rows]), mode="lines+markers",
                line=dict(color=TEAL), name="Gap и погрешность"))
            chosen = next(r for r in rows if r["k"] == result.k)
            fig.add_trace(go.Scatter(x=[result.k], y=[chosen["gap"]], mode="markers",
                marker=dict(color="#D47B35", size=12), name=f"Выбрано K = {result.k}"))
            fig.update_xaxes(dtick=1)
            fig.add_hline(y=result.auto["cutoff"], line_dash="dot", line_color="#8999A8")
            chart(fig, "auto_selection")
            st.caption("Пунктир — лучший Gap минус одна погрешность; выбирается первое K не ниже линии.")
        if reason == "search_limit":
            st.caption("Выбор достиг границы доступного поиска.")
        st.caption("Результат зависит от критерия и контрольной модели; это оценка структуры данных.")


def cluster_inspection(result, density, provenance):
    tree_tab, distances_tab = st.tabs(["Дерево полной связи", "Матрица расстояний"])
    with tree_tab:
        chart(tree_figure(result), "cluster_tree")
        st.caption("Дерево исходного разбиения. Числа в скобках — размеры свёрнутых ветвей.")
    with distances_tab:
        st.caption("Расстояния по полному набору до отсечения.")
        heat = base_figure(height=370, x_title="Индекс объекта", y_title="Индекс объекта")
        heat.add_trace(__import__("plotly.graph_objects", fromlist=["Heatmap"]).Heatmap(z=result.distances,
            colorscale=[[0, "#F5F6F6"], [1, TEAL]], colorbar=dict(title="d")))
        chart(heat, "distance_heatmap")
    show_gap(result)
    if provenance:
        with st.expander("Проверить по исходным группам"):
            origins = np.array([provenance["origin_labels"][item.id] for item in items])
            clean = origins > 0
            values_table([{"Источников": len(provenance["components"]), "Групп полной связи": result.k or len(set(result.labels)),
                           "ARI исходного разбиения": f"{adjusted_rand_index(origins[clean], result.labels[clean]):.3f}"}])
            chart(generation_figure(items, provenance, result.labels), "generation_vs_clusters")
            st.caption("Метки генератора не передаются алгоритмам. ARI = 1 означает одинаковое разбиение, "
                       "около 0 — согласие на уровне случайного. Заданные выбросы исключены из ARI.")
            st.dataframe(pd.crosstab(pd.Series(["Выбросы" if v == 0 else f"Источник {v}" for v in origins], name="Создано"),
                pd.Series([cluster_name(v) for v in result.labels], name="Полная связь")), width="stretch")
            if density is not None:
                st.caption("Обычные объекты, отвергнутые HDBSCAN, учитываются в ARI с меткой 0. "
                           "При полном шуме ARI не выводится; учитывайте долю распределённых объектов.")
                if np.any(origins == 0):
                    values_table([{"Заданных выбросов попало в шум": f"{int(np.sum((origins == 0) & (density.labels == 0)))}/{int(np.sum(origins == 0))}",
                                   "Обычных объектов попало в шум": int(np.sum((origins > 0) & (density.labels == 0)))}])
            st.caption("При перекрытии источников одинаковые σ могут иметь разные метки. Точное восстановление бывает невозможно.")
    if density is not None:
        with st.expander("Сопоставление групп двух методов"):
            st.dataframe(pd.crosstab(pd.Series([cluster_name(v) for v in result.labels], name="Полная связь"),
                pd.Series([cluster_name(v) for v in density.labels], name="HDBSCAN")), width="stretch")


if page == "Модель":
    st.title("Модель")
    st.caption("Нечёткая величина Рэлея · α-срез обновляется при движении ползунка")
    live_model(st.session_state.model_sigma, st.session_state.alpha, cfg.epsilon,
               len(items) < 300, model_controls_changed, model_add_requested)
    with st.expander("О модели и уровне принадлежности"):
        st.latex(r"\mu_\sigma(x)=\frac{x}{\sigma}\exp\left(\frac{1-x^2/\sigma^2}{2}\right),\quad x>0")
        st.write("При x ≤ 0 принадлежность равна нулю. Максимум равен 1 при x = σ. "
                 "Степень принадлежности не является вероятностью. α-срез — интервал, на котором μ(x) ≥ α.")
        st.caption(f"В вычислениях используются {cfg.levels} уровней от ε = {cfg.epsilon:g} до 1. "
                   "ε отсекает бесконечные хвосты; границы исходных срезов находятся методом Брента.")

elif page == "Операции":
    st.title("Операции")
    st.caption(f"Интервальная арифметика по α-срезам · α = {st.session_state.alpha:g}")
    work, settings = workspace()
    pool = {i.id: i.value for i in items + st.session_state.results}
    operands = []
    with settings:
        st.subheader("Параметры")
        for letter, key in [("A", "a"), ("B", "b")]:
            options = [f"Ввести σ {letter}"] + list(pool)
            if st.session_state.get(f"source_{key}", options[0]) not in options:
                st.session_state.pop(f"source_{key}", None)
            source = st.selectbox(f"Величина {letter}", options, key=f"source_{key}")
            if source.startswith("Ввести"):
                sigma = st.number_input(f"σ {letter}", min_value=.000001, max_value=1_000_000.,
                    step=.1, key=f"op_{key}_sigma", format="%.6f")
                operands.append(RayleighFuzzy(sigma, letter))
            else:
                operands.append(pool[source])
        operation = st.selectbox("Операция", ["Сложение", "Вычитание", "Умножение", "Деление", "Сравнение"], key="operation")
        full_tails = st.checkbox("Полные рабочие хвосты", value=False, key="full_tails")
        save_slot = st.container()
    a, b = operands
    symbols = {"Сложение": "+", "Вычитание": "−", "Умножение": "*", "Деление": "/"}
    try:
        if operation == "Сравнение":
            comparison = compare(a, b, cfg)
            with work:
                chart(operation_figure(a, b, None, cfg, st.session_state.alpha, full_tails), "compare_chart")
                overlap = "Нет" if comparison.overlap_05 is None else f"[{comparison.overlap_05[0]:.5g}; {comparison.overlap_05[1]:.5g}]"
                values_table([{"По рангу": f"A {comparison.relation} B", "Расстояние": f"{comparison.distance:.6g}",
                               "Перекрытие при α = 0,5": overlap}])
                st.caption(f"Ранг A = {comparison.rank_a:.6g}; ранг B = {comparison.rank_b:.6g}")
        else:
            symbol = "-" if operation == "Вычитание" else symbols[operation]
            result = arithmetic(a, b, symbol, cfg)
            lo, hi = result.alpha_cut(st.session_state.alpha)
            with work:
                chart(operation_figure(a, b, result, cfg, st.session_state.alpha, full_tails), "operation_chart")
                values_table([{"Операция": f"A {symbols[operation]} B", "Левая граница": f"{lo:.6g}",
                               "Правая граница": f"{hi:.6g}", "Уровней α": cfg.levels}])
            with save_slot:
                if st.button("Сохранить результат", type="primary", width="stretch", key="save_operation",
                             disabled=len(st.session_state.results) >= 100):
                    taken, j = set(pool), 1
                    while f"C{j:03d}" in taken:
                        j += 1
                    st.session_state.results.append(FuzzyItem(f"C{j:03d}", result))
                    st.session_state.notice = f"Результат C{j:03d} сохранён."
                    st.rerun()
    except (ValueError, ZeroDivisionError) as error:
        with work:
            st.error(str(error))
    with work:
        with st.expander("Как выполняются операции"):
            st.write("Операции выполняются над интервалами на каждом α-уровне. Форма результата сохраняется без подгонки к Рэлею. "
                     "Операнды трактуются как независимые: A − A обычно является интервалом. "
                     "Деление через срез, содержащий ноль, запрещено.")
            st.write("Ранг — средний центр α-срезов. Совпадение рангов не гарантирует совпадения форм.")
            st.caption(f"Уровень α выбирается в разделе «Модель». ε = {cfg.epsilon:g}. "
                       "При отключённых полных хвостах график сфокусирован на текущем срезе.")

elif page == "Набор величин":
    st.title("Набор")
    st.caption(dataset_caption())
    work, settings = workspace()
    with settings:
        st.subheader("Генерация")
        scenario = st.selectbox("Сценарий", list(SCENARIOS), format_func=lambda key: SCENARIOS[key][0], key="generator_scenario")
        with st.form("generator"):
            n = st.number_input("Число величин", 100, 300, key="samples", step=1)
            seed = st.number_input("Seed", 0, 2**32 - 1, key="seed", step=1)
            if scenario == "uniform":
                st.number_input("σ min", min_value=.000001, max_value=1_000_000., key="gen_sigma_min", format="%.6f")
                st.number_input("σ max", min_value=.000001, max_value=1_000_000., key="gen_sigma_max", format="%.6f")
            elif scenario == "custom":
                st.number_input("Число источников генерации", 2, 8, key="gen_groups", step=1)
                st.slider("Разделённость источников", 1., 12., key="gen_separation", step=.5)
                st.slider("Неравномерность размеров, %", 0, 100, key="gen_imbalance")
                st.slider("Выбросы, %", 0, 20, key="gen_outliers")
            submitted = st.form_submit_button("Создать набор", type="primary", width="stretch", key="generate_dataset")
        if submitted:
            try:
                batch = generate_scenario(int(n), int(seed), scenario,
                    groups=st.session_state.gen_groups, separation=st.session_state.gen_separation,
                    imbalance=st.session_state.gen_imbalance/100, outlier_fraction=st.session_state.gen_outliers/100,
                    sigma_min=st.session_state.gen_sigma_min if scenario == "uniform" else .5,
                    sigma_max=st.session_state.gen_sigma_max if scenario == "uniform" else 6.)
                st.session_state["items"] = batch.items
                st.session_state.generation = batch.provenance
                set_analysis()
                st.session_state.k = min(st.session_state.k, len(batch.items))
                st.session_state.pop("dataset_editor", None)
                st.session_state.notice = f"Создан набор из {n} величин."
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        apply_slot = st.container()
        with st.expander("Файлы"):
            csv_download()
            st.download_button("Скачать сессию JSON", dumps_session(snapshot()), "rayleigh-session.json", "application/json",
                               width="stretch", key="dataset_session_download")
            import_session()
        with st.expander("О генерации"):
            st.write(SCENARIOS[scenario][1])
            st.caption("Генерируются параметры σ разных величин. Каждая кривая сохраняет форму Рэлея. "
                       "Число источников относится к генератору, алгоритмы исходных меток не видят.")
    with work:
        frame = pd.DataFrame({"id": [i.id for i in items], "sigma": [i.value.sigma for i in items]})
        edited = st.data_editor(frame, hide_index=True, height=520, width="stretch", key="dataset_editor",
            disabled=["id"], num_rows="fixed", column_config={"id": st.column_config.TextColumn("Величина"),
            "sigma": st.column_config.NumberColumn("Масштаб σ", min_value=.000001, max_value=1_000_000., format="%.6f", required=True)})
        st.caption("Измените σ в таблице и нажмите «Применить изменения».")
        provenance = st.session_state.get("generation")
        if provenance:
            with st.expander("Источники и параметры созданного набора"):
                chart(generation_figure(items, provenance), "generation_sources")
                st.dataframe(pd.DataFrame([{"Источник": c["label"], "Величин": c["size"], "Центр σ": c["center"],
                    "Разброс σ": c["spread"]} for c in provenance["components"]]), hide_index=True, width="stretch")
                st.caption(f"Выбросов: {provenance['outliers']}. Цвет показывает источник генерации.")
    with apply_slot:
        if st.button("Применить изменения", width="stretch", key="apply_dataset"):
            try:
                updated = [FuzzyItem(row.id, RayleighFuzzy(float(row.sigma), row.id)) for row in edited.itertuples()]
                changes = sum(x.value.sigma != y.value.sigma for x, y in zip(items, updated))
                if changes:
                    st.session_state["items"] = updated
                    set_analysis()
                    st.session_state.generation = None
                    st.session_state.notice = f"Обновлено величин: {changes}. Выполните кластеризацию заново."
                    st.rerun()
                st.info("Изменений нет.")
            except (ValueError, TypeError) as error:
                st.error(str(error))

elif page == "Кластеры":
    st.title("Кластеризация")
    st.caption(dataset_caption())
    work, settings = workspace()
    with settings:
        st.subheader("Полная связь")
        mode_ui = st.selectbox("Способ разбиения", ["Авто", "Ровно K кластеров", "По порогу T"],
            key="cluster_mode", format_func=lambda value: "Авто (Gap)" if value == "Авто" else value,
            label_visibility="collapsed")
        mode = {"Авто": "auto", "Ровно K кластеров": "count", "По порогу T": "threshold"}[mode_ui]
        if mode == "count":
            if st.session_state.k > len(items):
                st.session_state.k = len(items)
            param = st.number_input("Число кластеров K", 1, len(items), key="k", step=1)
        elif mode == "threshold":
            param = st.number_input("Порог расстояния T", min_value=0., key="threshold", step=.1)
        else:
            param = None
        st.subheader("HDBSCAN")
        compare_density = st.checkbox("Сравнить с HDBSCAN", key="compare_hdbscan")
        if compare_density:
            st.number_input("Минимальный размер группы", 2, 300, key="hb_min_cluster_size", step=1)
            st.number_input("Соседей для оценки плотности", 1, 300, key="hb_min_samples", step=1,
                            help="Включает сам объект. Большее значение делает поиск плотных областей строже.")
            st.checkbox("Разрешить одну группу", key="hb_allow_single")
        if st.button("Рассчитать", type="primary", width="stretch", key="run_clustering"):
            try:
                with st.spinner("Выполняем анализ…"):
                    sigmas = tuple(i.value.sigma for i in items)
                    primary = cached_cluster(sigmas, cfg.epsilon, cfg.levels, mode, param)
                    density = (cached_density(sigmas, cfg.epsilon, cfg.levels, st.session_state.hb_min_cluster_size,
                        st.session_state.hb_min_samples, st.session_state.hb_allow_single) if compare_density else None)
                set_analysis(primary, density)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        csv_download(key="cluster_download")
        with st.expander("О настройках"):
            st.write("Оба метода получают одинаковые расстояния по α-срезам. Полная связь объединяет группы "
                     "по самой далёкой паре. HDBSCAN ищет устойчивые плотные группы и может выделить шум.")
            st.write("Минимальный размер ограничивает размер самостоятельной группы. Число соседей задаёт оценку плотности, "
                     "включая сам объект. Разрешение одной группы допускает такой результат вместе с шумом.")
            st.caption("Выбираются устойчивые группы EOM. Если объектов меньше любого из двух минимумов, весь набор становится шумом.")
    with work:
        result, density = st.session_state.cluster, st.session_state.get("density_cluster")
        provenance = st.session_state.get("generation")
        if result is None:
            st.info("Нажмите «Рассчитать», чтобы получить разбиение.")
            unclustered = cluster_strip(items, np.ones(len(items), dtype=int))
            unclustered.update_yaxes(tickvals=["1"], ticktext=["Набор"], title_text="")
            unclustered.update_layout(showlegend=False)
            chart(unclustered, "unclustered")
        else:
            origins = np.array([provenance["origin_labels"][i.id] for i in items]) if provenance else None
            rows = []
            for name, r in [("Полная связь", result)] + ([("HDBSCAN", density)] if density is not None else []):
                noise = int(np.sum(r.labels == 0))
                row = {"Метод": name, "Групп": len(set(r.labels) - {0}), "Шум": noise,
                       "В группах, %": f"{100 * (len(items) - noise) / len(items):.1f}"}
                if origins is not None:
                    clean = origins > 0
                    row["ARI исходного разбиения"] = (f"{adjusted_rand_index(origins[clean], r.labels[clean]):.3f}"
                                  if np.any(r.labels[clean] > 0) else "—")
                rows.append(row)
            values_table(rows)
            actual = (f"Авто (Gap), K = {result.k}" if result.mode == "auto" else
                      f"K = {result.k}" if result.mode == "count" else f"T = {result.threshold:g}")
            used = ""
            if density is not None:
                d = density.density
                used = f" · HDBSCAN: размер {d['min_cluster_size']}, соседей {d['min_samples']}, одна группа {'да' if d['allow_single_cluster'] else 'нет'}"
            st.caption(f"Выполненный анализ: {actual}{used}. Новые настройки применяются по кнопке.")
            live_clusters(items, cfg, st.session_state.cluster_profile, st.session_state.density_profile,
                          st.session_state.cluster_alpha, st.session_state.cluster_revision, cluster_alpha_changed)
            if density is not None and density.k == 0:
                st.info("HDBSCAN оставил весь набор шумом. Устойчивых групп при этих настройках не найдено.")
            cluster_inspection(result, density, provenance)
