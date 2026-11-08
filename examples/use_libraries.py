"""Run from the project root: python -m examples.use_libraries"""
from pathlib import Path
from fuzzy_core import RayleighFuzzy, FuzzyItem, SessionData, add, compare, generate, dumps_session
from fuzzy_clustering import cluster_auto, cluster_membership, membership_mask

a, b = RayleighFuzzy(1, "A"), RayleighFuzzy(2, "B")
total = add(a, b)
print("Срез суммы при α = 0.5:", total.alpha_cut(.5))
print("Сравнение:", compare(a, b))
items = generate(n=150, seed=42)
clusters = cluster_auto(items)
print("Автоматически выбрано K:", clusters.k)
print("Размеры кластеров:", [(j, sum(clusters.labels == j)) for j in sorted(set(clusters.labels))])
profile = cluster_membership(items, clusters)
alpha = .43
print(f"При пороге соответствия {alpha} осталось:", membership_mask(profile, alpha).sum())
session = SessionData(items, [FuzzyItem("C001", total)],
                      clustering={**clusters.to_dict(), "membership": profile.to_dict()},
                      controls={"cluster_alpha": alpha})
output = Path("runtime")
output.mkdir(exist_ok=True)
(output / "example-session.json").write_text(dumps_session(session), encoding="utf-8")
print("Сессия сохранена: runtime/example-session.json")
