"""Run: python -m examples.compare_methods. Uses no interface or source labels."""
from pathlib import Path
from fuzzy_core import generate_scenario, SessionData, dumps_session, dataset_csv
from fuzzy_clustering import cluster_auto, cluster_hdbscan

batch = generate_scenario(150, seed=42, scenario="outliers")
complete = cluster_auto(batch.items)
density = cluster_hdbscan(batch.items, min_cluster_size=10, min_samples=10,
                          allow_single_cluster=True)
print(f"Полная связь + Gap: {complete.k} групп")
print(f"HDBSCAN: {density.k} групп, шум: {sum(density.labels == 0)} объектов")
session = SessionData(batch.items, clustering=complete.to_dict(), comparison=density.to_dict(),
                      generation=batch.provenance)
# Save explicitly in the directory from which the example is run.
Path("comparison-session.json").write_text(dumps_session(session), encoding="utf-8")
Path("comparison-clusters.csv").write_text(dataset_csv(batch.items, complete.labels, density.labels), encoding="utf-8-sig")
