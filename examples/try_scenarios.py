"""Run: python -m examples.try_scenarios. No Streamlit/Plotly required."""
import numpy as np
from fuzzy_core import generate_scenario, SCENARIOS
from fuzzy_clustering import cluster_auto, adjusted_rand_index

for scenario, (title, _) in SCENARIOS.items():
    batch = generate_scenario(150, seed=42, scenario=scenario)
    result = cluster_auto(batch.items)
    origins = np.array([batch.provenance["origin_labels"][i.id] for i in batch.items])
    keep = origins > 0
    score = adjusted_rand_index(origins[keep], result.labels[keep])
    print(f"{title}: источников {len(batch.provenance['components'])}, "
          f"кластеров {result.k}, ARI без выбросов {score:.3f}")
