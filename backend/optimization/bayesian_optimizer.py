from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.simulation.printability_estimator import PrintabilityEstimator

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False


@dataclass
class SearchResult:
    candidates: pd.DataFrame
    best_objective: float


class BayesianMaterialOptimizer:
    def __init__(self, surrogate_model, bounds: dict[str, tuple[float, float]]):
        self.surrogate_model = surrogate_model
        self.bounds = bounds

    def _predict_objective(
        self,
        row: dict[str, float],
        target_strength: float | None = None,
    ) -> float:
        x_df = pd.DataFrame([row])
        pred_strength = float(self.surrogate_model.predict(x_df)[0])
        printability = float(PrintabilityEstimator.score(x_df)[0])

        penalty = 0.0
        if target_strength is not None and pred_strength < target_strength:
            penalty += (target_strength - pred_strength) * 2.0

        # Maximize strength and printability together.
        objective = pred_strength + printability * 10.0 - penalty
        return objective

    def optimize(
        self,
        n_trials: int = 100,
        target_strength: float | None = None,
        max_water_ratio: float | None = None,
        fiber_range: tuple[float, float] | None = None,
        top_k: int = 10,
    ) -> SearchResult:
        records: list[dict[str, Any]] = []

        if HAS_OPTUNA:
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def objective(trial):
                row: dict[str, float] = {}
                for name, (low, high) in self.bounds.items():
                    l, h = low, high
                    if name == "water_ratio" and max_water_ratio is not None:
                        h = min(h, max_water_ratio)
                    if name == "fiber" and fiber_range is not None:
                        l = max(l, fiber_range[0])
                        h = min(h, fiber_range[1])
                    row[name] = trial.suggest_float(name, l, h)

                score = self._predict_objective(row, target_strength)
                pred_strength = float(self.surrogate_model.predict(pd.DataFrame([row]))[0])
                row["pred_strength"] = pred_strength
                row["printability"] = float(PrintabilityEstimator.score(pd.DataFrame([row]))[0])
                row["objective"] = score
                records.append(row)
                return score

            study.optimize(objective, n_trials=n_trials)
            best_objective = float(study.best_value)
        else:
            best_objective = float("-inf")
            for _ in range(n_trials):
                row = {}
                for name, (low, high) in self.bounds.items():
                    l, h = low, high
                    if name == "water_ratio" and max_water_ratio is not None:
                        h = min(h, max_water_ratio)
                    if name == "fiber" and fiber_range is not None:
                        l = max(l, fiber_range[0])
                        h = min(h, fiber_range[1])
                    row[name] = float(np.random.uniform(l, h))

                score = self._predict_objective(row, target_strength)
                pred_strength = float(self.surrogate_model.predict(pd.DataFrame([row]))[0])
                row["pred_strength"] = pred_strength
                row["printability"] = float(PrintabilityEstimator.score(pd.DataFrame([row]))[0])
                row["objective"] = score
                best_objective = max(best_objective, score)
                records.append(row)

        result_df = pd.DataFrame(records).sort_values("objective", ascending=False).head(top_k)
        return SearchResult(candidates=result_df.reset_index(drop=True), best_objective=best_objective)
