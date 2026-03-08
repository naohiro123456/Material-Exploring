from __future__ import annotations

import numpy as np
import pandas as pd


class PrintabilityEstimator:
    """Simple heuristic estimator for buildability/printability.

    This is a placeholder until a physics-informed model is integrated.
    """

    @staticmethod
    def score(df: pd.DataFrame) -> np.ndarray:
        water = df.get("water_ratio", 0.35)
        fiber = df.get("fiber", 0.5)
        sp = df.get("superplasticizer", 0.8)
        sand = df.get("sand_ratio", 0.45)

        # Center around practical ranges for extrusion-based 3DP.
        score = (
            1.0
            - np.abs(water - 0.33) * 2.4
            - np.abs(sand - 0.47) * 1.8
            - np.maximum(0, fiber - 1.3) * 0.6
            - np.abs(sp - 0.9) * 0.8
        )
        return np.clip(score, 0.0, 1.0)
