from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False


@dataclass
class TrainReport:
    rmse: float
    mae: float
    r2: float


class SurrogateModel:
    def __init__(self, model_type: str = "xgboost", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = self._create_model(self.model_type)
        self.feature_columns: list[str] = []
        self.target_column: str | None = None

    def _create_model(self, model_type: str):
        if model_type == "xgboost" and HAS_XGBOOST:
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
            )
        if model_type == "xgboost" and not HAS_XGBOOST:
            return RandomForestRegressor(n_estimators=300, random_state=self.random_state)
        if model_type == "random_forest":
            return RandomForestRegressor(n_estimators=300, random_state=self.random_state)
        if model_type == "gaussian_process":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
            return GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
        raise ValueError(f"Unsupported model_type: {model_type}")

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: Iterable[str],
        target_column: str,
        test_size: float = 0.2,
    ) -> TrainReport:
        self.feature_columns = list(feature_columns)
        self.target_column = target_column

        X = df[self.feature_columns].values
        y = df[target_column].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)

        pred = self.model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        mae = float(mean_absolute_error(y_val, pred))
        r2 = float(r2_score(y_val, pred))
        return TrainReport(rmse=rmse, mae=mae, r2=r2)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        return self.model.predict(X)

    def save(self, path: str):
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "model_type": self.model_type,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        payload = joblib.load(path)
        obj = cls(model_type=payload["model_type"])
        obj.model = payload["model"]
        obj.feature_columns = payload["feature_columns"]
        obj.target_column = payload["target_column"]
        return obj
