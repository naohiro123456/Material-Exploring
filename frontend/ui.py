from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.models.surrogate_model import HAS_XGBOOST, SurrogateModel
from backend.optimization.bayesian_optimizer import BayesianMaterialOptimizer

DATA_PATH = ROOT / "backend" / "data" / "materials.csv"
MODEL_PATH = ROOT / "backend" / "models" / "surrogate.joblib"

st.set_page_config(page_title="Material AI Lab", layout="wide")
st.title("Material AI Lab - 建築3Dプリンター材料サロゲート")


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_working_df() -> pd.DataFrame:
    if "df" not in st.session_state:
        st.session_state.df = load_default_data()
    return st.session_state.df


def infer_bounds(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in feature_columns:
        cmin = float(df[col].min())
        cmax = float(df[col].max())
        margin = (cmax - cmin) * 0.1 if cmax > cmin else 0.05
        bounds[col] = (cmin - margin, cmax + margin)
    return bounds


tab1, tab2, tab3, tab4 = st.tabs(
    ["Material Database", "Train Surrogate Model", "Material Discovery", "Visualization"]
)

with tab1:
    st.subheader("材料データ管理")
    uploaded = st.file_uploader("CSVアップロード", type=["csv"])
    if uploaded is not None:
        st.session_state.df = pd.read_csv(uploaded)

    current_df = get_working_df()
    edited_df = st.data_editor(current_df, num_rows="dynamic", use_container_width=True)
    st.session_state.df = edited_df

    if st.button("CSV保存", key="save_csv"):
        edited_df.to_csv(DATA_PATH, index=False)
        st.success(f"Saved: {DATA_PATH}")

with tab2:
    st.subheader("モデル学習")
    df = get_working_df()
    columns = df.columns.tolist()

    target = st.selectbox("Target property", columns, index=columns.index("compressive_strength") if "compressive_strength" in columns else len(columns) - 1)
    default_features = [c for c in columns if c != target]
    feature_cols = st.multiselect("Feature columns", columns, default=default_features)

    model_options = ["xgboost", "random_forest", "gaussian_process"]
    if not HAS_XGBOOST and "xgboost" in model_options:
        st.info("xgboost未導入のため、xgboost選択時はrandom_forestで代替します。")
    model_type = st.selectbox("Model type", model_options)

    if st.button("Train Model", key="train_model"):
        if not feature_cols:
            st.error("Feature columnsを1つ以上選択してください。")
        else:
            surrogate = SurrogateModel(model_type=model_type)
            report = surrogate.train(df, feature_cols, target)
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            surrogate.save(str(MODEL_PATH))
            st.session_state.surrogate = surrogate
            st.session_state.feature_cols = feature_cols
            st.session_state.target = target
            st.success("Model trained and saved.")
            st.write({"rmse": report.rmse, "mae": report.mae, "r2": report.r2})

with tab3:
    st.subheader("材料探索")

    if "surrogate" not in st.session_state and MODEL_PATH.exists():
        st.session_state.surrogate = SurrogateModel.load(str(MODEL_PATH))
        st.session_state.feature_cols = st.session_state.surrogate.feature_columns
        st.session_state.target = st.session_state.surrogate.target_column

    if "surrogate" not in st.session_state:
        st.warning("先にモデル学習を実行してください。")
    else:
        surrogate = st.session_state.surrogate
        feature_cols = st.session_state.feature_cols
        df = get_working_df()
        bounds = infer_bounds(df, feature_cols)

        target_strength = st.number_input("Target strength (MPa)", value=40.0)
        default_max_water = min(bounds.get("water_ratio", (0.25, 0.5))[1], 0.40)
        max_water_ratio = st.number_input("Max water ratio", value=float(default_max_water))

        fiber_low_default, fiber_high_default = bounds.get("fiber", (0.0, 1.0))
        fiber_range = st.slider(
            "Fiber range",
            min_value=float(fiber_low_default),
            max_value=float(fiber_high_default),
            value=(float(fiber_low_default), float(fiber_high_default)),
        )

        n_trials = st.slider("Search trials", min_value=20, max_value=500, value=120, step=20)
        top_k = st.slider("Top candidates", min_value=3, max_value=30, value=10)

        if st.button("Generate Candidate Materials", key="generate_candidates"):
            optimizer = BayesianMaterialOptimizer(surrogate, bounds)
            result = optimizer.optimize(
                n_trials=n_trials,
                target_strength=target_strength,
                max_water_ratio=max_water_ratio,
                fiber_range=(float(fiber_range[0]), float(fiber_range[1])),
                top_k=top_k,
            )
            st.session_state.candidates = result.candidates
            st.write(f"Best objective: {result.best_objective:.3f}")
            st.dataframe(result.candidates, use_container_width=True)

with tab4:
    st.subheader("可視化")
    df = get_working_df()

    if "water_ratio" in df.columns and "compressive_strength" in df.columns:
        fig2d = px.scatter(
            df,
            x="water_ratio",
            y="compressive_strength",
            color="sand_ratio" if "sand_ratio" in df.columns else None,
            title="water_ratio vs compressive_strength",
        )
        st.plotly_chart(fig2d, use_container_width=True)

    if all(c in df.columns for c in ["water_ratio", "sand_ratio", "compressive_strength"]):
        fig3d = px.scatter_3d(
            df,
            x="water_ratio",
            y="sand_ratio",
            z="compressive_strength",
            color="compressive_strength",
            title="3D Material Space",
        )
        st.plotly_chart(fig3d, use_container_width=True)

    if "candidates" in st.session_state:
        st.markdown("#### Candidate Materials")
        st.dataframe(st.session_state.candidates, use_container_width=True)
