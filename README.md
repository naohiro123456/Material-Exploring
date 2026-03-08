# Material AI Lab

建築3Dプリンター材料向けのサロゲートモデル学習と材料探索の最小実装です。

## Structure

- `backend/data`: 材料データCSV
- `backend/models`: サロゲートモデル
- `backend/optimization`: ベイズ最適化
- `backend/simulation`: 3DP特性推定
- `backend/api`: FastAPI連携
- `frontend`: Streamlit UI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run UI

```bash
python main.py
```

または

```bash
streamlit run frontend/ui.py
```

## Run API

```bash
uvicorn backend.api.server:app --reload --port 8000
```

## Workflow

1. `Material Database` でCSVをアップロード/編集
2. `Train Surrogate Model` で目的変数とモデル種別を選び学習
3. `Material Discovery` で目標強度や制約を設定して候補配合を生成
4. `Visualization` で2D/3D散布図を確認

## Notes

- `xgboost` が無い場合、`xgboost`選択時は `random_forest` を代替利用します。
- `gpytorch` / `botorch` は将来拡張用に依存を含めています。
