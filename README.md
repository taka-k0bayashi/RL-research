# RL Research

強化学習アルゴリズムを、小さく再現可能な実験として検証するためのリポジトリです。

## CartPole

```powershell
uv sync --project lab
uv run --project lab cartpole --config experiments/cartpole_reinforce.toml
uv run --project lab plot-results runs/<run-directory>
```

Linear、MLP(32)、MLP(64×64)を3つのseedで学習します。学習曲線、確定した
設定、モデル、評価結果、比較CSVとグラフは `runs/` に保存されます。

```powershell
uv run --project lab python -m unittest discover -s lab/tests
```

## Structure

```text
lab/src/rl_lab/
├─ algorithms/reinforce.py  # Training and evaluation algorithm
├─ envs/
│  ├─ base.py               # Environment adapter contract
│  └─ cartpole.py           # Gymnasium CartPole adapter
├─ models/
│  ├─ linear.py
│  ├─ mlp_32.py
│  └─ mlp_64x64.py
└─ cartpole.py              # Configuration, execution, and aggregation
```
