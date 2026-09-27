# RL Research

強化学習アルゴリズムを、小さく再現可能な実験として検証するためのリポジトリです。

## CartPole

```powershell
uv sync --project lab
uv run --project lab cartpole --config experiments/cartpole_reinforce.toml
uv run --project lab cartpole --config experiments/cartpole_actor_critic.toml
uv run --project lab plot-results runs/<run-directory>
```

CUDA対応GPUが利用可能な場合、学習は自動的にGPUを使用します。

モデルや学習率を一時的に絞る場合：

```powershell
uv run --project lab cartpole --config experiments/cartpole_actor_critic.toml `
    --models mlp_64x64 --learning-rate 0.0003
```

Linear、MLP(32)、MLP(64×64)を3つのseedで学習します。学習曲線、確定した
設定、モデル、評価結果、比較CSVとグラフは `runs/` に保存されます。

```powershell
uv run --project lab python -m unittest discover -s lab/tests
```

## Pixel observations

The pixel experiment compares FC128 and FC512x128 CNN policies using four
grayscale 84 x 84 frames.

```powershell
uv run --project lab cartpole --config experiments/cartpole_pixels_reinforce.toml
uv run --project lab cartpole --config experiments/cartpole_pixels_actor_critic.toml
```

Press `Ctrl+C` once to save the current metrics and model before exiting.
`episodes_per_update` environments run concurrently and share one batched CNN
inference call.
Resume to the total episode count in the config with:

```powershell
uv run --project lab cartpole `
    --config experiments/cartpole_pixels_actor_critic.toml `
    --resume runs/<run-directory>/<model>/seed-<seed>/checkpoint.pt
```

## Structure

```text
lab/src/rl_lab/
├─ algorithms/
│  ├─ reinforce.py             # REINFORCE
│  └─ actor_critic.py          # Actor-Critic and evaluation
├─ envs/
│  ├─ base.py               # Environment adapter contract
│  └─ cartpole.py           # Gymnasium CartPole adapter
├─ models/
│  ├─ linear.py
│  ├─ mlp_32.py
│  └─ mlp_64x64.py
└─ cartpole.py              # Configuration, execution, and aggregation
```
