# RL Research

強化学習アルゴリズムを、小さく再現可能な実験として検証するためのリポジトリです。

## セットアップとテスト

```powershell
uv sync --project lab
uv run --project lab python -m unittest discover -s lab/tests
```

CUDA対応GPUが利用可能な場合、学習は自動的にGPUを使用します。

## CartPole

4次元の状態を入力する方策をREINFORCEまたはActor-Criticで学習します。

```powershell
uv run --project lab cartpole --config experiments/cartpole_reinforce.toml
uv run --project lab cartpole --config experiments/cartpole_actor_critic.toml
uv run --project lab plot-results runs/<run-directory>
```

利用できるモデルは `linear`、`mlp_32`、`mlp_64x64`、
`transformer_d32_h4_ff64` です。Transformer名の `d32` は埋め込み次元、
`h4` はAttentionヘッド数、`ff64` はFeedforward層の幅を表します。

現在のREINFORCE設定はTransformerを3,000エピソード、Actor-Critic設定は
4モデルを30,000エピソード、それぞれ3つのseedで学習します。

モデルや学習率を一時的に上書きできます。

```powershell
uv run --project lab cartpole --config experiments/cartpole_actor_critic.toml `
    --models transformer_d32_h4_ff64 --learning-rate 0.0003
```

学習曲線、確定した設定、モデル、評価結果、比較CSVとグラフは `runs/` に
保存されます。`Ctrl+C` を1回押すと、その時点の結果と
再開用チェックポイントを保存します。

```powershell
uv run --project lab cartpole `
    --config experiments/cartpole_actor_critic.toml `
    --resume runs/<run-directory>/<model>/seed-<seed>/checkpoint.pt
```

## Pixel CartPole

4枚のグレースケール84×84フレームを入力するCNN方策を学習します。
`cnn_16x32_fc128` と `cnn_16x32_fc512x128` が利用でき、現在の実験設定は
`cnn_16x32_fc128` を使用します。

```powershell
uv run --project lab cartpole --config experiments/cartpole_pixels_reinforce.toml
uv run --project lab cartpole --config experiments/cartpole_pixels_actor_critic.toml
```

`episodes_per_update` 個の環境を並列実行し、1回のバッチ推論を共有します。

## 方策蒸留

学習済みのベクトル方策を教師として、Pixel CNNへ蒸留します。

```powershell
uv run --project lab distill-cartpole `
    --teacher runs/<run-directory>/mlp_32/seed-42/model.pt `
    --teacher-model mlp_32
```

同じ教師から状態入力のTransformerへ蒸留する場合：

```powershell
uv run --project lab distill-cartpole `
    --teacher runs/<run-directory>/mlp_32/seed-42/model.pt `
    --teacher-model mlp_32 `
    --student-model transformer_d32_h4_ff64 `
    --student-observation state
```

## Structure

```text
lab/src/rl_lab/
├─ algorithms/
│  ├─ reinforce.py                    # REINFORCE
│  ├─ actor_critic.py                 # Actor-Criticと評価
│  └─ distillation.py                 # 方策蒸留
├─ envs/
│  ├─ base.py                         # 環境アダプターのProtocol
│  ├─ cartpole.py                     # 状態入力CartPole
│  └─ cartpole_pixels.py              # Pixel入力CartPole
├─ models/
│  ├─ linear.py
│  ├─ mlp_32.py
│  ├─ mlp_64x64.py
│  ├─ transformer_d32_h4_ff64.py
│  ├─ cnn_16x32_fc128.py
│  └─ cnn_16x32_fc512x128.py
├─ cartpole.py                        # 設定、実行、集計
├─ distill_cartpole.py                # 蒸留CLI
└─ plot_results.py                    # 結果の可視化
```
