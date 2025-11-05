# CLI ツール

## `scripts/run_iteration.py`

学習イテレーションを実行するメインスクリプト。YAML 設定 (`configs/self_play.yaml`) をロードし、自己対戦→学習を繰り返す。

```bash
python scripts/run_iteration.py --config configs/self_play.yaml --iterations 5
```

主なオプション：
- `--episodes` / `--train-steps` — イテレーションあたりの自己対戦数・学習ステップ数。
- `--temperature`, `--self-play-workers`, `--validation-sample-size` — 温度スケジュール、並列ワーカー、データ検証サンプル数の上書き。
- `--wandb-project` — W&B ログを有効化。

## `scripts/evaluate_model.py`

指定したモデルをルールベース／ランダム政策と対戦させ評価する。

```bash
python scripts/evaluate_model.py checkpoints/latest.pt --baseline rule --episodes 50
```

## `scripts/evaluate_legacy.py`

任意のチェックポイント同士を評価する汎用ツール。`random` / `rule` も指定可能。

```bash
python scripts/evaluate_legacy.py checkpoints/new.pt checkpoints/old.pt --episodes 100
```

## `scripts/gate_model.py`

評価＋ゲーティングを一括実行。勝率閾値 (`--threshold`) と最低対局数 (`--min-games`) を指定。

```bash
python scripts/gate_model.py checkpoints/new.pt checkpoints/old.pt --episodes 100 --threshold 0.55 --min-games 40
```

## `scripts/play_vs_ai.py`

コンソール対戦ツール（ログ機能付き）。

- 対戦ログ保存：`--log-file game.json`
- ログ再生：`--replay-log game.json [--replay-quiet]`

```bash
python scripts/play_vs_ai.py --checkpoint checkpoints/latest.pt --log-file logs/game.json
python scripts/play_vs_ai.py --replay-log logs/game.json
```

## `scripts/evaluate_legacy.py` & CI

CI ワークフロー `.github/workflows/evaluate-rule.yml` および `gate.yml` から実行され、自動評価・ゲーティングを実現。
