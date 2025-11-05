# 評価・ゲーティング

## 評価スクリプト

- `scripts/evaluate_model.py`
  - チェックポイントを読み込み、ランダム or ルールベースと対戦。
  - `--baseline {random|rule}` で基準政策を選択。
  - 結果（勝数・平均手数など）を JSON で標準出力。

- `scripts/evaluate_legacy.py`
  - 任意の 2 チェックポイント／ポリシー（`random`/`rule`も可）を対戦させる汎用評価ツール。

## ゲーティング

- `yonokuni/evaluation/gating.py` に `gate_model` 関数を実装。
  - 勝率と対局数をチェックし、閾値を満たす場合のみ昇格。
- `scripts/gate_model.py` で評価＋ゲーティングを一括実行。
  - JSON で `promote` フィールドを返し、昇格可否を明示。

## CI ワークフロー

- `.github/workflows/evaluate-rule.yml`
  - PR / push 時に `evaluate_model.py random --baseline rule` を実行し、ルールベース基準の健全性をチェック。
- `.github/workflows/gate.yml`
  - `scripts/gate_model.py random rule` を実施し、閾値に達しているか確認。

## ロギングとダッシュボード

- `SelfPlayTrainer.iteration()` が以下を TensorBoard / W&B に記録：
  - 学習損失平均、ポリシーエントロピー、バッファサイズ。
  - self-play 勝率（Team A/B）、引き分け率、平均手数、中央制覇率、死亡ターン平均。
- これにより、学習進行や挙動の変化をリアルタイムにモニタリング可能。

## 推奨評価フロー

1. 学習イテレーションの合間に `scripts/evaluate_model.py` で基本性能を確認。
2. 閣僚モデルとの比較や昇格判定には `scripts/gate_model.py` を利用。
3. CI ワークフローで評価を自動化し、回帰抑制。
