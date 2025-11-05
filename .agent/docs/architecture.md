# アーキテクチャ概要

Yonokuni AI は、完全情報 4 プレイヤー抽象ゲーム「Yonokuni」の学習エージェント構築を目的としたプロジェクトです。TypeScript ベースのゲームエンジンと Python/Gym 環境を共有し、AlphaZero 由来の自己対戦＋MCTS＋方策・価値ネットによる学習を行います。

## レイヤ構成

1. **ゲームロジック (`yonokuni/core`, `yonokuni/env`)**
   - `core/state.py` で盤面・プレイヤー状態を表現。
   - `core/rules.py` が移動、捕獲（同色サンド＆包囲）、勝敗判定を実装。
   - `env/gym_env.py` は Gymnasium API に準拠した学習環境を提供。

2. **特徴量・シンメトリ (`yonokuni/features`)**
   - 盤面テンソル生成、補助ベクトル構築 (`observation.py`)。
  - D4 対称性 + 色Permutationを扱うデータ拡張 (`symmetry.py`)。

3. **自己対戦・MCTS (`yonokuni/selfplay`, `yonokuni/mcts`)**
   - `SelfPlayManager` が学習用ステップを生成（温度スケジュール、並列ワーカー対応、センター制圧率・死亡ターンなどの集計付き）。
   - `mcts/puct.py` が PUCT ベースの探索を実装。Dirichlet ノイズ、価値符号反転をサポート。

4. **リプレイバッファ (`yonokuni/selfplay/replay_buffer.py`)**
   - 対称拡張付きのサンプリング、状態保存/復元、pickle 保存などを備える。
   - バッチ検証のため `yonokuni/validation/data_checks.py` と連携。

5. **ニューラルネット (`yonokuni/models`)**
   - ResNet ベースの方策・価値ネット (`network.py`)。
   - 推論ラッパ (`inference.py`) と TorchScript/ONNX エクスポート (`export.py`) を提供。

6. **学習オーケストレーション (`yonokuni/orchestration/loop.py`)**
   - `SelfPlayTrainer` が自己対戦→学習→検証→ログ記録を一括実行。
   - TensorBoard / W&B ロギング、チェックポイント保存、YAML 設定読込、温度スケジューリング等に対応。

7. **評価・ゲーティング (`yonokuni/evaluation`)**
   - `evaluate_policies`, `RuleBasedPolicy` による対戦評価。
   - `gating.py` による勝率閾値判定。

## データフロー

1. ゲーム状態 → 特徴量テンソル → ニューラルネット → 方策/価値。
2. `SelfPlayManager` が MCTS を用いて自己対戦を実行し、リプレイバッファへサンプル (`state`, `π`, `z`) を追加。
3. `Trainer` がバッファからサンプルを取り出し、`CE(policy) + MSE(value) + L2` 損失でネットワークを更新。
4. オーケストレーション層が各イテレーションの統計をロギングし、必要に応じてチェックポイントや評価を実施。

## CI ワークフロー

- `.github/workflows/evaluate-rule.yml`：ランダム vs ルールベース対戦を実施。
- `.github/workflows/gate.yml`：`gate_model.py` によるゲーティング判定を定期的に実行。
- これらは Pull Request/自動デプロイ前のヘルスチェックとして機能。

## 推論・配布

- `InferenceRunner` により mixed precision/batch 推論対応。
- `export_torchscript` / `export_onnx` で配布向けアーティファクト生成。
- `ModelMetadata` と checkpoint API がバージョン管理を補助。

この構成により、ルール実装・学習・評価・推論を一貫したパイプラインで扱えるようになっています。
