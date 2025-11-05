# 学習パイプライン

本プロジェクトでは、自己対戦生成・学習・検証を `SelfPlayTrainer` が統合的に実行します。

## 1. 自己対戦生成

- `SelfPlayManager.generate(episodes, workers)` が MCTS を用いて自己対戦データを収集。
- 温度スケジュール（例: 序盤は高温、終盤は低温）や Dirichlet ノイズを適用可能。
- 複数スレッドで並列自己対戦を実行し、センター制覇率・死亡ターンなどを統計化。
- サンプルは `ReplayBuffer` に保存 (`state`, `π`, `z`)。

## 2. リプレイバッファ

- データ増強（D4 対称、色Permutation）、重複排除、サンプル検証（NaN/範囲チェック）。
- `to_state`/`load_state`/`save` で永続化可能。

## 3. トレーニング

- `Trainer.train_step()` がバッファからミニバッチ（バッチサイズは設定）を取得。
- 損失：`CE(policy) + MSE(value) + λ * L2`。勾配クリップと mixed precision 対応。
- policy エントロピーなどの統計も出力し、TensorBoard/W&B に記録。
- `validate_buffer_sample` により不正データ検知を実施。

## 4. オーケストレーション (`SelfPlayTrainer`)

- コンフィグは `SelfPlayTrainerConfig` の YAML (`configs/self_play.yaml`) で管理。
- 各イテレーションで自己対戦→学習 N ステップ→検証→ロギング→チェックポイント保存。
- TensorBoard: `train/*`, `buffer/size`, `self_play/center_rate_*` 等を出力。
- W&B (任意): ログ・ハイパー設定を自動記録。

## 5. CLI

- `scripts/run_iteration.py --config configs/self_play.yaml --iterations 10` 等で実行。
- 引数で探索回数、温度、ワーカー数、検証サンプルサイズなどを上書き可能。

## 6. チェックポイント

- `SelfPlayTrainer.save_checkpoint` がモデル・オプティマイザ・バッファ状態を保存。
- `ModelMetadata` を利用し、バージョン・学習日時・設定情報を同梱可能。

## 推奨運用

1. YAML で自己対戦条件・ハイパーパラメータを定義。
2. `run_iteration.py` で複数イテレーションを回し、TensorBoard/W&B で収束を監視。
3. 適宜 `scripts/gate_model.py` や CI ワークフローで評価/ゲーティングを実施。
4. 安定したモデルを checkpoint として保存し、推論エクスポート (`export_torchscript`, `export_onnx`) を行う。
