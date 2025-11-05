# Yonokuni AI ドキュメント概要

このディレクトリには、本リポジトリに実装された Yonokuni AI プロジェクトの主要構成要素と運用方法をまとめたドキュメントを配置しています。目的別にファイルを分割し、開発者が実装を俯瞰しやすい形を目指しています。

## 構成

- `architecture.md` — 全体アーキテクチャと主要コンポーネントの関係を解説。
- `game_rules.md` — ゲームルールと TypeScript/Python 実装の要点。
- `training_pipeline.md` — 自己対戦生成、学習ループ、オーケストレーションの流れと設定方法。
- `evaluation_and_gating.md` — 評価用スクリプト、ゲーティングワークフロー、CI 連携の詳細。
- `cli_tools.md` — CLI スクリプト（`run_iteration`, `evaluate_legacy`, `gate_model`, `play_vs_ai`）の使い方。
- `inference_and_export.md` — 推論ユーティリティ、TorchScript/ONNX エクスポート、メタデータ保存について。

適宜この構成を拡張し、学習ログや運用ガイドラインなどの情報も追記していく想定です。
