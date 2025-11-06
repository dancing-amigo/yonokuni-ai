# Yonokuni AI

Yonokuni AI は、4 色チーム戦の完全情報ボードゲームに AlphaZero 系強化学習を適用し、自己対戦によって強力なエージェントを育成するプロジェクトです。実運用の TypeScript ゲームエンジンと、学習用の Python/Gym 環境を厳密に同期させ、学習とサービス提供で同一ルールが動作することを重視しています。

## プロジェクトの目的
- 8×8 盤・4 プレイヤー（赤/青/黄/緑）、2 チーム構成（赤&黄 vs 青&緑）の抽象ゲームに対し、AlphaZero（自己対戦 + MCTS + 方策・価値ネットワーク）を導入する。
- TypeScript 製ゲームエンジン（UI/サーバ）と Python/Gym 学習環境を同一仕様で実装・検証し、実運用と学習パイプラインの整合性を確保する。
- ルール固有の取り方（同色サンド・包囲取り）や死に駒処理、中央 4 マス勝利条件などを厳密に再現し、高度なデータ拡張・評価基盤を備えた学習パイプラインを構築する。

## クイックセットアップ

1. 仮想環境を作成し、依存をインストールします。
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

2. 学習イテレーションを実行します。
   ```bash
   .venv/bin/python scripts/run_iteration.py        --config configs/self_play.yaml        --iterations 5        --episodes 32        --train-steps 64
   ```

3. TensorBoard でメトリクスを確認します。
   ```bash
   .venv/bin/tensorboard --logdir logs/run_iteration
   ```
   ブラウザで http://localhost:6006 を開くと学習の進捗が見られます。

## プロジェクトの中身を知りたいとき

詳しい実装ドキュメントは `AGENT.md` および `.agent/docs/` 以下にまとめています。
- 全体アーキテクチャ: `.agent/docs/architecture.md`
- 学習パイプライン: `.agent/docs/training_pipeline.md`
- CLI ツール: `.agent/docs/cli_tools.md`

## 主なディレクトリ
- `yonokuni/`: Python 実装（ゲームルール、Env、MCTS、Self-Play、モデル、オーケストレーションなど）
- `scripts/`: 学習・評価・プレイ用 CLI スクリプト
- `tests/`: ユニットテスト
- `.agent/docs/`: 実装ドキュメント

## ライセンス
MIT License
