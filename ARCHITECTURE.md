# アーキテクチャ概要

Yonokuni AI は、Python/Gym を中心に 4 色チーム戦ゲームの AlphaZero パイプラインを構築するプロジェクトです。以下では、構成要素をレイヤ別に整理し、主要コンポーネントとデータフローを記述します。

## 1. ドメインモデリング層（Core Game Library）
- **State Model**：`GameState`（盤面 8×8、駒ごとの `player`・`is_dead`、捕獲数、死に駒フラグ、ターン情報、履歴）とユーティリティ（コピー、ハッシュ、チーム判定）。
- **Move Engine**：合法手列挙、移動検証、同色サンド取り、包囲取り、捕獲適用、死に駒化、中央勝利と敗北判定、手番スキップまでを手続き化した `apply_move`。
- **Rule Constants**：`BOARD_SIZE=8`, `STARTING_PIECES=6`, `DEAD_THRESHOLD=3`, 手数上限などを単一モジュールで定義。
- **Serialization**：状態・行動を JSON/NumPy 形式で保存・復元できるシリアライザ、ハッシュ化（Zobrist 等）を提供し MCTS/リプレイで再利用。

## 2. 環境・インターフェース層（Gym & Tooling）
- **Gym Environment (`YonokuniEnv`)**：Gymnasium API に基づき `reset`/`step`/`render`/`legal_actions` を提供。観測は 8 チャンネル盤面 + 手番/死に駒ワンホット、行動は 1,792 離散空間。
- **Observation Builder**：盤面データをテンソルへ変換し、死に駒/捕獲数などの補助情報を組み合わせるパイプライン。
- **Action Mapper**：行動 index ↔️ `(from_row, from_col, direction, distance)` のバイディレクション変換と合法手マスク生成。
- **Test Harness**：仕様ベースの決め打ちシナリオ、乱択フイジ、ゴールデンテストを包括し、環境 API とコアライブラリの整合性を保証。

## 3. 自己対戦・データ収集層
- **Self-Play Manager**：複数プロセス/スレッドで環境を並列起動し、各手番で MCTS を実行してアクションを決定。温度スケジュールや探索回数は設定ファイルで制御。
- **Replay Buffer**：`(state, π, z)` を保持するリングバッファ。データ拡張（D4 対称、色循環）の適用とサンプリングポリシー（均等/優先度）をサポート。
- **Logging & Telemetry**：自己対戦中の探索深度、価値推定、勝率、終局理由を記録し、異常検知や学習診断に利用。
- **Data Validators**：収集データの整合性（NaN、非合法手、勝敗矛盾）を定期チェックし、問題があればジョブを停止・警告。

## 4. 学習・推論層
- **Neural Network**：ResNet ベースの方策・価値ネット。入力テンソルに FiLM/加算で手番情報を条件付けし、1792 ロジットと [-1,1] 価値を同時出力。
- **Training Loop**：AdamW + Cosine Decay、mixed precision、勾配クリップ、L2 正則化、チェックポイント保存/再開、TensorBoard/W&B ログなどを内包。
- **MCTS Engine**：PUCT/Gumbel AlphaZero 実装。Dirichlet ノイズ、ルート温度、価値符号反転、訪問統計の抽出、バッチ推論でのポリシー・価値取得。
- **Evaluation Harness**：自己対戦以外に、ルールベース Bot・旧モデル・ランダムとの対戦を自動実行し、勝率・平均手数・中央勝利率をレポート。
- **Model Gating**：新旧モデルを対戦させ、勝率閾値（例：55%）を超えた場合のみ最新モデルを採用。結果とメタ情報をモデルレジストリに保存。

## 5. サービス統合・運用層
- **Inference Service**：推論専用エンジン（高速 MCTS またはポリシー単独）をモジュール化し、API/CLI/バッチから利用可能にする。
- **Configuration & Orchestration**：Hydra 等で全設定を一元管理し、学習・自己対戦・評価・推論ジョブを Airflow/Kubernetes/Slurm でスケジュール。
- **Monitoring & Dashboard**：学習曲線、探索指標、リソース使用率、モデルバージョン、勝率トレンドを可視化し、異常時に通知。
- **Data & Artifact Management**：モデル、リプレイバッファ、評価ログをオブジェクトストレージに保存し、バージョン管理とバックアップを行う。
- **Deployment Pipeline**：ゲーティング通過モデルを本番向けにタグ付けし、ローリングアップデート/ロールバック手順を標準化。

## データフロー
1. **Game Library** が合法手と状態遷移を提供し、Gym 環境が観測・合法手マスクを生成。
2. **Self-Play Manager** が Gym 環境 + MCTS + 最新モデルで自己対戦を実行し、`(state, π, z)` をリプレイバッファへ送る。
3. **Training Loop** がバッファからサンプルを読み出し、データ拡張を適用してニューラルネットを更新。新しいチェックポイントを生成。
4. **Evaluation Harness** が最新と参照モデルを対戦させ、メトリクスをモニタリング/ダッシュボードへ送信。ゲーティング結果に従いモデルを昇格。
5. **Inference Service** が採用済みモデルで対局 API を提供し、運用メトリクスを監視。必要に応じて学習ループへフィードバック（例：対局ログ）を返す。

このアーキテクチャにより、ルール実装・環境・学習・評価・運用が明確に分離されつつも、データフローは一方向に連結され、再現性の高い AlphaZero 学習基盤を構築できます。
