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

## Windows + GPU 環境

Windows 10 / RTX 5090 のような GPU マシンでも同じコードで動作します（Python 3.11.x も pyproject の `>=3.10,<3.12` に含まれるのでサポート対象です）。PowerShell 例:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 先に CUDA 版の torch を固定
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.0+cu121 torchvision==0.17.0+cu121 --no-cache-dir
pip install -r requirements.txt
pip install -e .
```

GPU で学習する場合は `configs/self_play_windows_gpu.yaml` を指定すると、`training.device=cuda` や高バッチサイズがセットされた状態で `scripts/run_iteration.py` を実行できます:

```powershell
.\.venv\Scripts\python.exe scripts/run_iteration.py `
  --config configs/self_play_windows_gpu.yaml `
  --iterations 200
```

### GPU 待機時間の監視

- `nvidia-smi dmon -s pucm` で GPU 利用率/メモリ/コピー帯域を継続監視。
- Windows 標準の「リソース モニター」や `Get-Counter '\Processor(_Total)\% Processor Time'` で CPU 飽和をチェック。
- TensorBoard (`.\.venv\Scripts\tensorboard.exe --logdir logs/windows_gpu --bind_all`) を併用して学習待ち時間を可視化。

Windows GPU 最適化の詳細は `docs/windows_gpu_training.md` を参照してください。

## 長時間トレーニング (200 iteration)

`configs/self_play_long.yaml` は GPU 前提で 200 iteration を一気に回すための設定です。

- 1 iteration あたり 64 ゲーム生成 / 256 ステップ学習
- 20 iteration ごとに checkpoint と評価 (log: `logs/long_run`, checkpoints: `checkpoints/long_run`)
- MCTS シミュレーション 128 回、バッファ 150 万局面

実行例:

```bash
.venv/bin/python scripts/run_iteration.py \
  --config configs/self_play_long.yaml \
  --iterations 200 \
  --resume-from checkpoints/long_run/checkpoint_00020.pt  # 中断・再開したい場合
```

20 iteration ごとに評価する簡易ループ:

```bash
for step in $(seq 20 20 200); do
  ckpt=checkpoints/long_run/checkpoint_$(printf "%05d" "$step").pt
  [ -f "$ckpt" ] || continue
  .venv/bin/python scripts/evaluate_model.py "$ckpt" \
    --episodes 40 \
    --mcts-simulations 128 \
    --baseline rule \
    > logs/eval_long_run_step${step}.json
done
```

GPU・CPU の空き状況によって `self_play_workers` や `episodes_per_iteration` を下げてください。

## AI と対戦する

学習済みチェックポイントを使ってコンソール上で AI と対戦できます。

```bash
.venv/bin/python scripts/play_vs_ai.py \
  --checkpoint checkpoints/run_iteration/checkpoint_00010.pt \
  --human-team A \
  --mcts-simulations 128 \
  --temperature 0.6 \
  --log-file logs/play/session001.json
```

- `--checkpoint` を省略するとランダム方策が相手になります。
- `--human-team` は先後（Team A = 赤/黄, Team B = 青/緑）を選択。
- `--log-file` を指定すると対局ログ(JSON)が保存され、`--replay-log path.json` で後から棋譜を再生できます（`--replay-quiet` で盤面表示を抑制）。
- `--max-ply` で最大手数、`--temperature` で AI の多様性を調整できます。

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
