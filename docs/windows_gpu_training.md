# Windows GPU トレーニング最適化ガイド

## ハードウェア前提
- OS: Windows 10 Pro 2009 / Build 26100
- GPU: NVIDIA GeForce RTX 5090（ドライバ 32.0.15.7688, 24 GB クラス想定）
- CPU: Intel Core Ultra 9 285 (24C / 24T)
- メモリ: 64 GB

この構成は `SelfPlayTrainer`（`yonokuni/orchestration/loop.py`）でのニューラルネット学習を GPU に、`SelfPlayManager`（`yonokuni/selfplay/self_play.py`）での自己対戦 + MCTS を CPU 並列に割り振ると最もスループットが出ます。

---

## 1. Windows + CUDA セットアップ

### 1.1 ドライバとツール
1. NVIDIA ドライバを最新化（最低でも CUDA 12.3 互換）。`nvidia-smi` で 12.x が表示されれば OK。
2. Visual C++ Build Tools / Windows SDK（PyTorch が一部で必要）。

### 1.2 PowerShell での環境構築
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.0+cu121 torchvision==0.17.0+cu121 --no-cache-dir
pip install -r requirements.txt  # ここでは CPU 版 torch が入らないよう先に GPU 版を固定
pip install -e .
```

### 1.3 動作確認
```powershell
python - <<'PY'
import torch, platform
print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("is_available", torch.cuda.is_available())
print("device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
```

---

## 2. GPU 向け推奨コンフィグ

`configs/self_play_windows_gpu.yaml` を追加しました。`scripts/run_iteration.py`（`README.md` に記載の CLI）からそのまま指定できます。

| フィールド | 値 | 目的 |
| --- | --- | --- |
| `episodes_per_iteration` | 96 | 自己対戦のバッチを大型化し CPU の 24C を活用 |
| `training_steps_per_iteration` | 512 | GPU に十分なバッチを供給 |
| `self_play_workers` | 12 | `ThreadPoolExecutor`（`yonokuni/selfplay/self_play.py:159-197`）がスケールする上限付近 |
| `mcts.num_simulations` | 160 | MCTS 精度と GPU 評価コストのバランス |
| `training.batch_size` | 512 | RTX 5090 の VRAM で余裕ある範囲のバッチサイズ |
| `training.device` | `cuda` | `Trainer` が `torch.from_numpy(...).to(device="cuda")`（`yonokuni/training/loop.py:63-71`）を使うよう強制 |

実行例（PowerShell）:
```powershell
.\.venv\Scripts\python.exe scripts/run_iteration.py `
  --config configs/self_play_windows_gpu.yaml `
  --iterations 200 `
  --self-play-workers 12 `
  --mcts-simulations 160
```

> `--self-play-workers` や `--mcts-simulations` を CLI 引数で上書きすると、`yaml` の値より優先されます（`scripts/run_iteration.py:20-90`）。

---

## 3. パフォーマンス調整レバー

### 3.1 GPU を常時活性化
- `SelfPlayTrainer` は `config.training_config.device` が `None` の場合 `cpu` にフォールバックします（`yonokuni/orchestration/loop.py:46-58`）。Windows 用コンフィグでは `cuda` を必ず指定してください。
- 長時間学習では以下の一行を `scripts/run_iteration.py` の imports 直後に追加すると畳み込みのオートチューニングが効きます。
  ```python
  torch.backends.cudnn.benchmark = True  # 一度最適なカーネルを探索
  ```
- FP32 で十分に安定しますが、`torch.set_float32_matmul_precision("high")` を起動時に実行すると Ada 世代で 5-10% スループットが伸びます。

### 3.2 自己対戦の CPU 並列度
- `SelfPlayManager.generate` は Python スレッド (`ThreadPoolExecutor`) で env + MCTS を複製しています（`yonokuni/selfplay/self_play.py:159-197`）。GIL 影響があるので、`self_play_workers = 12` を超えるとスケールが頭打ちになります。状況に応じて 8〜14 の範囲で調整し、`nvidia-smi` で GPU 待機時間が増えたらワーカー数を増やします。
- `episodes_per_iteration` を 64 → 96 に増やしたのは、CPU が生成するサンプルより GPU の学習が先に終わって待機する時間を隠すためです。CPU ボトルネックを感じたらここを 64 付近まで戻してください。

### 3.3 MCTS の粒度
- `mcts.num_simulations` を上げると 1 手あたりに `YonokuniNet` 推論を複数回呼びます。RTX 5090 では 160〜192 が妥当範囲です。CPU が追いつかない場合は 128 まで下げると 20% ほど高速化します。
- `dirichlet_alpha` は既存の長期設定 (`configs/self_play_long.yaml`) と合わせ 0.03 に固定しています。局面探索の多様性を確保しつつシミュレーション数を削減できます。

### 3.4 リプレイバッファ
- 現状 `buffer_capacity` は `SelfPlayTrainerConfig` のデフォルト 500,000 が使われます。より大きな経験リプレイを使いたい場合は `scripts/run_iteration.py` で引数を渡す必要があります。例:
  ```python
  config = SelfPlayTrainerConfig(
      episodes_per_iteration=episodes,
      training_steps_per_iteration=train_steps,
      buffer_capacity=cfg.get("buffer_capacity", 500_000),  # ★追加
      ...
  )
  ```
  2M サンプルで約 18 GB メモリを消費するので、64 GB RAM なら許容範囲です。

### 3.5 Mixed Precision / AMP
- `Trainer.train_step`（`yonokuni/training/loop.py:68-99`）は FP32 固定です。RTX 5090 の TensorCore を使うには `torch.cuda.amp.autocast` と `GradScaler` を導入してください:
  ```python
  scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
  with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
      policy_logits, value_pred = self.model(board_tensor, aux_tensor)
      ...
      total_loss = ...
  scaler.scale(total_loss).backward()
  scaler.step(self.optimizer)
  scaler.update()
  ```
  BatchNorm が多いため、`dtype=torch.float16` ではなく AMP の自動キャストを推奨します。

### 3.6 `torch.compile` (PyTorch 2.2)
- `YonokuniNet` は静的制御フローなので `torch.compile(self.model, mode="reduce-overhead")` を `SelfPlayTrainer.__init__` で一度だけ呼ぶと、学習側の forward/backward が 5〜8% 改善します。ただし初回コンパイルで 2〜3 分かかるため、長期学習ジョブに限定してください。

---

## 4. Windows 固有の運用 Tips

1. **電源プラン**: `powercfg /SETACTIVE SCHEME_MIN` (高パフォーマンス) で CPU 周波数ドロップを防止。
2. **GPU メモリ先読み**: `setx PYTORCH_CUDA_ALLOC_CONF "max_split_size_mb:256,garbage_collection_threshold:0.6"` により断片化を低減。
3. **ログ保存先**: `logs/` と `checkpoints/` は NVMe ローカルに置いてください。リモートドライブ経由だと 200 iteration で数十 GB を転送するため大きな待ちが発生します。
4. **監視**: `nvidia-smi dmon -s pucm` で GPU Util/Cuda/Memory の推移をターミナルに流し、CPU 側は「リソース モニター」または `Get-Counter '\Processor(_Total)\% Processor Time'`.
5. **TensorBoard**: `.\.venv\Scripts\tensorboard.exe --logdir logs/windows_gpu --bind_all` のように `--bind_all` を付ければ RDP 越しでもブラウザ観測できます。

---

## 5. 想定トラブルと切り分け

| 症状 | 確認ポイント |
| --- | --- |
| GPU 利用率が 0% のまま | `training.device` が `cuda` か (`configs/self_play_windows_gpu.yaml`)、`torch.cuda.is_available()` の結果を確認 |
| CUDA out-of-memory | `training.batch_size` を 384 以下へ、`mcts.num_simulations` を 128 へ、`PYTORCH_CUDA_ALLOC_CONF` で分割サイズを縮小 |
| CPU 100% で GPU が待機 | `self_play_workers` を減らす、`episodes_per_iteration` を 64 に戻す |
| 途中で停止 / SIGTERM | `checkpoints/windows_gpu/` がネットワーク越しだと I/O が追いつかないことがあるためローカルに変更 |

---

### 今後検討できる拡張
1. **自己対戦のプロセス並列**: `ThreadPoolExecutor` では GIL の影響があるので、`multiprocessing` もしくは `ray` で env を分散すると 24C をフルに使えます。
2. **経験共有ストレージ**: GPU ノードとは別に self-play 専用ノードを立て、`ReplayBuffer` を gRPC で受信する構成にすると GPU 稼働率がさらに上がります。
3. **モデルサイズの自動切り替え**: `YonokuniNetConfig` を Hydra から注入できるよう `scripts/run_iteration.py` を拡張すると、Windows GPU と Mac CPU で別ネットワークを選択しやすくなります。

このガイドに従えば Windows + RTX 5090 環境でも Mac 以上の学習スループットを確保できます。必要に応じてパラメータを調整し、`tensorboard` で収束速度を観測してください。
