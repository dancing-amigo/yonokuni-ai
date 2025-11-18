# yonokuni-ai トレーニング実行マニュアル（Windows / PowerShell）

## 1. 前提

- プロジェクトディレクトリ  
  `C:\Users\jphot\Desktop\yonokuni-ai`
- 仮想環境ディレクトリ  
  `C:\Users\jphot\Desktop\yonokuni-ai\.venv`
- 使用シェル  
  Windows PowerShell（リモートデスクトップ経由で利用）

以下、すべて **PowerShell** での操作を前提とする。

---

## 2. 実行ポリシーの緩和（権限設定）

仮想環境の `Activate.ps1` を実行するため、**ユーザー単位でスクリプト実行を許可**する。

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
````

* 一度設定すれば、以降は不要。
* 管理者 PowerShell で実行しておくと確実。

---

## 3. 仮想環境の作成と有効化

### 3.1 仮想環境を作成（初回のみ）

```powershell
cd C:\Users\jphot\Desktop\yonokuni-ai
python -m venv .venv
```

### 3.2 仮想環境を有効化

```powershell
cd C:\Users\jphot\Desktop\yonokuni-ai
.\.venv\Scripts\Activate.ps1
```

プロンプトが以下のようになれば有効化できている。

```text
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai>
```

### 3.3 仮想環境の無効化

```powershell
deactivate
```

---

## 4. 依存パッケージのインストール（初回 or 更新時）

仮想環境を有効化した状態で実行：

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> python -m pip install --upgrade pip
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> pip install -r requirements.txt
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> pip install -e .
```

---

## 5. トレーニングの実行（バックグラウンド、ログ出力あり）

### 5.1 共通の変数設定

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $wd    = "C:\Users\jphot\Desktop\yonokuni-ai"
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $out   = Join-Path $wd "train.out"   # 標準出力ログ
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $err   = Join-Path $wd "train.err"   # 標準エラーログ
```

### 5.2 GPU 用設定ファイルでの実行例（self_play_windows_gpu.yaml）

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $config = "configs\self_play_windows_gpu.yaml"

(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $proc = Start-Process `
  -FilePath "$wd\.venv\Scripts\python.exe" `
  -ArgumentList "-u",
                "scripts\run_iteration.py",
                "--config",$config,
                "--iterations","200",
                "--checkpoint-dir","checkpoints\windows_gpu",
                "--checkpoint-interval","10" `
  -WorkingDirectory $wd `
  -RedirectStandardOutput $out `
  -RedirectStandardError  $err `
  -PassThru
```

* `-u`
  Python のバッファリングを無効化（ログがすぐファイルに出る）。
* `--iterations 200`
  トレーニング反復回数。
* `--checkpoint-dir checkpoints\windows_gpu`
  チェックポイント保存先（相対パス）。
  → `C:\Users\jphot\Desktop\yonokuni-ai\checkpoints\windows_gpu\checkpoint_*.pt`
* `--checkpoint-interval 10`
  10 iteration ごとにチェックポイントを保存。
* `$proc`
  `Start-Process` から返されるプロセスオブジェクト。
  `$proc.Id` で PID を取得できる。

### 5.3 CPU 用の軽量実行例（確認用）

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $config = "configs\self_play.yaml"

(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $proc = Start-Process `
  -FilePath "$wd\.venv\Scripts\python.exe" `
  -ArgumentList "-u",
                "scripts\run_iteration.py",
                "--config",$config,
                "--iterations","5",
                "--checkpoint-dir","checkpoints\windows_cpu",
                "--checkpoint-interval","1" `
  -WorkingDirectory $wd `
  -RedirectStandardOutput $out `
  -RedirectStandardError  $err `
  -PassThru
```

---

## 6. ログの確認方法

### 6.1 標準出力ログ（train.out）

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> Get-Content $out -Wait
```

### 6.2 標準エラーログ（train.err）

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> Get-Content $err -Wait
```

* `tqdm` の進捗バーや、多くのログが `stderr` 側に出る可能性があるため、
  実際には `train.err` に有用な情報が多く含まれる。
* `-Wait` を付けると、`tail -f` のようにリアルタイムで追尾できる。
  終了するときは `Ctrl + C`。

---

## 7. プロセスの確認と停止

### 7.1 実行中の python プロセス確認

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> Get-Process python
```

### 7.2 Start-Process からの PID 取得

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $proc.Id
```

必要なら PID をファイルに保存しておくことも可能：

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $proc.Id | Out-File (Join-Path $wd "train.pid")
```

### 7.3 プロセスの停止

PID がわかっている場合：

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> Stop-Process -Id 26856 -Force
```

または、PID ファイルを使う場合：

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> $pid = Get-Content (Join-Path $wd "train.pid")
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> Stop-Process -Id $pid -Force
```

停止後に確認：

```powershell
(.venv) PS C:\Users\jphot\Desktop\yonokuni-ai> Get-Process python
# → 何も表示されなければ python プロセスは残っていない
```

---

## 8. リモートデスクトップとの関係

* `Start-Process` で起動したプロセスは、**PowerShell セッションと独立した OS プロセス**として動作する。
* **RDP セッションを切断しても（サインアウトしなければ）プロセスは継続**する。
* Windows から完全にサインアウト／再起動すると当然終了するので注意。

---

## 9. 典型的なワークフロー（まとめ）

1. PowerShell 起動
2. プロジェクトディレクトリへ移動

   ```powershell
   cd C:\Users\jphot\Desktop\yonokuni-ai
   ```
3. 仮想環境有効化

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. ログ出力先など変数設定

   ```powershell
   $wd    = "C:\Users\jphot\Desktop\yonokuni-ai"
   $out   = Join-Path $wd "train.out"
   $err   = Join-Path $wd "train.err"
   $config = "configs\self_play_windows_gpu.yaml"
   ```
5. トレーニング開始（バックグラウンド）

   ```powershell
   $proc = Start-Process `
     -FilePath "$wd\.venv\Scripts\python.exe" `
     -ArgumentList "-u",
                   "scripts\run_iteration.py",
                   "--config",$config,
                   "--iterations","200",
                   "--checkpoint-dir","checkpoints\windows_gpu",
                   "--checkpoint-interval","10" `
     -WorkingDirectory $wd `
     -RedirectStandardOutput $out `
     -RedirectStandardError  $err `
     -PassThru
   ```
6. ログ確認

   ```powershell
   Get-Content $out -Wait
   Get-Content $err -Wait
   ```
7. 必要に応じて停止

   ```powershell
   Stop-Process -Id $proc.Id -Force
   ```

この手順どおりに実行すれば、

* 仮想環境を正しく利用しつつ
* リモートデスクトップ切断後もトレーニングを継続させ
* 出力とエラーをログファイルで追跡し
* 必要なタイミングで安全に停止・再開できる。

