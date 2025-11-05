# 推論・エクスポート

## 推論ランナー (`yonokuni/models/inference.py`)

- `InferenceRunner`
  - mixed precision (`torch.cuda.amp`) とバッチ推論に対応。
  - `InferenceConfig` でデバイス・dtype・AMP 有無を設定。
- 使い方:
  ```python
  runner = InferenceRunner(model)
  policy, value = runner.run(board_tensor, aux_tensor)
  ```

## エクスポート (`yonokuni/models/export.py`)

- `save_checkpoint`
  - モデル・オプティマイザ状態に加え、`ModelMetadata` を保存可能。
- `export_torchscript`
  - サンプル入力を元に `torch.jit.trace` を実施。
- `export_onnx`
  - ダイナミックバッチ対応で ONNX ファイルを出力。
- `load_checkpoint`
  - CPU/GPU いずれの環境でもロード可能。

## メタデータ (`ModelMetadata`)

- `version`, `trained_at`, `config`, `notes` を保持。
- checkpoint 保存時にテキスト管理し、モデルの文脈を追跡。

## 推奨ワークフロー

1. 学習済みモデルを `save_checkpoint` で保存（メタデータ付）。
2. 推論用途には `InferenceRunner`＋混合精度で高速化。
3. 配布・デプロイには `export_torchscript` もしくは `export_onnx` を利用。
