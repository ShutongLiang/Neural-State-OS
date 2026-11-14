# Model setup

The repository does not redistribute Gemma weights. Supply a local checkpoint
that matches one of the released artifact banks.

## Gemma 4 12B IT — MLX

Expected contract:

- dense 48-layer text decoder;
- hidden width 3,840;
- MLX-compatible 4-bit checkpoint;
- Apple silicon host.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-mlx.txt
.venv/bin/python demo/server_mlx.py \
  --model /absolute/path/to/gemma-4-12b-it-4bit-mlx \
  --writer artifacts/gemma4-12b/writer/direct44_raw_three_layer_writer.npz \
  --reader artifacts/gemma4-12b/reader/direct44_postwriter_live_affect_reader.npz
```

## Gemma 4 31B IT — CUDA

Expected contract:

- dense 60-layer text decoder;
- hidden width 5,376;
- BF16 checkpoint compatible with Hugging Face Transformers;
- CUDA-enabled PyTorch and sufficient GPU memory.

```bash
python3 -m venv .venv
# Install the CUDA-specific torch build first.
.venv/bin/pip install -r requirements-cuda.txt
.venv/bin/python demo/server_cuda.py \
  --model /absolute/path/to/gemma-4-31b-it-bf16 \
  --writer artifacts/gemma4-31b/writer/direct44_raw_three_layer_writer.npz \
  --reader artifacts/gemma4-31b/reader/direct44_postwriter_live_affect_reader.npz
```

## Integrity checks

The checks load only the small NumPy artifacts:

```bash
python scripts/verify_direct44_bundle.py
python scripts/verify_direct44_31b_bundle.py
```

Never mix the 12B and 31B artifacts. A dimension mismatch should be treated as
a configuration error, not coerced or projected.

## Model directory policy

Keep checkpoints outside the Git repository. `models/` and common model-file
extensions are ignored by default. If you use a removable drive, pass the
absolute model path at launch rather than creating a tracked copy.
