#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ENV="$PROJECT_ROOT/.env.local"

if [[ -f "$LOCAL_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LOCAL_ENV"
  set +a
fi

RUNTIME_HOME="$HOME/Library/Application Support/Neural-State-OS"
VENV="${NEURAL_STATE_OS_VENV:-$RUNTIME_HOME/venv-mlx}"
MODEL="${NEURAL_STATE_OS_12B_MODEL:-/Volumes/My Passport/Codex/models/gemma-4-12B-it-4bit-mlx}"
HOST="${NEURAL_STATE_OS_HOST:-127.0.0.1}"
PORT="${NEURAL_STATE_OS_PORT:-8768}"
WRITER="$PROJECT_ROOT/artifacts/gemma4-12b/writer/direct44_raw_three_layer_writer.npz"
READER="$PROJECT_ROOT/artifacts/gemma4-12b/reader/direct44_postwriter_live_affect_reader.npz"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "MLX environment is missing. Run scripts/setup_mlx_macos.sh first." >&2
  exit 1
fi
if [[ ! -f "$MODEL/config.json" || ! -f "$MODEL/model.safetensors.index.json" ]]; then
  echo "Gemma 4 12B MLX checkpoint not found at: $MODEL" >&2
  echo "Set NEURAL_STATE_OS_12B_MODEL in .env.local to the checkpoint directory." >&2
  exit 1
fi
for artifact in "$WRITER" "$READER"; do
  if [[ ! -f "$artifact" ]]; then
    echo "Required artifact is missing: $artifact" >&2
    exit 1
  fi
done

cd "$PROJECT_ROOT"
exec "$VENV/bin/python" -u demo/server_mlx.py \
  --model "$MODEL" \
  --writer "$WRITER" \
  --reader "$READER" \
  --host "$HOST" \
  --port "$PORT" \
  --max-tokens "${NEURAL_STATE_OS_MAX_TOKENS:-512}" \
  --thinking-max-tokens "${NEURAL_STATE_OS_THINKING_MAX_TOKENS:-1024}"
