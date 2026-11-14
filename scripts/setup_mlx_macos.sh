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
PIP_INDEX="${NEURAL_STATE_OS_PIP_INDEX_URL:-https://pypi.org/simple}"

python_is_compatible() {
  "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1
}

PYTHON="${NEURAL_STATE_OS_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in "$HOME/miniconda3/bin/python" "$(command -v python3 2>/dev/null || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]] && python_is_compatible "$candidate"; then
      PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]] || ! python_is_compatible "$PYTHON"; then
  echo "Neural State OS requires Python 3.10 or newer for the MLX runtime." >&2
  echo "Set NEURAL_STATE_OS_PYTHON in .env.local to a compatible interpreter." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  mkdir -p "$(dirname "$VENV")"
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --index-url "$PIP_INDEX" --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --index-url "$PIP_INDEX" -r "$PROJECT_ROOT/requirements-mlx.txt"

"$VENV/bin/python" - <<'PY'
from importlib.metadata import version
import mlx  # noqa: F401
import mlx_vlm  # noqa: F401
import numpy
print(
    "MLX runtime ready: "
    f"mlx={version('mlx')}, mlx-vlm={version('mlx-vlm')}, numpy={numpy.__version__}"
)
PY
