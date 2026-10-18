#!/usr/bin/env bash
# Start the Serving_Endpoint (the single documented start command): bash scripts/serve.sh [f16|q4]   (default q4)
# Reads config/serve.yaml (or $FTS_SERVE_CONFIG). When require_api_key is true and the key variable is unset
# or empty, it exits 1 naming the variable before starting anything. Otherwise it starts or recreates only
# the fts Compose project's serve container and waits until /health returns 200.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
config="${FTS_SERVE_CONFIG:-config/serve.yaml}"
setting() { sed -n "s/^$1:[[:space:]]*//p" "$config" | head -1; }

variant="${1:-q4}"
case "$variant" in
  f16) model="$(sed -n 's/^  f16:[[:space:]]*//p' "$config" | head -1)" ;;
  q4) model="$(sed -n 's/^  q4:[[:space:]]*//p' "$config" | head -1)" ;;
  *) echo "usage: bash scripts/serve.sh [f16|q4]" >&2; exit 2 ;;
esac

files=(-f docker-compose.yml)
if [ "$(setting require_api_key)" = "true" ]; then
  key_env="$(setting api_key_env)"
  if [ -z "${!key_env:-}" ]; then
    echo "error: required environment variable $key_env is unset or empty; the Serving_Endpoint was not started" >&2
    exit 1
  fi
  # Compose reads the key from the environment at start-up; only the variable's name is written here.
  mkdir -p runs
  printf 'services:\n  serve:\n    environment:\n      LLAMA_API_KEY: ${%s}\n' "$key_env" > runs/serve.api-key.yml
  files+=(-f runs/serve.api-key.yml)
fi

models_dir="${FTS_ARTIFACTS:-$HOME/fts-artifacts}/models"
if [ ! -f "$models_dir/$model" ]; then
  echo "error: $models_dir/$model is missing; run python -m fts.quantize first" >&2
  exit 1
fi

FTS_SERVE_MODEL="$model" docker compose "${files[@]}" up -d --force-recreate serve
health="$(setting health_url)"
for _ in $(seq "${FTS_SERVE_WAIT_S:-300}"); do
  if curl -sf "$health" > /dev/null; then
    echo "Serving_Endpoint ready: $model at $(setting base_url)"
    exit 0
  fi
  sleep 1
done
echo "error: $health did not return 200 within ${FTS_SERVE_WAIT_S:-300} s" >&2
docker compose "${files[@]}" logs --tail 50 serve >&2 || true
exit 1
