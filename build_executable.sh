#!/usr/bin/env bash
set -euo pipefail

required_files=(
  "pipeline_optimization_app.py"
  "pipeline_desktop_app.py"
  "logo.png"
  "secrets.toml"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 1
  fi
done

if ! ls *.csv >/dev/null 2>&1; then
  echo "Missing required CSV files (*.csv)." >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller pywebview

DATA_ARGS=(
  --add-data "pipeline_optimization_app.py:."
  --add-data "logo.png:."
  --add-data "secrets.toml:."
  --add-data "*.csv:."
)

pyinstaller --clean --onefile --name pipeline_optimizer \
  "${DATA_ARGS[@]}" \
  pipeline_desktop_app.py

python verify_bundle.py dist/pipeline_optimizer \
  --require pipeline_optimization_app.py \
  --require logo.png \
  --require secrets.toml \
  --write dist/bundle_manifest.txt

echo "Executable built at dist/pipeline_optimizer"
