#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer
EXP61=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
MARKER="$ROOT/runtime/PACKAGE_INSTALL_VALIDATED.json"

printf '%s\n' "============================================================" " EXP 6.5 + 6.8 — PACKAGE INSTALL VALIDATION" "============================================================"
whoami
hostname
date --iso-8601=seconds

[[ "$(whoami)" == "wtc9" ]] || { echo "ERROR: expected user wtc9"; exit 1; }
[[ "$(hostname)" == "wtc9-server" ]] || { echo "ERROR: expected host wtc9-server"; exit 1; }
[[ -d "$ROOT" ]] || { echo "ERROR: combined root missing: $ROOT"; exit 1; }
[[ -d "$EXP61/src" ]] || { echo "ERROR: EXP 6.1 source missing: $EXP61/src"; exit 1; }
[[ -x "$PYTHON" ]] || { echo "ERROR: Python missing: $PYTHON"; exit 1; }

mkdir -p "$ROOT/runtime/logs" "$ROOT/results" "$ROOT/reports"

export PYTHONPATH="$ROOT/src:$EXP61/src"

missing="$($PYTHON - <<'PY'
missing=[]
for name in ('sklearn','umap'):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
print(' '.join(missing))
PY
)"

if [[ -n "$missing" ]]; then
  echo "Installing missing analysis dependencies: $missing"
  "$PYTHON" -m pip install --upgrade -r "$ROOT/requirements-analysis.txt"
fi

"$PYTHON" - <<'PY'
import numpy, pandas, torch, matplotlib, sklearn, umap
print('NumPy:', numpy.__version__)
print('Pandas:', pandas.__version__)
print('PyTorch:', torch.__version__)
print('Matplotlib:', matplotlib.__version__)
print('scikit-learn:', sklearn.__version__)
print('umap-learn:', umap.__version__)
PY

find "$ROOT/src/exp65_68" -type f -name '*.py' -print0 | xargs -0 -n1 "$PYTHON" -m py_compile
"$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
"$PYTHON" -m exp65_68.cli --version

ROOT="$ROOT" "$PYTHON" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
root=Path(os.environ['ROOT'])
path=root/'runtime/PACKAGE_INSTALL_VALIDATED.json'
path.write_text(json.dumps({
  'status':'PACKAGE_INSTALL_VALIDATED',
  'created_utc':datetime.now(timezone.utc).isoformat(),
  'package_version':'1.0.0'
}, indent=2, sort_keys=True)+'\n', encoding='utf-8')
print(path)
PY

echo "EXP65_EXP68_PACKAGE_INSTALL_PASS"
