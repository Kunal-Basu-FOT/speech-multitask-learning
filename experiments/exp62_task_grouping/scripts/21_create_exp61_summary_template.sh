#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/reports/exp62_reduced/exp61_matched_summary.json"
mkdir -p "$(dirname "$TARGET")"
cat > "$TARGET" <<'JSON'
{
  "one_bit": {
    "lid_accuracy": null,
    "lid_macro_f1": null,
    "dialect_language_macro_accuracy": null,
    "dialect_language_macro_f1": null
  },
  "sixteen_bit": {
    "lid_accuracy": null,
    "lid_macro_f1": null,
    "dialect_language_macro_accuracy": null,
    "dialect_language_macro_f1": null
  }
}
JSON
echo "Template created: $TARGET"
echo "Fill it only after EXP 6.1 matched test metrics exist."
