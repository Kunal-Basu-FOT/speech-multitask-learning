# Validation record

Validation date: 2026-08-08

## Content validation

- 1,436 final archive files including this validation record and `SHA256SUMS`, excluding Git metadata
- 372 JSON files parsed successfully with `jq`
- 159 Python files parsed successfully with the Python AST parser
- 60 shell scripts passed `bash -n`
- 558 CSV files were non-empty and readable with a header line
- 197 PNG files were identified as valid PNG image data
- 134 retained completion-marker JSON files
- 0 symbolic links
- 0 forbidden checkpoint, embedding-array, audio, log, Python-cache, per-example prediction, or raw-gradient-sample files
- Largest retained file: 1,620,522 bytes (`figure_04_umap_lid_dialect_precision.png`)

An initial static-validation wrapper returned a nonzero status only because `sort | head` caused an expected broken-pipe under `pipefail` after all checks had run; the size query was rerun without that wrapper issue. It reported the largest file above and no content-validation errors.

## Source-copy validation

An `rsync --dry-run --checksum-equivalent metadata comparison` using the same exclusion rules reported no pending changes for all six canonical experiment copies. EXP 6.9's top-level and `Notebooks` sources were also compared recursively and showed no differences.

## Unit tests

The captured experiment environment does not include `pytest`, so the retained `unittest`-based suites were executed directly with bytecode/cache generation disabled:

| Experiment | Tests | Result |
|---|---:|---|
| EXP 6.1 | 28 | PASS |
| EXP 6.2 | 5 | PASS |
| EXP 6.5/6.8 | 4 | PASS |
| EXP 6.9 | 8 | PASS |
| Total | 45 | PASS |

EXP 6.6 and EXP 6.7 contain no `test*.py` modules in their supplied repositories. Their source syntax, shell syntax, JSON, CSV, PNG, completion markers, and report artifacts passed the applicable validations above.

## GitHub/offline checks

The final precheck verifies an initialized Git repository with every archive file staged, no configured remote, no tracked forbidden artifacts, no tracked file at or above GitHub's 100 MB hard limit, no obvious credential material, and a valid full-file SHA-256 manifest. Nested source `.gitignore` rules were overridden only after the forbidden-artifact scan so retained reports, metrics, figures, and histories are Git-visible. The ZIP is tested separately with `unzip -t`.
