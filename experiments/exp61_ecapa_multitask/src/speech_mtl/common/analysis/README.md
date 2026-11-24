# EXP 6.1 common analysis package

This package creates the compact report artifact set agreed for EXP 6.1.

## Main figures

1. One-bit train/development accuracy
2. One-bit train/development loss
3. One-bit composite confusion matrices
4. Sixteen-bit train/development accuracy
5. Sixteen-bit train/development loss
6. Sixteen-bit composite confusion matrices
7. Matched and cross-precision summary

## Appendix figures

- 1-bit-trained model evaluated on 16-bit cache: composite confusion matrices
- 16-bit-trained model evaluated on 1-bit cache: composite confusion matrices

Each composite confusion figure contains:

- one 8-class language matrix
- one 2-class gender matrix
- eight language-specific dialect matrices

Matrices use row-normalized percentages from 0 to 100. Raw counts remain
available in evaluation JSON and can be exported separately.

## Tables

1. Experimental configuration
2. Matched 1-bit and 16-bit results with deltas
3. Per-language dialect results
4. Matched and cross-precision matrix
5. Runtime and throughput

## Plotting policy

- Matplotlib only
- no seaborn
- 300 DPI PNG
- consistent 0–1 score axes
- phase boundaries after epochs 5 and 35
- absolute raw values preserved in CSV/JSON
