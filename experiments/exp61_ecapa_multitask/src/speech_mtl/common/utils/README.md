# EXP 6.1 common utilities

This package contains the operational utilities shared by the training,
evaluation, analysis, and queue entry points.

## Configuration

- safe YAML loading
- recursive override merging
- required nested-key lookup
- no mutation of source configuration dictionaries

## Atomic files

- atomic UTF-8 text writes
- atomic JSON writes
- fsync before replacement
- SHA-256 file hashing

## Reproducibility

`seed_everything()` seeds:

- Python
- NumPy
- PyTorch CPU
- all visible CUDA devices

Deterministic algorithms remain optional because fully deterministic kernels
can reduce training speed or reject some operations. The chosen setting must
be written to run metadata.

## GPU policy

EXP 6.1 is intended for physical GPU 1 on WTC9.

The correct launch sequence is:

1. query physical GPU 1 with `nvidia-smi`
2. reject the launch when it is occupied or over the configured thresholds
3. set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
4. set `CUDA_VISIBLE_DEVICES=1`
5. import PyTorch
6. require exactly one visible CUDA device
7. use process-local `cuda:0`

CPU fallback is explicitly rejected by `verify_single_visible_cuda_device()`.

The validation script only queries `nvidia-smi`; it does not initialize a
CUDA context and does not reserve either GPU.

## Runtime evidence

Runtime metadata records the host, Python executable, package versions, and
the relevant CUDA/thread environment variables without initializing CUDA.
