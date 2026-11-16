# Convex solver backends

`icore_c`, `icore_cpp`, and `jax_icore` use the separately installed ICORE
0.20.0+a63ebffb0057 public Solver/ChainQP interface. Install it using
`python tools/install_icore.py` in the same environment as the notebook.
`qoco_native` uses the pinned original QOCO source included in that package,
so its first native build needs no additional download.
`cvxpy_qoco` remains an optional modeling comparator.
Compilation is excluded from solve timing. No warm-start solution is injected.
