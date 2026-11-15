"""Paired, synchronized RK4 timings; compilation is excluded."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from ctstl.application import load_application
from ctstl.disc import jit_cal_disc_fcn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Optional NPZ with X, U, S arrays.')
    parser.add_argument('--repetitions', type=int, default=40)
    parser.add_argument('--output', type=Path, default=Path('runs/discretization-benchmark.json'))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    _, p, _ = load_application()
    if args.snapshot:
        with np.load(args.snapshot) as arrays:
            trajectory = tuple(jnp.asarray(arrays[k]) for k in ('X', 'U', 'S'))
    else:
        trajectory = tuple(jnp.asarray(p[k]) for k in ('X_last', 'U_last', 'S_last'))
    methods = {mode: jit_cal_disc_fcn(dict(p, discretization_mode=mode), compile=False)
               .lower(*trajectory).compile() for mode in ('reference', 'jax', 'native')}
    outputs = {mode: np.asarray(f(*trajectory)) for mode, f in methods.items()}
    errors = {mode: float(np.max(np.abs(value-outputs['reference'])/(1+np.abs(outputs['reference']))))
              for mode, value in outputs.items()}
    assert all(np.isfinite(v).all() for v in outputs.values()) and max(errors.values()) < 1e-9
    for f in methods.values():
        for _ in range(10):
            jax.block_until_ready(f(*trajectory))
    samples = {mode: [] for mode in methods}
    rng = np.random.default_rng(71)
    for _ in range(args.repetitions):
        for mode in rng.permutation(list(methods)):
            start = perf_counter()
            jax.block_until_ready(methods[mode](*trajectory))
            samples[mode].append(1000*(perf_counter()-start))
    means = {mode: float(np.mean(v)) for mode, v in samples.items()}
    result = dict(platform=platform.platform(), jax=jax.__version__, snapshot=str(args.snapshot),
                  repetitions=args.repetitions, mean_ms=means, normalized_errors=errors,
                  native_speedup={m: means[m]/means['native'] for m in ('reference', 'jax')}, samples_ms=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples_ms'}, indent=2))


if __name__ == '__main__':
    main()
