"""Run an example without Jupyter; --iterations limits a smoke run."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from application import BACKENDS, EXAMPLES, load_application, prepare_solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=EXAMPLES, default='quad_flight')
    parser.add_argument('--backend', choices=BACKENDS, default='icore_cpp')
    parser.add_argument('--ad', choices=('forward', 'reverse'), default='forward')
    parser.add_argument('--discretization', choices=('fast', 'jax', 'reference', 'native'), default='fast')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--audit', action='store_true', help='Check every ICORE solve against the original ChainQP KKT equations.')
    parser.add_argument('--output', type=Path, help='Optional JSON run summary and NPZ trajectory.')
    args = parser.parse_args()
    if args.iterations is not None and args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.audit and args.backend not in ('icore_c', 'icore_cpp', 'jax_icore'):
        parser.error('--audit requires an ICORE backend')
    _, params, functions = load_application(args.backend, args.ad, args.discretization, args.example)
    if args.iterations is not None:
        params['ite'] = args.iterations
    audits = []
    started = perf_counter()
    problem = prepare_solver(params, functions, args.backend)
    preparation = perf_counter() - started
    if args.audit:
        from tests.audit_chain import audit_chain
        solve = problem.solver.solve
        def checked(qp):
            result = solve(qp)
            audit = audit_chain(problem.solver.last_model, problem.solver.last_result,
                                tolerance=problem.solver.tolerance)
            audits.append(audit)
            if not audit['accepted']:
                raise RuntimeError(f'KKT check failed: {audit}')
            return result
        problem.solver.solve = checked
    try:
        from scp import prox_convex
        result = prox_convex(params, problem, functions)
        summary = dict(example=args.example, backend=args.backend, dynamics_ad=args.ad,
                       discretization=functions['cal_disc_jitted'].discretization_backend,
                       preparation_seconds=preparation, converged=bool(result['converged']),
                       convex_calls=result['convex_solve_count'], final_cost=float(result['nlc'][-1]),
                       discretization_seconds=result['total_discretization_seconds'],
                       convex_seconds=result['total_convex_seconds'],
                       scp_seconds=result['total_scp_seconds'], audits=audits)
        print(json.dumps({k: v for k, v in summary.items() if k != 'audits'}, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + '\n')
            np.savez_compressed(args.output.with_suffix('.npz'),
                                X=result['X_new'], U=result['U_new'], S=result['S_new'])
    finally:
        if hasattr(problem, 'close'):
            problem.close()


if __name__ == '__main__':
    main()
