"""Load the same setup cells used by the five example notebooks."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ('quad_flight', 'di_always_stc_aug', 'di_eventually_aug', 'di_until', 'di_until_aug')
BACKENDS = ('cvxpy_qoco', 'qoco_native', 'jax_icore', 'icore_c', 'icore_cpp')


def load_application(backend='icore_cpp', autodiff='forward', discretization='fast',
                     example='quad_flight'):
    """Return notebook namespace, parameters and functions without solving."""
    if example not in EXAMPLES or backend not in BACKENDS:
        raise ValueError('Unknown example or solver backend')
    filename = ROOT / f'run_{example}.ipynb'
    notebook = json.loads(filename.read_text())
    namespace = {'__name__': '__trajectory_example__'}
    for index, cell in enumerate(notebook['cells'][:12]):
        if cell['cell_type'] != 'code':
            continue
        source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                           if not line.lstrip().startswith('%'))
        exec(compile(source, f'{filename.name}:cell{index}', 'exec'), namespace)
        if index == 2:
            namespace.update(SOLVER_BACKEND=backend, DYNAMICS_AD=autodiff,
                             DISCRETIZATION_MODE=discretization, EAGER_COMPILE=False)
    return namespace, namespace['params_sc'], namespace['fcn_dict']


def prepare_solver(params, functions, backend='icore_cpp'):
    if backend == 'cvxpy_qoco':
        from solvers.cvx import parse_convex_problem
        return parse_convex_problem(params, functions)
    from solvers.ipm_adapter import OptionalConvexProblem
    problem = OptionalConvexProblem(params, functions, backend,
                                    tolerance=params['qoco_options']['abstol'],
                                    qoco_options=params['qoco_options'])
    problem.prepare(functions)
    return problem
