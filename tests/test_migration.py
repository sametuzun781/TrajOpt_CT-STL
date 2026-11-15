"""Compare all adapters against the existing CVXPY model at the same iterate."""
import gc
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ctstl.application import EXAMPLES, load_application
from solvers.cvx import parse_convex_problem, solve_parsed_problem
from ctstl.full_scp import linearize_trajectory, evaluate_costs
from solvers.ipm_adapter import OptionalConvexProblem
from ctstl.scp import set_parameters, scp_noncvx_cost
from tests.audit_chain import audit_chain


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize('example', EXAMPLES)
def test_five_solvers_match_original_model(example):
    _, p, functions = load_application(example=example)
    X, U, S = (jnp.asarray(p[k]) for k in ('X_last', 'U_last', 'S_last'))
    values = jax.jit(lambda X, U, S: linearize_trajectory(X, U, S, p, functions))(X, U, S)
    values['w_tr'] = jnp.asarray(p['w_ptr'])
    original = parse_convex_problem(p, functions)
    set_parameters(original, **jax.device_get(values))
    solve_parsed_problem(original, p)
    reference = float(original.value)
    for backend in ('qoco_native', 'jax_icore', 'icore_c', 'icore_cpp'):
        problem = OptionalConvexProblem(p, functions, backend, qoco_options=p['qoco_options'])
        try:
            set_parameters(problem, **values)
            _, x, u, s, nu = problem.solve()
            x, u, s, nu = jax.device_get((x, u, s, nu))
            if backend != 'qoco_native':
                audit = audit_chain(problem.solver.last_model, problem.solver.last_result)
                assert audit['accepted'], (example, backend, audit)
            candidate = dict(X=x, U=u, nu=nu, dx=x-np.asarray(X), du=u-np.asarray(U))
            if p['free_final_time']:
                candidate.update(S=s, ds=s-np.asarray(S))
            for key, value in candidate.items():
                original.var_dict[key].value = value
            objective = float(original.objective.value)
            violation = max(float(np.max(c.violation(), initial=0.)) for c in original.constraints)
            assert violation < 1e-7, (example, backend, violation)
            assert abs(objective-reference)/(1+abs(reference)) < 1e-7, (example, backend, objective, reference)
        finally:
            problem.close()


@pytest.mark.parametrize('example', EXAMPLES)
def test_nonlinear_cost_matches_original(example):
    _, p, f = load_application(example=example)
    rng = np.random.default_rng(37)
    X, U, S = (jnp.asarray(p[k]) for k in ('X_last', 'U_last', 'S_last'))
    values = linearize_trajectory(X, U, S, p, f)
    x = X + jnp.asarray(rng.normal(size=X.shape)*1e-5)
    x = x.at[p['n_dyn']:].set(X[p['n_dyn']:] + jnp.abs(x[p['n_dyn']:]-X[p['n_dyn']:]))
    u = U + jnp.asarray(rng.normal(size=U.shape)*1e-5)
    s = S * 1.001
    nu = np.zeros((p['n_x'], p['K']-1))
    continuous = f['int_mult_jitted'](x, u, s)
    legacy = scp_noncvx_cost(
        X_new=np.asarray(x), U_new=np.asarray(u), S_new=np.asarray(s),
        nl_nu_new=np.asarray(x[:, 1:]-continuous[:, :, -1]),
        ncvx_dt_cost=np.asarray(f['ncvx_dt_fcn_jitted'](x)),
        ncvx_ct_cost=np.asarray(f['ncvx_ct_fcn_jitted'](continuous)),
        ncvx_comp_cost=np.asarray(f['ncvx_comp_fcn_jitted'](x)),
        params=p, fcn_dict=f, cost_dict={})
    model = f['cvx_cons_fcn'].__module__.split('.')[1]
    actual = evaluate_costs(x, u, s, nu, values, 0., p, f, model)
    assert np.isfinite(actual['nlc']) and np.isfinite(legacy['nlc'][-1])
    np.testing.assert_allclose(actual['nlc'], legacy['nlc'][-1], rtol=1e-12, atol=1e-10)


def test_notebooks_are_clean():
    root = Path(__file__).resolve().parents[1]
    for path in root.glob('run_*.ipynb'):
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                assert cell['outputs'] == [], path.name
                assert cell['execution_count'] is None, path.name
