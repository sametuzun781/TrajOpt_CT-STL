"""Independent RK4, mode routing, sensitivity and CT-history checks."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ctstl.application import load_application
from ctstl.disc import cal_disc, jit_cal_disc_fcn
from ctstl.jitting import jit_ncvx_ct_fcn_grad
from ctstl.full_scp import linearize_trajectory


def error(a, b):
    pairs = zip(jax.tree.leaves(a), jax.tree.leaves(b))
    return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y))/(1+np.abs(np.asarray(x)))))
               for x, y in pairs)


@pytest.mark.parametrize('interpolation', ['FOH', 'ZOH'])
@pytest.mark.parametrize('time_mode', ['fixed', 'global', 'dilated'])
@pytest.mark.parametrize('ad', ['forward', 'reverse'])
def test_generic_transport(interpolation, time_mode, ad):
    nx, nu, nodes, steps = 4, 2, 5, 4
    def field(x, u, t):
        return jnp.array([x[1]+u[0], -.2*x[0]+jnp.sin(u[1])+.02*t,
                          x[0]*x[1]+u[1], x[2]+jnp.cos(x[3])*u[0]])
    sizes = np.cumsum([0, nx, nx*nx, nx*nu, nx*nu, nx, nx])
    derivative = jax.jacfwd if ad == 'forward' else jax.jacrev
    p = dict(n_x=nx, n_u=nu, K=nodes, rk4_steps=steps, t_scp=.2,
             f_func=field, A_func=derivative(field, 0), B_func=derivative(field, 1),
             inp_param=interpolation, free_final_time=time_mode!='fixed',
             time_dil=time_mode=='dilated', dynamics_autodiff=ad)
    p.update({f'i{i}': int(v) for i, v in enumerate(sizes)})
    rng = np.random.default_rng(12)
    X = jnp.asarray(rng.normal(size=(nx, nodes))*.1)
    U = jnp.asarray(rng.normal(size=(nu, nodes))*.1)
    S = jnp.full((1, nodes-1), .2) if p['time_dil'] else jnp.array(.8)
    reference = jax.jit(lambda X, U, S: cal_disc(X, U, S, p))(X, U, S)
    actual = jit_cal_disc_fcn(p, compile=False)(X, U, S)
    assert np.isfinite(actual).all()
    assert error(reference, actual) < 1e-12


@pytest.mark.parametrize('ad', ['forward', 'reverse'])
def test_quad_modes_and_changed_parameters(ad):
    ns, p, _ = load_application(autodiff=ad)
    p = dict(p, I_x=p['I_x']*1.2, T_max=p['T_max']*.95, rk4_steps=7)
    p = ns['dynamics'](p)
    args = tuple(jnp.asarray(p[k]) for k in ('X_last', 'U_last', 'S_last'))
    reference = jit_cal_disc_fcn(dict(p, discretization_mode='reference'), compile=False)(*args)
    for mode in ('jax', 'native', 'fast'):
        f = jit_cal_disc_fcn(dict(p, discretization_mode=mode), compile=False)
        actual = f(*args)
        assert np.isfinite(actual).all()
        assert error(reference, actual) < 1e-9, mode
        assert f.discretization_backend == ('jax' if mode == 'jax' else 'native')
        if mode != 'jax':
            assert error(actual[:, -1], f.icore_terminal_discretizer(*args)) < 1e-12


def test_native_fallback_and_strict_mode(monkeypatch):
    _, p, _ = load_application()
    from native_discretization.native_quad import eligibility
    custom = dict(p, f_func=lambda x, u, t: p['f_func'](x, u, t)*1.001)
    assert not eligibility(custom)[0]
    assert jit_cal_disc_fcn(custom, compile=False).discretization_backend == 'jax'
    with pytest.raises(ValueError, match='unavailable'):
        jit_cal_disc_fcn(dict(custom, discretization_mode='native'), compile=False)
    def no_compiler(*args, **kwargs):
        raise RuntimeError('test compiler unavailable')
    monkeypatch.setattr('native_discretization.native_quad.make_native_discretizer', no_compiler)
    with pytest.warns(RuntimeWarning, match='using JAX'):
        assert jit_cal_disc_fcn(p, compile=False).discretization_backend == 'jax'
    with pytest.raises(RuntimeError, match='compiler unavailable'):
        jit_cal_disc_fcn(dict(p, discretization_mode='native'), compile=False)


def test_nonzero_ct_keeps_history_and_contractions():
    _, p, functions = load_application()
    def cost(X, p):
        return jnp.array([jnp.sum(X[2]**2)*.003+jnp.sum(jnp.sin(X[1]))*.002])
    args = tuple(jnp.asarray(p[k]) for k in ('X_last', 'U_last', 'S_last'))
    V = functions['cal_disc_jitted'](*args)
    grad_ref = None
    for mode in ('forward', 'reverse', 'auto'):
        f, g = jit_ncvx_ct_fcn_grad(cost, p, autodiff=mode, compile=False)
        assert not g.icore_zero_continuous_cost
        actual = g(V)
        if grad_ref is None:
            # Direct chain rule, independently using the original block layout.
            nx, nu, k, steps = p['n_x'], p['n_u'], p['K']-1, p['rk4_steps']
            X = V[:, :, :nx].transpose(2, 0, 1)
            gradient = jax.jacrev(lambda X: cost(X, p))(X)
            rows = [cost(X, p)]
            for lo, hi, width in [(p['i1'],p['i2'],nx), (p['i2'],p['i3'],nu),
                                  (p['i3'],p['i4'],nu), (p['i4'],p['i5'],1)]:
                blocks = V[:, :, lo:hi].reshape(k, steps, nx, width)
                rows.append(jnp.stack([jnp.sum(gradient[:, :, i, :, None]*blocks[i].transpose(1,0,2)[None], axis=(1,2))
                                       for i in range(k)], axis=-1).reshape(1,-1))
            grad_ref = tuple(rows)
        assert error(grad_ref, actual) < 1e-12
    custom = dict(functions, ncvx_ct_fcn_jitted=f, ncvx_ct_fcn_grad_jitted=g)
    reference = dict(custom, cal_disc_jitted=jit_cal_disc_fcn(dict(p, discretization_mode='reference'), compile=False))
    assert error(linearize_trajectory(*args, p, reference), linearize_trajectory(*args, p, custom)) < 1e-9
    for callback in (lambda X,p: jnp.array([2.]), lambda X,p: jnp.array([jnp.sum(X[0])])):
        assert not jit_ncvx_ct_fcn_grad(callback, p, compile=False)[1].icore_zero_continuous_cost
