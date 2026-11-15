"""Native DI dynamics, time conventions, and weighted temporal-logic gradients."""
import gc
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ctstl.application import EXAMPLES, load_application
from ctstl.disc import int_mult, jit_cal_disc_fcn
from ctstl.stl import UNTIL, conjunction, disjunction
from native_discretization.native_di import eligibility, make_native_discretizer


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize('example', EXAMPLES[1:])
def test_native_di_modes_and_parameters(example):
    ns, base, _ = load_application(example=example)
    rng = np.random.default_rng(94)
    X = np.asarray(base['X_last']).copy() + rng.normal(0., .03, base['X_last'].shape)
    X[6:] = np.abs(X[6:]) + .1
    U = np.asarray(base['U_last']).copy() + rng.normal(0., .1, base['U_last'].shape)
    for ad, interpolation, mode in (
        ('forward', 'FOH', 'fixed'), ('reverse', 'ZOH', 'fixed'),
        ('forward', 'ZOH', 'global'), ('reverse', 'FOH', 'dilated'),
    ):
        p = dict(base, g_0=base['g_0']*1.03, w_states_theta=base['w_states_theta']*1.2,
                 T_max=base['T_max']*.95, rk4_steps=5, t_scp=.08,
                 dynamics_autodiff=ad, inp_param=interpolation,
                 free_final_time=mode!='fixed', time_dil=mode=='dilated')
        if 'p_w' in p:
            p.update(p_w=np.asarray(p['p_w'])+.02, r_w=p['r_w']*1.03)
        p = ns['dynamics'](p)
        S = jnp.full((1, p['K']-1), .08) if p['time_dil'] else jnp.array(.32)
        args = jnp.asarray(X), jnp.asarray(U), S
        expected = jit_cal_disc_fcn(dict(p, discretization_mode='reference'), compile=False)(*args)
        full = make_native_discretizer(p)(*args)
        endpoint = make_native_discretizer(p, terminal_only=True)(*args)
        np.testing.assert_allclose(full, expected, rtol=2e-10, atol=2e-10)
        np.testing.assert_array_equal(endpoint, full[:, -1])
        physical = jax.jit(lambda *args: int_mult(*args, p))(*args)
        native = make_native_discretizer(p, states_only=True)(*args)
        np.testing.assert_allclose(native, physical, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize('example', EXAMPLES[1:])
def test_native_di_certificate_and_fallback(example, monkeypatch):
    _, p, f = load_application(example=example)
    assert f['cal_disc_jitted'].discretization_backend == 'native'
    assert f['ncvx_ct_fcn_grad_jitted'].icore_zero_continuous_cost == (example!='di_until')
    custom = dict(p, f_func=lambda x, u, t: p['f_func'](x, u, t)*1.001)
    assert not eligibility(custom)[0]
    assert jit_cal_disc_fcn(custom, compile=False).discretization_backend == 'jax'
    with pytest.raises(ValueError, match='unavailable'):
        jit_cal_disc_fcn(dict(custom, discretization_mode='native'), compile=False)
    def no_compiler():
        raise RuntimeError('test compiler unavailable')
    monkeypatch.setattr('native_discretization.native_di.registered_target', no_compiler)
    with pytest.warns(RuntimeWarning, match='using JAX'):
        assert jit_cal_disc_fcn(p, compile=False).discretization_backend == 'jax'
    with pytest.raises(RuntimeError, match='compiler unavailable'):
        jit_cal_disc_fcn(dict(p, discretization_mode='native'), compile=False)


@pytest.mark.parametrize('p', (1, 2))
def test_until_matches_original_prefix_values_and_gradients(p):
    rng = np.random.default_rng(13)
    w_f = jnp.asarray(rng.uniform(.3, 1.3, 10))
    w_g = jnp.asarray(rng.uniform(.4, 1.4, 10))
    w_fg = jnp.array([.7, 1.2])
    c = .005
    def reference(f, g):
        terms = [conjunction(c, p, w_fg,
                    jnp.array([conjunction(c, p, w_f[:i+1], f[:i+1]), g[i]]))
                 for i in range(f.shape[0])]
        return disjunction(c, p, w_g, jnp.stack(terms))
    def optimized(f, g):
        return UNTIL(c, p, w_f, w_g, w_fg, f, g)
    expected = jax.jit(jax.value_and_grad(reference, argnums=(0, 1)))
    actual = jax.jit(jax.value_and_grad(optimized, argnums=(0, 1)))
    for f, g in ((rng.normal(size=10), rng.normal(size=10)),
                 (rng.uniform(.2, 1., 10), rng.uniform(.2, 1., 10)),
                 (-rng.uniform(.1, 1., 10), -rng.uniform(.1, 1., 10)),
                 (np.zeros(10), np.zeros(10))):
        for x, y in zip(jax.tree.leaves(expected(f, g)), jax.tree.leaves(actual(f, g))):
            np.testing.assert_allclose(x, y, rtol=2e-11, atol=2e-12)
