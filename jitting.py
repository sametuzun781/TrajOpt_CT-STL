import jax
from jax.extend import core as jax_core
from jax import config, jit, jacfwd, jacrev, lax
config.update("jax_enable_x64", True)
import jax.numpy as jnp

# -------------------------------------------
# --------------- prox-convex ---------------
# -------------------------------------------
def jit_ncvx_cvx_fcn_grad(jax_fcn_cvx, params, compile=True):

    def f_cvx_fcn(X_last):
        return jax_fcn_cvx(X_last, params)
    
    def f_gf_cvx_fcn(X_last):

        K = params['K']
        n_x = params['n_x']
        N_cvx = params['f_cvx_dim']

        f_cvx = f_cvx_fcn(X_last).reshape((K * N_cvx,))
        gf_cvx = (jacrev if N_cvx < n_x else jacfwd)(f_cvx_fcn)(X_last).reshape(K * N_cvx, K * n_x) # HERE NOT SURE

        return tuple((f_cvx, gf_cvx))
    
    ncvx_cvx_fcn_jitted = jit(f_cvx_fcn)
    if compile:
        ncvx_cvx_fcn_jitted = ncvx_cvx_fcn_jitted.lower(params['X_last']).compile()
    ncvx_cvx_fcn_grad_jitted = jit(f_gf_cvx_fcn)
    if compile:
        ncvx_cvx_fcn_grad_jitted = ncvx_cvx_fcn_grad_jitted.lower(params['X_last']).compile()

    return ncvx_cvx_fcn_jitted, ncvx_cvx_fcn_grad_jitted

def jit_ncvx_smth_fcn_grad(jax_fcn_smth, params, compile=True):

    K = params['K']
    N_cvx = params['f_cvx_dim']
    N_cons = params['f_comp_dim']

    def f_smth_fcn(y_cvx):
        return jax_fcn_smth(y_cvx, params)
    
    def f_gf_smth_fcn(y_cvx):

        f_smth = f_smth_fcn(y_cvx).reshape((N_cons,))
        gf_smth = (jacrev if N_cons < K * N_cvx else jacfwd)(f_smth_fcn)(y_cvx).reshape(N_cons, K * N_cvx) # HERE NOT SURE

        return tuple((f_smth, gf_smth))
    
    y_cvx = jnp.zeros(( N_cvx, K ))

    ncvx_smth_fcn_jitted = jit(f_smth_fcn)
    if compile:
        ncvx_smth_fcn_jitted = ncvx_smth_fcn_jitted.lower(y_cvx).compile()
    ncvx_smth_fcn_grad_jitted = jit(f_gf_smth_fcn)
    if compile:
        ncvx_smth_fcn_grad_jitted = ncvx_smth_fcn_grad_jitted.lower(y_cvx).compile()

    return ncvx_smth_fcn_jitted, ncvx_smth_fcn_grad_jitted

def jit_ncvx_comp_fcn_grad(jax_fcn_comp, params, compile=True):
    def f_comp_fcn(X_last):
        return jax_fcn_comp(X_last, params)
    
    def f_gf_comp_fcn(X_last):

        K = params['K']
        n_x = params['n_x']
        N_cons = params['f_comp_dim']

        f_comp = f_comp_fcn(X_last).reshape((N_cons,))
        gf_comp = (jacrev if N_cons < K * n_x else jacfwd)(f_comp_fcn)(X_last).reshape(N_cons, K * n_x)

        return tuple((f_comp, gf_comp))
    
    ncvx_comp_fcn_jitted = jit(f_comp_fcn)
    if compile:
        ncvx_comp_fcn_jitted = ncvx_comp_fcn_jitted.lower(params['X_last']).compile()
    ncvx_comp_fcn_grad_jitted = jit(f_gf_comp_fcn)
    if compile:
        ncvx_comp_fcn_grad_jitted = ncvx_comp_fcn_grad_jitted.lower(params['X_last']).compile()

    return ncvx_comp_fcn_jitted, ncvx_comp_fcn_grad_jitted
# -------------------------------------------
# --------------- prox-convex ---------------
# -------------------------------------------

def jit_ncvx_dt_fcn_grad(jax_fcn_dt, params, compile=True):
    def f_dt_fcn(X_last):
        return jax_fcn_dt(X_last, params)
    
    def f_gf_dt_fcn(X_last):

        n_x = params['n_x']
        K = params['K']
        
        N_cons = params['f_dt_dim']

        f_dt = f_dt_fcn(X_last).reshape((N_cons,))
        gf_dt = (jacrev if N_cons < K * n_x else jacfwd)(f_dt_fcn)(X_last).reshape(N_cons, K * n_x)

        return tuple((f_dt, gf_dt))
    
    ncvx_dt_fcn_jitted = jit(f_dt_fcn)
    if compile:
        ncvx_dt_fcn_jitted = ncvx_dt_fcn_jitted.lower(params['X_last']).compile()
    ncvx_dt_fcn_grad_jitted = jit(f_gf_dt_fcn)
    if compile:
        ncvx_dt_fcn_grad_jitted = ncvx_dt_fcn_grad_jitted.lower(params['X_last']).compile()

    return ncvx_dt_fcn_jitted, ncvx_dt_fcn_grad_jitted

def jit_ncvx_ct_fcn_grad(jax_fcn_ct, params, autodiff='auto', compile=True):
    if autodiff not in ('auto', 'forward', 'reverse'):
        raise ValueError("autodiff must be auto, forward or reverse.")
    reverse = autodiff == 'reverse' or (autodiff == 'auto' and params['f_ct_dim'] < params['n_x'] * (params['K'] - 1) * params['rk4_steps'])
    derivative = jacrev if reverse else jacfwd
    def f_ct_fcn(X_CT):
        return jax_fcn_ct(X_CT, params)

    def f_gf_ct_fcn(V_CT):
        n_x = params['n_x']
        n_u = params['n_u']
        K = params['K']
        
        i0, i1 = params['i0'], params['i1']
        i2, i3 = params['i2'], params['i3']
        i4, i5 = params['i4'], params['i5']

        steps = params['rk4_steps']
        N_cons = params['f_ct_dim']

        X_CT = V_CT[:, :, i0:i1].transpose(2, 0, 1)
        A_CT = V_CT[:, :, i1:i2].transpose(2, 0, 1).reshape(n_x, n_x, K - 1, steps)
        B_CT = V_CT[:, :, i2:i3].transpose(2, 0, 1).reshape(n_x, n_u, K - 1, steps)
        C_CT = V_CT[:, :, i3:i4].transpose(2, 0, 1).reshape(n_x, n_u, K - 1, steps)
        S_CT = V_CT[:, :, i4:i5].transpose(2, 0, 1).reshape(n_x, 1, K - 1, steps)

        f_ct = f_ct_fcn(X_CT).reshape((N_cons,))
        gf_ct = derivative(f_ct_fcn)(X_CT).reshape(N_cons, n_x, K - 1, steps)

        # Contract all intervals and sample points together, preserving row order.
        # gf_ct[o,i,k,t] multiplies sensitivity[k,t,i,j]. No global dense
        # trajectory Jacobian with respect to all interval decisions is built.
        A = V_CT[:, :, i1:i2].reshape(K - 1, steps, n_x, n_x)
        B = V_CT[:, :, i2:i3].reshape(K - 1, steps, n_x, n_u)
        C = V_CT[:, :, i3:i4].reshape(K - 1, steps, n_x, n_u)
        S = V_CT[:, :, i4:i5].reshape(K - 1, steps, n_x)
        gf_A_ct = jnp.einsum('oikt,ktij->ojk', gf_ct, A).reshape(N_cons, -1)
        gf_B_ct = jnp.einsum('oikt,ktij->ojk', gf_ct, B).reshape(N_cons, -1)
        gf_C_ct = jnp.einsum('oikt,ktij->ojk', gf_ct, C).reshape(N_cons, -1)
        gf_S_ct = jnp.einsum('oikt,kti->ok', gf_ct, S).reshape(N_cons, -1)

        return tuple((f_ct, gf_A_ct, gf_B_ct, gf_C_ct, gf_S_ct))

    X_CT = jnp.zeros((params['n_x'], (params['K'] - 1), params['rk4_steps']))
    V_CT = jnp.zeros((params['K'] - 1, params['rk4_steps'], params['i6']))

    ncvx_ct_fcn_jitted = jit(f_ct_fcn)
    if compile:
        ncvx_ct_fcn_jitted = ncvx_ct_fcn_jitted.lower(X_CT).compile()
    ncvx_ct_fcn_grad_jitted = jit(f_gf_ct_fcn)
    if compile:
        ncvx_ct_fcn_grad_jitted = ncvx_ct_fcn_grad_jitted.lower(V_CT).compile()

    # Endpoint-only integration is valid only if the complete CT callback is
    # structurally independent of its trajectory and its constant value is zero.
    # The augmented physical dynamics still contain all continuous constraints.
    graph = jax.make_jaxpr(f_ct_fcn)(jax.ShapeDtypeStruct(X_CT.shape, X_CT.dtype)).jaxpr
    dependent = set(graph.invars)
    for equation in graph.eqns:
        if any(not isinstance(var, jax_core.Literal) and var in dependent for var in equation.invars):
            dependent.update(equation.outvars)
    constant = all(isinstance(var, jax_core.Literal) or var not in dependent for var in graph.outvars)
    safe_constants = not graph.effects and all(e.primitive.name in {
        'broadcast_in_dim', 'reshape', 'convert_element_type', 'squeeze'
    } for e in graph.eqns)
    zero = constant and safe_constants and bool(jnp.all(ncvx_ct_fcn_jitted(X_CT) == 0.0))
    ncvx_ct_fcn_grad_jitted.icore_zero_continuous_cost = zero
    return ncvx_ct_fcn_jitted, ncvx_ct_fcn_grad_jitted
