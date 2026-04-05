from jax import config, jit, jacfwd, lax
config.update("jax_enable_x64", True)
import jax.numpy as jnp

# -------------------------------------------
# --------------- prox-convex ---------------
# -------------------------------------------
def jit_ncvx_cvx_fcn_grad(jax_fcn_cvx, params):

    def f_cvx_fcn(X_last):
        return jax_fcn_cvx(X_last, params)
    
    def f_gf_cvx_fcn(X_last):

        K = params['K']
        n_x = params['n_x']
        N_cvx = params['f_cvx_dim']

        f_cvx = f_cvx_fcn(X_last).reshape((K * N_cvx,))
        gf_cvx = jacfwd(f_cvx_fcn)(X_last).reshape(K * N_cvx, K * n_x) # HERE NOT SURE

        return tuple((f_cvx, gf_cvx))
    
    ncvx_cvx_fcn_jitted = jit(f_cvx_fcn).lower(params['X_last']).compile()
    ncvx_cvx_fcn_grad_jitted = jit(f_gf_cvx_fcn).lower(params['X_last']).compile()

    return ncvx_cvx_fcn_jitted, ncvx_cvx_fcn_grad_jitted

def jit_ncvx_smth_fcn_grad(jax_fcn_smth, params):

    K = params['K']
    N_cvx = params['f_cvx_dim']
    N_cons = params['f_comp_dim']

    def f_smth_fcn(y_cvx):
        return jax_fcn_smth(y_cvx, params)
    
    def f_gf_smth_fcn(y_cvx):

        f_smth = f_smth_fcn(y_cvx).reshape((N_cons,))
        gf_smth = jacfwd(f_smth_fcn)(y_cvx).reshape(N_cons, K * N_cvx) # HERE NOT SURE

        return tuple((f_smth, gf_smth))
    
    y_cvx = jnp.zeros(( N_cvx, K ))

    ncvx_smth_fcn_jitted = jit(f_smth_fcn).lower(y_cvx).compile()
    ncvx_smth_fcn_grad_jitted = jit(f_gf_smth_fcn).lower(y_cvx).compile()

    return ncvx_smth_fcn_jitted, ncvx_smth_fcn_grad_jitted

def jit_ncvx_comp_fcn_grad(jax_fcn_comp, params):
    def f_comp_fcn(X_last):
        return jax_fcn_comp(X_last, params)
    
    def f_gf_comp_fcn(X_last):

        K = params['K']
        n_x = params['n_x']
        N_cons = params['f_comp_dim']

        f_comp = f_comp_fcn(X_last).reshape((N_cons,))
        gf_comp = jacfwd(f_comp_fcn)(X_last).reshape(N_cons, K * n_x)

        return tuple((f_comp, gf_comp))
    
    ncvx_comp_fcn_jitted = jit(f_comp_fcn).lower(params['X_last']).compile()
    ncvx_comp_fcn_grad_jitted = jit(f_gf_comp_fcn).lower(params['X_last']).compile()

    return ncvx_comp_fcn_jitted, ncvx_comp_fcn_grad_jitted
# -------------------------------------------
# --------------- prox-convex ---------------
# -------------------------------------------

def jit_ncvx_dt_fcn_grad(jax_fcn_dt, params):
    def f_dt_fcn(X_last):
        return jax_fcn_dt(X_last, params)
    
    def f_gf_dt_fcn(X_last):

        n_x = params['n_x']
        K = params['K']
        
        N_cons = params['f_dt_dim']

        f_dt = f_dt_fcn(X_last).reshape((N_cons,))
        gf_dt = jacfwd(f_dt_fcn)(X_last).reshape(N_cons, K * n_x)

        return tuple((f_dt, gf_dt))
    
    ncvx_dt_fcn_jitted = jit(f_dt_fcn).lower(params['X_last']).compile()
    ncvx_dt_fcn_grad_jitted = jit(f_gf_dt_fcn).lower(params['X_last']).compile()

    return ncvx_dt_fcn_jitted, ncvx_dt_fcn_grad_jitted

def jit_ncvx_ct_fcn_grad(jax_fcn_ct, params):
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
        gf_ct = jacfwd(f_ct_fcn)(X_CT).reshape(N_cons, n_x, K - 1, steps)

        def compute_grad(i_prb):
            def compute_k(k):
                def compute_j(j):
                    return (
                        gf_ct[i_prb, :, k, j] @ A_CT[:, :, k, j],
                        gf_ct[i_prb, :, k, j] @ B_CT[:, :, k, j],
                        gf_ct[i_prb, :, k, j] @ C_CT[:, :, k, j],
                        gf_ct[i_prb, :, k, j] @ S_CT[:, :, k, j]
                    )
                
                gf_A_j, gf_B_j, gf_C_j, gf_S_j = lax.map(compute_j, jnp.arange(steps))
                
                gf_A = jnp.sum(gf_A_j, axis=0)
                gf_B = jnp.sum(gf_B_j, axis=0)
                gf_C = jnp.sum(gf_C_j, axis=0)
                gf_S = jnp.sum(gf_S_j, axis=0)

                return gf_A, gf_B, gf_C, gf_S

            gf_A_k, gf_B_k, gf_C_k, gf_S_k = lax.map(compute_k, jnp.arange(K - 1))

            gf_A_k = gf_A_k.T.flatten(order='C')
            gf_B_k = gf_B_k.T.flatten(order='C')
            gf_C_k = gf_C_k.T.flatten(order='C')
            gf_S_k = gf_S_k.T.flatten(order='C')

            return gf_A_k, gf_B_k, gf_C_k, gf_S_k

        gf_A_ct, gf_B_ct, gf_C_ct, gf_S_ct = lax.map(compute_grad, jnp.arange(N_cons))

        return tuple((f_ct, gf_A_ct, gf_B_ct, gf_C_ct, gf_S_ct))

    X_CT = jnp.zeros((params['n_x'], (params['K'] - 1), params['rk4_steps']))
    V_CT = jnp.zeros((params['K'] - 1, params['rk4_steps'], params['i6']))

    ncvx_ct_fcn_jitted = jit(f_ct_fcn).lower(X_CT).compile()
    ncvx_ct_fcn_grad_jitted = jit(f_gf_ct_fcn).lower(V_CT).compile()

    return ncvx_ct_fcn_jitted, ncvx_ct_fcn_grad_jitted
