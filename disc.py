from jax import config, jit, lax, vmap
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import typing as T
import numpy as np

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def rk4(func : T.Any,
        y0 : np.ndarray, 
        tf : float, 
        steps : int, 
        *args : T.Tuple[T.Any]) -> np.ndarray:
    """
    Implementation of the fourth-order Runge-Kutta (RK4) method for numerical integration.

    Parameters:
    - f: Function representing the system of ordinary differential equations (ODEs).
    - y0: Initial conditions (numpy array, n-dimensional column vector).
    - t: Time points for which the solution is calculated.

    Returns:
    - y: Solution of the ODEs at each time point.
    """

    t = np.linspace(0, tf, int(steps))  # Time points

    # Ensure y0 is a NumPy array (n-dimensional column vector)
    y0 = y0.reshape(-1, 1)

    # Initialize solution array
    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0.flatten(order='C')

    # Perform RK4 integration
    for i in range(len(t) - 1):

        h = t[i + 1] - t[i]
        k1 = h * func(y[i], t[i], args)
        k2 = h * func(y[i] + 0.5 * k1, t[i] + 0.5 * h, args)
        k3 = h * func(y[i] + 0.5 * k2, t[i] + 0.5 * h, args)
        k4 = h * func(y[i] + k3, t[i] + h, args)

        y[i + 1, :] = y[i, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y

def dxdt(x : np.ndarray,
         t : float, 
         *args : T.Tuple[T.Any],
        ) -> np.ndarray:

    """
    return: x_dot(t)
    """

    tt, tf, u_0, u_1, params = args[0]
    
    if params['inp_param'] == 'FOH':
        u = u_0 + (t / tf) * (u_1 - u_0)
    elif params['inp_param'] == 'ZOH':
        u = u_0.copy()

    tau = tt + t
    return params['f_func'](x, u, tau)

def int_dyn(x : np.ndarray,
                       u_0 : np.ndarray,
                       u_1 : np.ndarray,
                       params : T.Dict[str, T.Any],
                       tf : float,
                       tt : float,
                      ) -> T.Tuple[np.ndarray]:
    """
    Integration of the vehicle dynamics [0, tf]
    return: x[t+dt] and u
    """
    x_next = rk4(dxdt, x, tf, params['rk4_steps'], tt, tf, u_0, u_1, params)[-1,:]
    return x_next

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def rk4_jax(func, y0, tf, steps, *args):
    h = tf / (steps - 1)
    t = jnp.arange(steps) * h

    def step(y_t, t_i):
        y, _ = y_t
        k1 = h * func(y, t_i, *args)
        k2 = h * func(y + 0.5 * k1, t_i + 0.5 * h, *args)
        k3 = h * func(y + 0.5 * k2, t_i + 0.5 * h, *args)
        k4 = h * func(y + k3, t_i + h, *args)
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6.
        return (y_next, t_i), y_next

    (_, _), ys = lax.scan(step, (y0, t[0]), t[:-1])
    ys = jnp.vstack([y0[None, :], ys])
    return ys

def dxdt_jax(x, t, tt, tf, u_k, u_kp1, params):
    
    if params['inp_param'] == 'FOH':
        u = u_k + (t / tf) * (u_kp1 - u_k)
    elif params['inp_param'] == 'ZOH':
        u = u_k
    
    tau = tt + t
    return params['f_func'](x, u, tau)

def int_mult(X, U, S, params):

    if params['free_final_time']:
        if params['time_dil']:
            S_vec = S[0]
        else:
            S_k = S / (params['K'] - 1)
            S_vec = jnp.ones((params['K'] - 1)) * S_k
    else:
        S_k = params['t_scp']
        S_vec = jnp.ones((params['K'] - 1)) * S_k

    tt_vec = jnp.cumsum(jnp.concatenate([jnp.zeros((1,)), S_vec]))[:-1]

    def body(k):
        x_k = X[:, k]
        u_k = U[:, k]
        u_kp1 = U[:, k + 1]

        S_k = S_vec[k]  # if needed
        tt = tt_vec[k]

        return rk4_jax(dxdt_jax, x_k, S_k, params['rk4_steps'], tt, S_k, u_k, u_kp1, params)
    
    return vmap(body)(jnp.arange(params['K'] - 1)).transpose(2, 0, 1) # (K-1) x steps x n_x -> n_x x K-1 x steps

def jit_int_mult_fcn(params):
    def int_mult_wrapped(X, U, S):
        return int_mult(X, U, S, params)
    return jit(int_mult_wrapped).lower(
                    params['X_last'],
                    params['U_last'],
                    params['S_last']).compile()

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def dVdt(V, t, u_0, u_1, S, tt, params):

    n_x = params['n_x']
    n_u = params['n_u']
    K = params['K']

    i0 = params['i0']
    i1 = params['i1']
    i2 = params['i2']
    i3 = params['i3']
    i4 = params['i4']
    i5 = params['i5']
    i6 = params['i6']

    x = V[i0:i1]

    if params['inp_param'] == 'ZOH':
        beta = 0.
    elif params['inp_param'] == 'FOH':
        beta = t / params['t_scp']
        if params['free_final_time']:
            if params['time_dil']:
                beta = t
            else:
                beta = (t / (1 / (K- 1)))

    alpha = 1 - beta
    u = u_0 + beta * (u_1 - u_0)
    tau = tt + t

    A_subs = S * params['A_func'](x, u, tau)
    B_subs = S * params['B_func'](x, u, tau)
    f_subs = params['f_func'](x, u, tau)

    z_t = jnp.where(
        params['free_final_time'],
        -A_subs @ x - B_subs @ u,
        f_subs - A_subs @ x - B_subs @ u
    )

    dVdt_parts = [
        S * f_subs.T,
        (A_subs @ V[i1:i2].reshape(n_x, n_x)).reshape(-1),
        (A_subs @ V[i2:i3].reshape(n_x, n_u) + B_subs * alpha).reshape(-1),
        (A_subs @ V[i3:i4].reshape(n_x, n_u) + B_subs * beta).reshape(-1),
        (A_subs @ V[i4:i5] + f_subs).reshape(-1),
        (A_subs @ V[i5:i6] + z_t).reshape(-1)
    ]
    return jnp.concatenate(dVdt_parts)

def cal_disc(X, U, S, params):
    
    n_x = params['n_x']
    K = params['K']

    i2 = params['i2']
    i6 = params['i6']

    def single_step(k):
        x_k = X[:, k]
        u0 = U[:, k]
        u1 = U[:, k + 1]

        if params['free_final_time']:
            if params['time_dil']:
                t_scp = 1.
                tt = jnp.cumsum(jnp.concatenate((jnp.zeros(1), S[0, :])))
                tt_k = tt[k]
                S_k = S[0, k]
            else:
                t_scp = 1. / (K - 1)
                tt_k = k * S / (K - 1)
                S_k = S
        else:
            t_scp = params['t_scp']
            tt_k = k * t_scp
            S_k = 1.

        V0 = jnp.concatenate([
            x_k,                                 # i0:i1
            jnp.eye(n_x).reshape(-1),            # i1:i2
            jnp.zeros(i6 - i2)                   # rest: i2 onward
        ])
        
        return rk4_jax(dVdt, V0, t_scp, params['rk4_steps'], u0, u1, S_k, tt_k, params)

    return vmap(single_step)(jnp.arange(K - 1))

def jit_cal_disc_fcn(params):
    def cal_disc_wrapped(X, U, S):
        return cal_disc(X, U, S, params)
    return jit(cal_disc_wrapped).lower(
                    params['X_last'],
                    params['U_last'],
                    params['S_last']).compile()

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
