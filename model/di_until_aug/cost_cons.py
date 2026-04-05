from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap

import typing as T
import numpy as np
import cvxpy as cp
from utils import dict_append
from stl import conjunction, disjunction, UNTIL

def cvx_cost_fcn(
    X:    np.ndarray,
    U:    np.ndarray,
    S:    np.ndarray,
    nu:   np.ndarray,
    params: T.Dict[str, T.Any],
    npy: bool,
    cost_dict: T.Dict[str, T.Any] = None
) -> T.Union[float, T.Tuple[T.Any, T.Dict[str, T.Any]]]:
    
    # --------------------------------------------------------------------------------------------------------------
    
    convex_vehicle_cost = 0.
    if params['free_final_time']:
        if params['time_dil']:
            if npy:
                S_cost = params['w_S'] * np.linalg.norm(S[0, :], 1) / params['t_f']
            else:
                S_cost = params['w_S'] * cp.pnorm(S[0, :], 1) / params['t_f']
        else:
            if npy:
                S_cost = params['w_S'] * S / params['t_f']
            else:
                S_cost = params['w_S'] * cp.pnorm(S, 1) / params['t_f']

        convex_vehicle_cost += S_cost

        if cost_dict is not None: # We compute the nonlin cost; otherwise lin
            cost_dict = dict_append(cost_dict, 'S_cost', S_cost)

    # --------------------------------------------------------------------------------------------------------------
    
    if cost_dict is not None: # We compute the nonlin cost; otherwise lin
        dyn_cost = params['w_con_dyn'] * np.linalg.norm(nu[:params['n_dyn'], :].reshape(-1), 1)
        ctc_cost = params['w_con_ctc'] * np.linalg.norm(nu[params['n_dyn']:, :].reshape(-1), 1)
        cost_dict = dict_append(cost_dict, 'dyn_cost', dyn_cost)
        cost_dict = dict_append(cost_dict, 'ctc_cost', ctc_cost)

    # --------------------------------------------------------------------------------------------------------------

    if cost_dict is not None: # We compute the nonlin cost; otherwise lin
        return convex_vehicle_cost, cost_dict
    else:
        return convex_vehicle_cost

def cvx_cons_fcn(
    X:      np.ndarray,
    U:      np.ndarray,
    S:      np.ndarray,
    params: T.Dict[str, T.Any],
) -> T.List:
    
    vehicle_cons = []

    # --------------------------------------------------------------------------------------------------------------

    if params['free_final_time']:
        if params['time_dil']:
            if params['add_minmax_sig']:
                vehicle_cons += [
                        S[0, :] >= params['min_S'], 
                        S[0, :] <= params['max_S'], 
                    ]
            else:
                vehicle_cons += [
                        S[0, :] >= 1e-4, 
                    ]

    # --------------------------------------------------------------------------------------------------------------

    vehicle_cons += [
        X[:6,  0] == params['x_init'][:6],
        X[:6, -1] == params['x_final'][:6],
        X[6,  0] == 0.0, # Integral
        X[7,  0] == 1.0, # Geometric Integral
    ]

    # CTCS
    vehicle_cons += [
        X[params['n_dyn']:params['n_dyn']+1, 0] == 0.
    ]
    vehicle_cons += [
        X[params['n_dyn']:params['n_dyn']+1, 1:] - X[params['n_dyn']:params['n_dyn']+1, 0:-1] <= 1e-4
    ]

    # Input conditions:
    vehicle_cons += [
        U[:, 0] == params['u_init'],
        U[:, -1] == params['u_final'],
    ]

    # --------------------------------------------------------------------------------------------------------------
    
    return vehicle_cons

# -------------------------------------------
# --------------- prox-convex ---------------
# -------------------------------------------
def ncvx_cvx_cp_fcn(
    X_last: np.ndarray, # n_states x K
    params: T.Dict[str, T.Any],
) -> cp.Expression:

    return np.zeros((1, X_last[0,:].shape[0]))

def ncvx_cvx_fcn(
    X_last: np.ndarray, # n_states x K
    params: T.Dict[str, T.Any],
) -> jnp.ndarray:
    
    return jnp.zeros((1, X_last[0,:].shape[0]))

def ncvx_smth_fcn(
    y_cvx: jnp.ndarray, # n_cvx x K
    params: T.Dict[str, T.Any],
) -> jnp.ndarray:
    
    return jnp.atleast_1d([jnp.array(0.0)])

def ncvx_comp_fcn(
    X_last: np.ndarray, # n_states x K
    params: T.Dict[str, T.Any],
) -> jnp.ndarray:
    
    # out: n_comp x 1
    y_cvx = ncvx_cvx_fcn(X_last, params) # n_cvx x K
    return ncvx_smth_fcn(y_cvx, params) # 1 x 1
# -------------------------------------------
# --------------- prox-convex ---------------
# -------------------------------------------

def ncvx_dt_fcn(
    X_last: np.ndarray, # n_states x K
    params: T.Dict[str, T.Any],
) -> jnp.ndarray:
    
    # n_dt x 1
    c = 1e-3
    z = jnp.sum(jnp.array(0.0) * X_last)
    z += ( (c + (0.1*X_last[7, -1]))**(1/2) - c**(1/2) )
    return jnp.atleast_1d([z]) 

def ncvx_ct_fcn(
    X_CT:   np.ndarray, # n_states x (K x n_dt)
    params: T.Dict[str, T.Any],
) -> jnp.ndarray:
    
    # n_ct x 1
    X = X_CT.reshape((params['n_x'], -1))
    z = jnp.sum(jnp.array(0.0) * X)
    return jnp.atleast_1d([z])
