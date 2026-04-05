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

    if params['add_inp']:
        if npy:
            input_cost = ( params['w_inp'] * np.sum(np.square(U[3,:])))
        else:
            input_cost = ( params['w_inp'] * cp.sum(cp.square(U[3,:])))

        convex_vehicle_cost += input_cost

    if params['add_inp_trq']:
        input_cost = 0
        if npy:
            input_cost += ( params['w_inp_trq'] * np.sum(np.square(U[0,:])))
            input_cost += ( params['w_inp_trq'] * np.sum(np.square(U[1,:])))
            input_cost += ( params['w_inp_trq'] * np.sum(np.square(U[2,:])))
        else:
            input_cost += ( params['w_inp_trq'] * cp.sum(cp.square(U[0,:])))
            input_cost += ( params['w_inp_trq'] * cp.sum(cp.square(U[1,:])))
            input_cost += ( params['w_inp_trq'] * cp.sum(cp.square(U[2,:])))

        convex_vehicle_cost += input_cost

        if cost_dict is not None: # We compute the nonlin cost; otherwise lin
            cost_dict = dict_append(cost_dict, 'input_cost', input_cost)

    # --------------------------------------------------------------------------------------------------------------
    
    # Dont forget to update linear/nonlinear cost functions
    if params['add_elv_rate']:
        if npy:
            elv_rate_cost = ( params['w_elv_r'] * np.sum(np.square(X[5,:])))
        else:
            elv_rate_cost = ( params['w_elv_r'] * cp.sum(cp.square(X[5,:])))

        convex_vehicle_cost += elv_rate_cost

        if cost_dict is not None: # We compute the nonlin cost; otherwise lin
            cost_dict = dict_append(cost_dict, 'elv_rate_cost', elv_rate_cost)

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

    # CTCS
    vehicle_cons += [X[params['n_dyn']:, 0] == 0.]
    vehicle_cons += [X[params['n_dyn']:, 1:] - X[params['n_dyn']:, 0:-1] <= 1e-4]

    # --------------------------------------------------------------------------------------------------------------

    vehicle_cons += [
        X[0:8, 0] == params['x_init'][0:8],
        X[3:8, -1] == params['x_final'][3:8],
        X[9:12, 0] == params['x_init'][9:12],
        X[9:12, -1] == params['x_final'][9:12],
    ]

    vehicle_cons += [
        X[0, -1] >= params['x_final'][0] * 0.8,
        X[0, -1] <= params['x_final'][0] * 1.0,
    ]

    vehicle_cons += [
        X[12, 0] == 0.0,
        X[13, 0] == 0.0,
        X[14, 0] == 0.0,
        X[15, 0] == 0.0,
        X[16, 0] == 0.0,
        X[17, 0] == 0.0,
    ]

    vehicle_cons += [
        X[18, 0] == 0.0, # time
    ]

    vehicle_cons += [
        X[19, 0] == 1.0, # GI
        X[20, 0] == 0.0, # I
        X[21, 0] == 1.0, # GI
        X[22, 0] == 0.0, # I
    ]

    # Input conditions:
    vehicle_cons += [
        U[:, 0] == params['u_init'],
        U[:, -1] == params['u_final'],
        U[3, :] <= params['T_max'],
        U[3, :] >= params['T_min'],
        cp.abs(U[0:3, :]) <= params['tau_max'],
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

    z1 = jnp.sum(jnp.array(0.0) * X_last)
    z2 = jnp.sum(jnp.array(0.0) * X_last)
    z3 = jnp.sum(jnp.array(0.0) * X_last)
    z4 = jnp.sum(jnp.array(0.0) * X_last)
    z5 = jnp.sum(jnp.array(0.0) * X_last)
    z6 = jnp.sum(jnp.array(0.0) * X_last)
    z7 = jnp.sum(jnp.array(0.0) * X_last)
    z8 = jnp.sum(jnp.array(0.0) * X_last)

    z1 += X_last[12, -1]
    z2 += X_last[13, -1]
    z3 += X_last[14, -1]
    z4 += X_last[15, -1]
    z5 += X_last[16, -1]
    z6 += X_last[17, -1]

    c = 1e-2
    z7 += -3*((c + 1e2*X_last[20, -1])**(1/2) - (c + (0.1*X_last[19, -1]))**(1/2) )
    z8 += -3*((c + 1e2*X_last[22, -1])**(1/2) - (c + (0.1*X_last[21, -1]))**(1/2) )

    return jnp.atleast_1d([z1, z2, z3, z4, z5, z6, z7, z8])

def ncvx_ct_fcn(
    X_CT:   np.ndarray, # n_states x (K x n_dt)
    params: T.Dict[str, T.Any],
) -> jnp.ndarray:
    
    # n_ct x 1

    X = X_CT.reshape((params['n_x'], -1))
    z = jnp.sum(jnp.array(0.0) * X)

    return jnp.atleast_1d([z])
