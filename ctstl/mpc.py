import typing as T
import numpy as np
from ctstl.disc import int_dyn

def RUN(
    prox_results: T.Dict[str, np.ndarray],
    params:       T.Dict[str, T.Any]
) -> T.Dict[str, np.ndarray]:
    """
    Simulate the generated trajectory by subdividing each SCP interval
    and calling `int_dyn` at each sub‐step.

    Returns a dict with:
      - times_all:      (M,) array of all simulation times
      - times_nodes:    (K,) array of node times
      - x_all:          (M, n_x) stacked states
      - u_all:          (2, n_u, M-1) stacked control pairs at each sub-step
      - x_nmpc_all:     (n_x, K) the NMPC states at the K nodes
      - u_nmpc_all:     (n_u, K) the NMPC inputs at the K nodes
      - s_nmpc_all:     (K-1,) the NMPC segment durations
    """

    X_new = prox_results['X_new']    # shape (n_x, K)
    U_new = prox_results['U_new']    # shape (n_u, K)
    S_new = prox_results['S_new']    # shape (1, K-1) or (K-1,)

    K = params['K']
    t_f = params['t_f']
    t_scp = params['t_scp']
    N_dt = params['N_dt']
    free_final_time = params['free_final_time']
    time_dil = params['time_dil']
    inp_param = params['inp_param']

    if free_final_time:
        t_f = np.sum(S_new)

    if time_dil:
        S_k = S_new[0,:]
    else:
        S_k = np.ones(K-1) * t_f / (K-1)

    x_next = X_new[:,0].copy()
    x_all = [x_next]
    u_all = []
    times_all = []
    times_nodes = np.concatenate((np.zeros(1), np.cumsum(S_k)))
    for idx in range(K - 1):

        N_dt_k = int(np.ceil(S_k[idx] / (t_scp / N_dt)))
        delta_S = S_k[idx] / N_dt_k
        for t_in in np.linspace(0, S_k[idx]-(delta_S), N_dt_k):
            params['t_curr'] = times_nodes[idx] + t_in
            times_all.append(params['t_curr'])

            if inp_param == 'ZOH':
                u_0 = U_new[:, idx]
                u_1 = U_new[:, idx]
                
            elif inp_param == 'FOH':
                beta_1 = t_in / S_k[idx]
                beta_2 = (t_in+delta_S) / S_k[idx]

                u_0 = U_new[:, idx] + beta_1 * (U_new[:, idx+1] - U_new[:, idx])
                u_1 = U_new[:, idx] + beta_2 * (U_new[:, idx+1] - U_new[:, idx])
            
            x_next = int_dyn(x_next, u_0, u_1, params, delta_S, tt=params['t_curr'])

            x_all.append(x_next)
            u_all.append(np.vstack((u_0, u_1)))

    times_all.append(times_nodes[-1])
    times_all = np.array(times_all)

    scp_results = {
        "times_all":   times_all,
        "times_nodes": times_nodes,
        "x_all":       np.vstack(x_all),
        "u_all":       np.dstack(u_all),
        "x_nmpc_all":  np.dstack(X_new),
        "u_nmpc_all":  np.dstack(U_new),
        "s_nmpc_all":  S_new,
    }

    return scp_results
