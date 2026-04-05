import typing as T
import numpy as np

def params_fcn( 
    t_f: float,
    K: int,
    free_final_time: bool,
    time_dil: bool
) -> T.Dict[str, T.Any]:
    """
    Build and return all static parameters for the trajectory optimization.
    """
    # time‐scaling
    t_scp = t_f / (K - 1)

    # number of sub‐steps per interval
    N_dt = 10

    # dimensions
    n_dyn = 6
    n_ctc = 1
    n_x = n_dyn + n_ctc
    n_u = 3

    # gravity
    g_0 = 9.806

    penalty_fcn_list = ['abs', 'max', 'none']  

    # initial / final states and inputs
    x_init = np.hstack(([-10.0, -10.0, 0.0], np.zeros(n_x - 3)))
    x_final = np.hstack(([ 10.0,  10.0, 0.0], np.zeros(n_x - 3)))
    u_init = np.array([0.0, 0.0, g_0])
    u_final = u_init.copy()

    # index offsets
    dims = [
        n_x,            # states
        n_x * n_x,      # state × state terms
        n_x * n_u,      # state × input
        n_x * n_u,      # state × input
        n_x,            # states
        n_x             # states
    ]

    offsets = np.concatenate(([0], np.cumsum(dims))).astype(int)
    i0, i1, i2, i3, i4, i5, i6 = tuple(offsets)

    # lin‐spaced warm start
    X_last = np.linspace(x_init, x_final, K).T
    U_last = np.zeros((n_u, K))
    U_last[2, :] = g_0

    # time dilation warm start
    if time_dil:
        S_last = t_scp * np.ones((1, K - 1))
    else:
        S_last = np.array(t_f, dtype=np.float64)

    return {
        # basic settings
        "t_f": t_f,
        "K": K,
        "t_scp": t_scp,
        "N_dt":   N_dt,
        "free_final_time": free_final_time,
        "time_dil": time_dil,

        # dimensions
        "n_dyn": n_dyn,
        "n_x": n_x,
        "n_u": n_u,

        # index offsets into flattened decision vector
        "i0": i0,
        "i1": i1,
        "i2": i2,
        "i3": i3,
        "i4": i4,
        "i5": i5,
        "i6": i6,

        # dynamics warm starts
        "X_last": X_last,
        "U_last": U_last,
        "S_last": S_last,

        # initial / final conditions
        "x_init" : x_init,
        "x_final": x_final,
        "u_init" : u_init,
        "u_final": u_final,

        # physics
        "g_0" : g_0,
        "T_max": g_0 * 1.75,

        # input‐parameter interpolation mode
        "inp_param": ["ZOH", "FOH"][1],

        "add_minmax_sig": False,
        "min_S": t_scp * 0.1,
        "max_S": t_scp * 100.0,
        "w_S": 1.0,

        "add_inp": False,
        "w_inp": 1.0,

        # discrete‐time penalty settings
        "f_dt_dim":  1,
        "w_dt_fcn":  [10],
        "pen_dt_fcn": [penalty_fcn_list[2]],   # or whatever penalty list index you want
        "name_dt_fcn": [['until_pos_spd', 'spd', 'eventually_visit_dt', 'none'][3]],  # must match the shape your solver expects

        # (and similarly, if you use continuous‐time penalties)
        "f_ct_dim":  1,
        "w_ct_fcn":  [10],
        "pen_ct_fcn": [penalty_fcn_list[2]],
        "name_ct_fcn": [['until_pos_spd', 'none'][0]],

        # -------------------------------------------
        # --------------- prox-convex ---------------
        # -------------------------------------------
        "f_cvx_dim":  1,
        "f_comp_dim":  1,
        "w_comp_fcn":  [10],

        ###
        "name_comp_fcn": [["spd_circle", "eventually_visit", "none"][1]],
        "ncvx_solver" : ['pl', 'pcx'][0],
        # -------------------------------------------
        # --------------- prox-convex ---------------
        # -------------------------------------------

        # vehicle limits
        "spd_lim":       5.0,
        "p_w":           np.array([2.5, -10.0, 0.0]),
        "r_w":           0.2,

        # vehicle limits
        "v_max": 20.0,

        # converted to radians
        "theta": np.deg2rad(45),

        # state‐weight penalties
        "w_states_theta": 1.0,
        "w_states_v_max": 1.0,
        "w_states_T_max": 1.0,

        # optimization settings
        "ite": 100,
        "w_con_dyn": 1e2,
        "w_con_ctc": 1e0,

        "adaptive_step": True,
        "ptr_term": 1e-4,
        "w_ds": 1.0,
        "w_ptr": 1e0,
        "w_ptr_min": 1e-3,
        "r0": 0.01,
        "r1": 0.1,
        "r2": 0.9,

        # solver settings
        "generate_code": False,
        "use_generated_code": True,
        "convex_solver": "QOCO",  # or "ECOS", "CLARABEL", "MOSEK", etc.

        # integrator
        "rk4_steps": 10,

        # scaling
        "scale_fac": 10.0,

        # plotting
        "save_fig": True,
        "fig_format": "pdf",
        "fig_png_dpi": 600,
    }

def scale_params(params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
    """
    Scale certain parameters (positions, inputs, moments, etc.) by params['scale_fac'].
    """
    return params

def unscale_prox_results(
    prox_results: T.Dict[str, np.ndarray],
    params: T.Dict[str, T.Any]
) -> T.Dict[str, np.ndarray]:
    """
    Undo the scaling on the primal variables returned by the prox solver.
    """
    return prox_results
