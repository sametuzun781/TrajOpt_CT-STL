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
    N_dt = int(10.00)

    # dimensions
    n_dyn = 12+6+1+2+2 # 12 vehicle, 6 obs, 1 time, 2*2 evnt
    n_ctc = 1
    n_x = n_dyn + n_ctc  # 12 states + 1 slack
    n_u = 4

    # gravity
    g_0 = 9.806

    # vehicle limits
    max_ang = 45.0
    max_ang_rate = 100.0
    max_ang_rate_yaw = 100.0

    penalty_fcn_list = ['abs', 'max', 'None']  

    # initial / final states and inputs
    x_init = np.hstack(([  0.0, -10.0, 1.0], np.zeros(n_x - 3)))
    x_final = np.hstack(([ 10.0, 0.0, 1.0], np.zeros(n_x - 3)))
    u_init = np.array([0.0, 0.0, 0.0, g_0])
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
    U_last[3, :] = g_0

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
        "x_init": x_init,
        "x_final": x_final,
        "u_init": u_init,
        "u_final": u_final,

        # physics
        "g_0": g_0,
        "T_max": g_0 * 1.75,
        "T_min": g_0 * 0.06,
        "tau_max": 0.25,

        # input‐parameter interpolation mode
        "inp_param": ["ZOH", "FOH"][1],

        "add_minmax_sig": False,
        "min_S": t_scp * 0.1,
        "max_S": t_scp * 100.0,
        "w_S": 1.0,

        "add_inp": False,
        "add_inp_trq": False,
        "add_elv_rate": False,
        "w_inp": 1.0,
        "w_inp_trq": 1.0,
        "w_elv_r": 1.0,

        # discrete‐time penalty settings
        "f_dt_dim":  8,
        "w_dt_fcn":  [1.0, 
                      1.0, 
                      1.0,
                      1.0,
                      1.0,
                      1.0,
                      1.0,
                      1.0,
                      ],
        "pen_dt_fcn": [penalty_fcn_list[1], 
                       penalty_fcn_list[1],
                       penalty_fcn_list[1],
                       penalty_fcn_list[1],
                       penalty_fcn_list[1],
                       penalty_fcn_list[1],
                       penalty_fcn_list[1],
                       penalty_fcn_list[1],
                       ],   # or whatever penalty list index you want
        "name_dt_fcn": [['obs_4', 'None_dt1'][0], 
                        ['obs_1', 'None_dt2'][0],
                        ['obs_2', 'None_dt3'][0],
                        ['obs_5', 'None_dt4'][0],
                        ['obs_6', 'None_dt5'][0],
                        ['obs_3', 'None_dt6'][0],
                        ['evnt_0', 'None_dt7'][0],
                        ['evnt_1', 'None_dt8'][0],
                        ],  # must match the shape your solver expects

        # (and similarly, if you use continuous‐time penalties)
        "f_ct_dim":  1,
        "w_ct_fcn":  [20.0],
        "pen_ct_fcn": [penalty_fcn_list[2]],
        "name_ct_fcn": [['until_pos_spd', 'spd', 'None_ct'][2]],

        # -------------------------------------------
        # --------------- prox-convex ---------------
        # -------------------------------------------
        # composite function : comp = smth(cvx)
        ###
        "f_cvx_dim":  1,
        "f_comp_dim":  1,
        "w_comp_fcn":  [1e1]*1,

        ###
        "name_comp_fcn": [["None_comp"][0]],
        "ncvx_solver" : ['pl', 'pcx'][0],
        # -------------------------------------------
        # --------------- prox-convex ---------------
        # -------------------------------------------

        # physical inertia (kg·m²)
        "I_x": 7e-2,
        "I_y": 7e-2,
        "I_z": 1.27e-1,

        # vehicle limits
        "vehicle_v_max": 20.0,
        "spd_lim":       2.0,

        # ----------------------------------------------
        "p_w_1":           np.array([-5.0, 0.0, 1.0]),
        "p_w_2":           np.array([-10.0, -5.0, 1.0]),
        "r_w_1":           0.1*5,
        "r_w_2":           0.1*5,

        "pw1_base": -16.0,
        "pw2_base":   7.0,
        "pw_range":  23.0,
        # ----------------------------------------------


        # ----------------------------------------------
        "x_stc_7": 0.5,
        "x_stc_8": 1.5,
        "y_stc_4": 1.0,

        "x_stc_1": 0.5,
        "x_stc_2": 1.5,
        "y_stc_1": -1.0,
        # ----------------------------------------------

        # ----------------------------------------------
        "x_stc_3": 3.5,
        "x_stc_4": 4.5,
        "y_stc_2": -8.0,

        "x_stc_9": 3.5,
        "x_stc_10": 4.5,
        "y_stc_5": -10.0,
        # ----------------------------------------------

        # ----------------------------------------------

        "x_stc_11": 6.5,
        "x_stc_12": 7.5,
        "y_stc_6": 1.0,

        "x_stc_5": 6.50,
        "x_stc_6": 7.50,
        "y_stc_3": -1.00,
        # ----------------------------------------------

        "y_stc_range": 3,

        # ----------------------------------------------

        # vehicle limits
        "max_ang": max_ang,
        "max_ang_rate": max_ang_rate,
        "max_ang_rate_yaw": max_ang_rate_yaw,

        # converted to radians
        "phi_bd": np.deg2rad(max_ang),
        "theta_bd": np.deg2rad(max_ang),
        "phi_rate": np.deg2rad(max_ang_rate),
        "theta_rate": np.deg2rad(max_ang_rate),
        "yaw_rate": np.deg2rad(max_ang_rate_yaw),

        # altitude limits
        "add_min_alt": False,
        "add_max_alt": False,
        "min_alt": -5.0,
        "max_alt": 10.0,

        "yaw_fixed": False,
        "yaw_fixed_all": False,
        "yaw_fx_deg": 0.0,

        # state‐weight penalties
        "w_states_spd":  100.0,
        "w_states_alt":  100.0,

        "w_states_phi":  100.0,
        "w_states_tht":  100.0,

        "w_states_p":    100.0,
        "w_states_q":    100.0,
        "w_states_r":    100.0,

        # optimization settings
        "ite": 3000,
        "w_con_dyn": 300,
        "w_con_ctc": 200,

        "adaptive_step": False,
        "ptr_term": 1e-4,
        # "ptr_term": 1e-7,
        "w_ds": 400.0,
        "w_ptr": 1.0,
        "w_ptr_min": 1e-3,
        "r0": 0.01,
        "r1": 0.1,
        "r2": 0.9,

        # solver settings
        "generate_code": True,
        "use_generated_code": True,
        "convex_solver": "QOCO",  # or "ECOS", "CLARABEL", "MOSEK", etc.

        # integrator
        "rk4_steps": int(10.00),

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
    sc = params["scale_fac"]

    # divide‐by‐scale parameters
    to_divide = [
        "g_0", "T_max", "T_min", "vehicle_v_max",
        "min_alt", "max_alt", "spd_lim", 
        "p_w_1", "r_w_1",
        "p_w_2", "r_w_2",
        "x_stc_1", "x_stc_2",
        "x_stc_3", "x_stc_4",
        "x_stc_5", "x_stc_6",
        "x_stc_7", "x_stc_8",
        "x_stc_9", "x_stc_10",
        "x_stc_11", "x_stc_12",
        "y_stc_1", "y_stc_2",
        "y_stc_3", "y_stc_4",
        "y_stc_5", "y_stc_6",
        "y_stc_range",
        "pw1_base", "pw2_base", "pw_range",
    ]

    for key in to_divide:
        params[key] /= sc

    # array‐specific slices
    params["x_init"][:6]   /= sc
    params["x_final"][:6]  /= sc
    params["X_last"][:6, :] /= sc

    params["u_init"][3]   /= sc
    params["u_final"][3]  /= sc
    params["U_last"][3, :] /= sc

    # multiply‐by‐scale parameters
    to_multiply = ["I_x", "I_y", "I_z", "tau_max"]
    for key in to_multiply:
        params[key] *= sc

    return params

def unscale_prox_results(
    prox_results: T.Dict[str, np.ndarray],
    params: T.Dict[str, T.Any]
) -> T.Dict[str, np.ndarray]:
    """
    Undo the scaling on the primal variables returned by the prox solver.
    """
    sc = params["scale_fac"]

    prox_results["X_new"][:6, :] *= sc
    prox_results["U_new"][3, :]  *= sc
    prox_results["U_new"][:3, :] /= sc

    return prox_results
