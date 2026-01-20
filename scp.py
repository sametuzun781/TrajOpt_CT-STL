import time
import copy
import typing as T
import numpy as np
from utils import dict_append, update_cost_dict, print_ite
from cvx import solve_parsed_problem

def scp_noncvx_cost(
    X_new:                  np.ndarray = None,
    U_new:                  np.ndarray = None,
    S_new:                  np.ndarray = None,
    X_last:                 np.ndarray = None,
    U_last:                 np.ndarray = None,
    S_last:                 np.ndarray = None,
    nu_new:                 np.ndarray = None,
    nl_nu_new:              np.ndarray = None,
    ncvx_dt_cost:           np.ndarray = None,
    ncvx_ct_cost:           np.ndarray = None,
    ncvx_cost_grad_dt_last: T.Tuple[np.ndarray, np.ndarray] = None,
    ncvx_cost_grad_ct_last: T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None,
    w_tr:                   float = None,
    params:                 T.Dict[str, T.Any] = None,
    fcn_dict:               T.Dict[str, T.Any] = None,
    cost_dict:              T.Dict[str, T.Any] = None,
) -> T.Dict[str, T.Any]:
    """
    Compute and append all cost components for one SCP iteration:
      - nonlinear dynamic cost
      - convex cost (cvx_cost via fcn_dict['cvx_cost_fcn'])
      - linearized nonconvex penalty (ncvx_cost)
      - prox‐trust cost (ptr_cost)

    Returns the updated cost_dict.
    """
    
    # 1) Nonlinear dynamic cost
    nl_dyn = params['w_con_dyn'] * np.linalg.norm(nl_nu_new[:12, :].reshape(-1), 1)
    nl_stt = params['w_con_stt'] * np.linalg.norm(nl_nu_new[12, :].reshape(-1), 1)
    nlcd   = nl_dyn + nl_stt
    
    # 2) Convex cost
    cvx_cost, cost_dict = fcn_dict['cvx_cost_fcn'](
        X_new, U_new, S_new, nl_nu_new, params, npy=True, cost_dict=cost_dict
    )

    # 3) Helper to compute linearized nonconvex penalty
    def compute_lin_ncvx( 
        lin_ncvx_dt_cost: np.ndarray,
        lin_ncvx_ct_cost: np.ndarray,
        params: T.Dict[str, T.Any],
        cost_dict: T.Dict[str, T.Any] = None,
    ) -> T.Tuple[float, T.Dict[str, T.Any]]:
        # --------------------------------------------------------------------------------------------------------------
        
        lin_ncvx_cost = 0.
        if cost_dict is not None: # We compute the nonlin cost; otherwise lin 
            ncvx_cost = 0.

        for i_dt in range(params['f_dt_dim']):

            if params['pen_dt_fcn'][i_dt] == 'abs':
                lin_ncvx_dt_i = params['w_dt_fcn'][i_dt] * np.abs(lin_ncvx_dt_cost[i_dt])

            elif params['pen_dt_fcn'][i_dt] == 'max':
                lin_ncvx_dt_i = params['w_dt_fcn'][i_dt] * np.maximum(0.0, lin_ncvx_dt_cost[i_dt])

            else:
                lin_ncvx_dt_i = params['w_dt_fcn'][i_dt] * lin_ncvx_dt_cost[i_dt]

            lin_ncvx_cost += lin_ncvx_dt_i

            if cost_dict is not None: # We compute the nonlin cost; otherwise lin
                cost_dict = dict_append(cost_dict, params['name_dt_fcn'][i_dt], lin_ncvx_dt_i)
                ncvx_cost += lin_ncvx_dt_i
            
        for i_ct in range(params['f_ct_dim']):

            if params['pen_ct_fcn'][i_ct] == 'abs':
                lin_ncvx_ct_i = params['w_ct_fcn'][i_ct] * np.abs(lin_ncvx_ct_cost[i_ct])
            elif params['pen_ct_fcn'][i_ct] == 'max':
                lin_ncvx_ct_i = params['w_ct_fcn'][i_ct] * np.maximum(0.0, lin_ncvx_ct_cost[i_ct])
            else:
                lin_ncvx_ct_i = params['w_ct_fcn'][i_ct] * lin_ncvx_ct_cost[i_ct]

            lin_ncvx_cost += lin_ncvx_ct_i

            if cost_dict is not None: # We compute the nonlin cost; otherwise lin
                cost_dict = dict_append(cost_dict, params['name_ct_fcn'][i_ct], lin_ncvx_ct_i)
                ncvx_cost += lin_ncvx_ct_i

        if cost_dict is not None: # We compute the nonlin cost; otherwise lin
            return ncvx_cost, cost_dict
        else:
            return lin_ncvx_cost
    
    ncvx_cost, cost_dict = compute_lin_ncvx(lin_ncvx_dt_cost=ncvx_dt_cost, 
                                            lin_ncvx_ct_cost=ncvx_ct_cost,
                                            params=params,
                                            cost_dict=cost_dict)

    n_lin_cost = nlcd + cvx_cost + ncvx_cost

    ptr_cost = 0.0
    lin_cost = 0.0
    lcd      = 0.0
    if w_tr:
        dx = X_new - X_last
        du = U_new - U_last
        ds = 0.
        if params['free_final_time']:
            ds = (S_new - S_last)

            if params['time_dil']:
                cp_sum = np.linalg.norm(ds[0,:] / params['t_f'] * params['w_ds'], axis=0)**2
                sig_time_nl = (S_new - S_last)[:, :params['K']-1]

            else:
                cp_sum = np.abs(ds / params['t_f'] * params['w_ds'])**2
                sig_time_nl = np.ones( (1, params['K'] - 1) ) * (S_new - S_last)

            S_S_nl = (ncvx_cost_grad_ct_last[4] @ sig_time_nl.T).flatten()

        else:
            cp_sum = 0.0
            S_S_nl = 0.0

        ptr_cost = w_tr * ((np.linalg.norm(dx, axis=0)**2 + np.linalg.norm(du, axis=0)**2).sum() + cp_sum.sum())

        lcd = params['w_con_dyn'] * np.linalg.norm(nu_new[:12, :].reshape(-1), 1)
        lcd += params['w_con_stt'] * np.linalg.norm(nu_new[12, :].reshape(-1), 1)

        lin_ncvx_dt_cost = ncvx_cost_grad_dt_last[0] + ncvx_cost_grad_dt_last[1] @ (X_new - X_last).flatten(order='C')

        lin_ncvx_ct_cost = (ncvx_cost_grad_ct_last[0] 
                            + (ncvx_cost_grad_ct_last[1] @ ((X_new - X_last)[:, :params['K']-1].flatten(order='C'))) 
                            + (ncvx_cost_grad_ct_last[2] @ ((U_new - U_last)[:, :params['K']-1].flatten(order='C'))) 
                            + (ncvx_cost_grad_ct_last[3] @ ((U_new - U_last)[:,  1:params['K']].flatten(order='C'))) 
                            + S_S_nl)

        lin_ncvx_cost = compute_lin_ncvx(lin_ncvx_dt_cost=lin_ncvx_dt_cost, 
                                                lin_ncvx_ct_cost=lin_ncvx_ct_cost,
                                                params=params)

        lin_cost = lcd + cvx_cost + lin_ncvx_cost + ptr_cost

    cost_dict = dict_append(cost_dict, 'ptr_cost', ptr_cost)
    
    cost_dict = dict_append(cost_dict, 'lcd', lcd)
    cost_dict = dict_append(cost_dict, 'nlcd', nlcd)
    
    cost_dict = dict_append(cost_dict, 'lc', lin_cost)
    cost_dict = dict_append(cost_dict, 'nlc', n_lin_cost)

    return cost_dict

def set_parameters(problem, **kwargs):
    for key in kwargs:
        if key in problem.param_dict.keys():
            problem.param_dict[key].value = kwargs[key]

    return problem

def prox_linear(
    params: T.Dict[str, T.Any],
    cvx_prb: T.Any,
    fcn_dict: T.Dict[str, T.Any],
) -> T.Dict[str, T.Any]:

    """
    Solves the non-convex trajectory optimization problem using Penalized Trust Region method (PTR)
    """

    X_last = params['X_last']
    U_last = params['U_last']
    S_last = params['S_last']
    w_tr   = params['w_ptr']

    last_cost  = None
    converged  = False

    cost_dict = update_cost_dict({}, Ite=0, T_Ite=0, T_Disc=0, T_SubP=0, T_J_Np = 0,)

    # Initial cost value: J(x_0, u_0, simga_0)
    X_CT = fcn_dict['int_mult_jitted'](X_last, U_last, S_last)
    nl_nu_last = (X_last[:, 1:] - X_CT[:, :, -1])

    ncvx_dt_cost = np.asarray(fcn_dict['ncvx_dt_fcn_jitted'](X_last))
    ncvx_ct_cost = np.asarray(fcn_dict['ncvx_ct_fcn_jitted'](X_CT))

    cost_dict = scp_noncvx_cost(X_new=X_last,
                                 U_new=U_last,
                                 S_new=S_last,
                                 nl_nu_new=nl_nu_last,
                                 ncvx_dt_cost=ncvx_dt_cost, 
                                 ncvx_ct_cost=ncvx_ct_cost,
                                 params=params,
                                 fcn_dict=fcn_dict,
                                 cost_dict=cost_dict)

    last_cost = cost_dict['nlc'][-1]

    cost_dict = update_cost_dict(
        cost_dict,
        D_NL=0,
        D_L=0,
        Rho=0,
        W_TR=w_tr,
        Note='Start',
        Staus='Start',
    )

    print_ite(cost_dict)

    for ite in range(params['ite']):

        t0_ite = time.time()

        t0_dicr = time.time()
        V_CT = fcn_dict['cal_disc_jitted'](X_last, U_last, S_last)
        t_dicr = time.time() - t0_dicr

        ncvx_cost_grad_dt_last = fcn_dict['ncvx_dt_fcn_grad_jitted'](X_last)
        ncvx_cost_grad_ct_last = fcn_dict['ncvx_ct_fcn_grad_jitted'](V_CT)

        # ---------------------------------------------------------------------------------

        t0_jax2np = time.time()

        V_CT = np.asarray(V_CT[:, -1, :]).T

        ncvx_cost_grad_dt_last = tuple([ np.asarray(ncvx_cost_grad_dt_last[i]) for i in range(len(ncvx_cost_grad_dt_last)) ])
        ncvx_cost_grad_ct_last = tuple([ np.asarray(ncvx_cost_grad_ct_last[i]) for i in range(len(ncvx_cost_grad_ct_last)) ])

        t_j_n = time.time() - t0_jax2np

        # ---------------------------------------------------------------------------------

        set_parameters(cvx_prb, 
                        f_bar      = V_CT[params['i0']:params['i1'], :],
                        A_bar      = V_CT[params['i1']:params['i2'], :],
                        B_bar      = V_CT[params['i2']:params['i3'], :],
                        C_bar      = V_CT[params['i3']:params['i4'], :],
                        z_bar      = V_CT[params['i5']:params['i6'], :],
                        X_last     = X_last, 
                        U_last     = U_last,
                        f_dt_last  = ncvx_cost_grad_dt_last[0],
                        gf_dt_last = ncvx_cost_grad_dt_last[1],
                        f_ct_last  = ncvx_cost_grad_ct_last[0],
                        A_ct_last  = ncvx_cost_grad_ct_last[1],
                        B_ct_last  = ncvx_cost_grad_ct_last[2],
                        C_ct_last  = ncvx_cost_grad_ct_last[3],
                        )
        
        if params['free_final_time']:

            set_parameters(cvx_prb, 
                           S_bar     = V_CT[params['i4']:params['i5'], :],
                           S_last    = S_last,
                           S_ct_last = ncvx_cost_grad_ct_last[4],
                           )

        while True:

            set_parameters(cvx_prb, w_tr=w_tr)
            
            t0_cvx = time.time()
            status, X_new, U_new, S_new, nu_new = solve_parsed_problem(cvx_prb, params)
            t_sub_prob = time.time() - t0_cvx

            X_new_CT = fcn_dict['int_mult_jitted'](X_new, U_new, S_new)
            nl_nu_new = (X_new[:, 1:] - X_new_CT[:, :, -1])

            ncvx_dt_cost = np.asarray(fcn_dict['ncvx_dt_fcn_jitted'](X_new))
            ncvx_ct_cost = np.asarray(fcn_dict['ncvx_ct_fcn_jitted'](X_new_CT))

            cost_dict = scp_noncvx_cost(X_new=X_new, U_new=U_new, S_new=S_new, 
                                         X_last=X_last, U_last=U_last, S_last=S_last, 
                                         nu_new=nu_new, nl_nu_new=nl_nu_new,
                                         ncvx_dt_cost=ncvx_dt_cost, 
                                         ncvx_ct_cost=ncvx_ct_cost,
                                         ncvx_cost_grad_dt_last=ncvx_cost_grad_dt_last, 
                                         ncvx_cost_grad_ct_last=ncvx_cost_grad_ct_last,
                                         w_tr=w_tr, params=params, 
                                         fcn_dict=fcn_dict,
                                         cost_dict=cost_dict,
                                        )

            delta_J = last_cost - cost_dict['nlc'][-1]
            delta_L = last_cost - cost_dict['lc'][-1]
            rho = delta_J / delta_L
            
            if (ite == 0) or not(params['adaptive_step']):
                X_last = X_new.copy()
                U_last = U_new.copy()
                S_last = S_new.copy()

                last_cost = cost_dict['nlc'][-1]
                note = 'First'
                break

            else:

                if delta_L < -1e-6:
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxx Predicted change is negative xxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('delta_L', delta_L)
                    print('w_tr', w_tr)
                    w_tr = w_tr * 2.0
                    note = '!!!'
                    break

                if delta_L < (params['ptr_term']) or (w_tr >= 1e8):
                    converged = True
                    print('delta_L', delta_L)
                    note = 'Conv'
                    break
                
                if rho <= params['r0']:
                    w_tr = w_tr * 2
                    note = 'Rjct'

                    cost_dict = update_cost_dict(
                        cost_dict,
                        Ite    = ite + 1,
                        T_Ite  = time.time() - t0_ite,
                        T_Disc = t_dicr,
                        T_SubP = t_sub_prob,
                        T_J_Np = t_j_n,
                        D_NL   = delta_J,
                        D_L    = delta_L,
                        Rho    = rho,
                        W_TR   = w_tr,
                        Note   = note,
                        Staus  = status
                    )

                    print_ite(cost_dict)

                else:
                    X_last = X_new.copy()
                    U_last = U_new.copy()
                    S_last = copy.deepcopy(S_new)
                    last_cost = cost_dict['nlc'][-1]

                    if rho < params['r1']:
                        w_tr = w_tr * 2.0
                        note = 'Decr'
                    elif params['r2'] <= rho:
                        w_tr = np.maximum(w_tr / 2.0, params['w_ptr_min'])
                        note = 'Incr'
                    else:
                        note = 'Go'
                    break

        if converged:
            print(f'Converged after {ite + 1} iterations.')
            break
        else:
            cost_dict = update_cost_dict(
                cost_dict,
                Ite    = ite + 1,
                T_Ite  = time.time() - t0_ite,
                T_Disc = t_dicr,
                T_SubP = t_sub_prob,
                T_J_Np = t_j_n,
                D_NL   = delta_J,
                D_L    = delta_L,
                Rho    = rho,
                W_TR   = w_tr,
                Note   = note    if params['adaptive_step'] else 'fixed w_tr',
                Staus  = status
            )

            print_ite(cost_dict)

    if not converged:
        print("Reached max iterations without convergence.")
        
    cost_dict['X_new'] = X_last
    cost_dict['U_new'] = U_last
    cost_dict['S_new'] = S_last

    return cost_dict
