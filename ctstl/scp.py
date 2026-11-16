import time
import copy
import typing as T
import numpy as np
import jax
import jax.numpy as jnp
from ctstl.array_io import to_host, to_device
from ctstl.utils import dict_append, update_cost_dict, print_ite
from solvers.cvx import solve_parsed_problem

def scp_noncvx_cost(
    X_new                    : np.ndarray = None,
    U_new                    : np.ndarray = None,
    S_new                    : np.ndarray = None,
    X_last                   : np.ndarray = None,
    U_last                   : np.ndarray = None,
    S_last                   : np.ndarray = None,
    nu_new                   : np.ndarray = None,
    nl_nu_new                : np.ndarray = None,
    ncvx_dt_cost             : np.ndarray = None,
    ncvx_ct_cost             : np.ndarray = None,
    ncvx_comp_cost           : np.ndarray = None,
    ncvx_cost_grad_dt_last   : T.Tuple[np.ndarray, np.ndarray] = None,
    ncvx_cost_grad_ct_last   : T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None,
    ncvx_cost_grad_cvx_last  : T.Tuple[np.ndarray, np.ndarray] = None,
    ncvx_cost_grad_smth_last : T.Tuple[np.ndarray, np.ndarray] = None,
    ncvx_cost_grad_comp_last : T.Tuple[np.ndarray, np.ndarray] = None,
    w_tr                     : float = None,
    params                   : T.Dict[str, T.Any] = None,
    fcn_dict                 : T.Dict[str, T.Any] = None,
    cost_dict                : T.Dict[str, T.Any] = None,
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
    nl_dyn = params['w_con_dyn'] * np.linalg.norm(nl_nu_new[:params['n_dyn'], :].reshape(-1), 1)
    nl_ctc = params['w_con_ctc'] * np.linalg.norm(nl_nu_new[params['n_dyn']:, :].reshape(-1), 1)
    nlcd   = nl_dyn + nl_ctc
    
    # 2) Convex cost
    cvx_cost, cost_dict = fcn_dict['cvx_cost_fcn'](
        X_new, U_new, S_new, nl_nu_new, params, npy=True, cost_dict=cost_dict
    )

    # 3) Helper to compute linearized nonconvex penalty
    def compute_lin_ncvx( 
        lin_ncvx_dt_cost: np.ndarray,
        lin_ncvx_ct_cost: np.ndarray,
        lin_ncvx_comp_cost: np.ndarray,
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

        # -------------------------------------------
        # --------------- prox-convex ---------------
        # -------------------------------------------
        for i_comp in range(params['f_comp_dim']):

            lin_ncvx_comp_i = params['w_comp_fcn'][i_comp] * lin_ncvx_comp_cost[i_comp]
            lin_ncvx_cost += lin_ncvx_comp_i

            if cost_dict is not None: # We compute the nonlin cost; otherwise lin
                cost_dict = dict_append(cost_dict, params['name_comp_fcn'][i_comp], lin_ncvx_comp_i)
                ncvx_cost += lin_ncvx_comp_i
        # -------------------------------------------
        # --------------- prox-convex ---------------
        # -------------------------------------------

        if cost_dict is not None: # We compute the nonlin cost; otherwise lin
            return ncvx_cost, cost_dict
        else:
            return lin_ncvx_cost
    
    ncvx_cost, cost_dict = compute_lin_ncvx(lin_ncvx_dt_cost=ncvx_dt_cost, 
                                            lin_ncvx_ct_cost=ncvx_ct_cost,
                                            lin_ncvx_comp_cost=ncvx_comp_cost,
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

        ptr_cost = w_tr * ((np.linalg.norm(dx, axis=0)**2 + np.linalg.norm(du, axis=0)**2).sum() + np.sum(cp_sum))

        lcd = params['w_con_dyn'] * np.linalg.norm(nu_new[:params['n_dyn'], :].reshape(-1), 1)
        lcd += params['w_con_ctc'] * np.linalg.norm(nu_new[params['n_dyn']:, :].reshape(-1), 1)

        lin_ncvx_dt_cost = ncvx_cost_grad_dt_last[0] + ncvx_cost_grad_dt_last[1] @ (X_new - X_last).flatten(order='C')

        lin_ncvx_ct_cost = (ncvx_cost_grad_ct_last[0] 
                            + (ncvx_cost_grad_ct_last[1] @ ((X_new - X_last)[:, :params['K']-1].flatten(order='C'))) 
                            + (ncvx_cost_grad_ct_last[2] @ ((U_new - U_last)[:, :params['K']-1].flatten(order='C'))) 
                            + (ncvx_cost_grad_ct_last[3] @ ((U_new - U_last)[:,  1:params['K']].flatten(order='C'))) 
                            + S_S_nl)

        lin_ncvx_comp_cost = 0.0
        lin_ncvx_comp_cost += ncvx_cost_grad_comp_last[0]
        if params['ncvx_solver'] == 'pl':
            lin_ncvx_comp_cost += ncvx_cost_grad_comp_last[1] @ (X_new - X_last).flatten(order='C')
        elif params['ncvx_solver'] == 'pcx':
            cvx = fcn_dict['ncvx_cvx_fcn'](X_new, params) # n_cvx x K
            cvx_last = fcn_dict['ncvx_cvx_fcn'](X_last, params) # n_cvx x K

            d_cvx = np.asarray(cvx - cvx_last).flatten(order='C') # n_cvx * K
            d_lin = ncvx_cost_grad_cvx_last[1] @ (X_new - X_last).flatten(order='C') # n_cvx * K

            lin_ncvx_comp_cost += np.minimum(0, ncvx_cost_grad_smth_last[1]) @ d_lin
            lin_ncvx_comp_cost += np.maximum(0, ncvx_cost_grad_smth_last[1]) @ d_cvx
        else:
            raise ValueError(f"Unknown ncvx_solver: {params['ncvx_solver']}")

        lin_ncvx_cost = compute_lin_ncvx(lin_ncvx_dt_cost=lin_ncvx_dt_cost, 
                                         lin_ncvx_ct_cost=lin_ncvx_ct_cost,
                                         lin_ncvx_comp_cost=lin_ncvx_comp_cost,
                                         params=params)

        lin_cost = lcd + cvx_cost + lin_ncvx_cost + ptr_cost

    cost_dict = dict_append(cost_dict, 'ptr_cost', ptr_cost)
    
    cost_dict = dict_append(cost_dict, 'lcd', lcd)
    cost_dict = dict_append(cost_dict, 'nlcd', nlcd)
    
    cost_dict = dict_append(cost_dict, 'lc', lin_cost)
    cost_dict = dict_append(cost_dict, 'nlc', n_lin_cost)

    return cost_dict

def check_cvxpy_scaling(params_dict, small_tol=1e-5, large_tol=1e4):
    """
    Scans a dictionary of arrays/matrices for numerical scaling issues 
    that commonly cause CVXPY solvers to fail.
    """
    
    issues_found = False
    for name, val in params_dict.items():
        if val is None:
            continue
            
        arr = np.asarray(val, dtype=float)
        
        # 1. Check for NaNs and Infs (Guaranteed to crash CVXPY)
        if np.isnan(arr).any():
            print(f"  WARNING: '{name}' contains NaN values!")
            issues_found = True
        if np.isinf(arr).any():
            print(f"  WARNING: '{name}' contains Inf values!")
            issues_found = True
            
        # 2. Check for extremely large values
        large_mask = np.abs(arr) > large_tol
        if np.any(large_mask):
            large_vals = arr[large_mask]
            print(f"   LARGE VALUES in '{name}':")
            print(f"   -> Found {len(large_vals)} elements > {large_tol}")
            print(f"   -> Max absolute value: {np.max(np.abs(large_vals)):.2e}")
            # Print the first few specific indices and values
            indices = np.argwhere(large_mask)
            for i in range(min(3, len(indices))):
                idx = tuple(indices[i])
                print(f"      @ index {idx} = {arr[idx]:.2e}")
            issues_found = True
            
        # 3. Check for vanishingly small (but non-zero) values
        # We ignore pure zeros, as sparsity is fine for solvers.
        small_mask = (np.abs(arr) < small_tol) & (arr != 0)
        if np.any(small_mask):
            small_vals = arr[small_mask]
            print(f"   SMALL VALUES in '{name}':")
            print(f"   -> Found {len(small_vals)} non-zero elements < {small_tol}")
            print(f"   -> Min absolute non-zero value: {np.min(np.abs(small_vals)):.2e}")
            # Print the first few specific indices and values
            indices = np.argwhere(small_mask)
            for i in range(min(3, len(indices))):
                idx = tuple(indices[i])
                print(f"      @ index {idx} = {arr[idx]:.2e}")
            issues_found = True

def set_parameters(problem, **kwargs):
    for key in kwargs:
        if key in problem.param_dict.keys():
            problem.param_dict[key].value = kwargs[key]

    return problem

def prox_convex(
    params: T.Dict[str, T.Any],
    cvx_prb: T.Any,
    fcn_dict: T.Dict[str, T.Any],
) -> T.Dict[str, T.Any]:

    """
    Solves the non-convex trajectory optimization problem using Penalized Trust Region method (PTR)
    """

    call_started = time.perf_counter()
    # Python makes scalar decisions; trajectories/Jacobians stay on the device.
    from ctstl.full_scp import compile_array_kernels
    model = fcn_dict['cvx_cons_fcn'].__module__.split('.')[1]
    optional = hasattr(cvx_prb, 'solve_ipm')
    compilation_before = getattr(cvx_prb, 'compile_seconds', 0.0)
    (X_last, U_last, S_last), initial_upload = to_device(tuple(
        params[name] for name in ('X_last', 'U_last', 'S_last')))
    w_tr = params['w_ptr']
    converged = False
    total_convex_seconds = 0.0
    total_solution_decode_seconds = 0.0
    convex_solve_count = 0
    solvers_used = set()
    trial_timings = []
    previous_weight = weight_value = None

    # Reuse the very same array equations as full-JIT; no new model/solver.
    if not hasattr(cvx_prb, '_array_linearize'):
        started = time.perf_counter()
        cvx_prb._array_linearize, cvx_prb._array_costs, cvx_prb._cost_names = compile_array_kernels(
            X_last, U_last, S_last, params, fcn_dict, model)
        array_compilation = time.perf_counter() - started
        print(f'Array-kernel preparation: {1000.0 * array_compilation:.3f} ms', flush=True)
    else:
        array_compilation = 0.0
    linearize, evaluate = cvx_prb._array_linearize, cvx_prb._array_costs
    cost_names = (['S_cost'] if params['free_final_time'] else [])
    if model == 'quad_flight':
        if params['add_inp_trq']:
            cost_names.append('input_cost')
        if params['add_elv_rate']:
            cost_names.append('elv_rate_cost')
    cost_names += ['dyn_cost', 'ctc_cost']
    cost_names += params['name_dt_fcn'] + params['name_ct_fcn'] + params['name_comp_fcn']
    cost_names += ['ptr_cost', 'lcd', 'nlcd', 'lc', 'nlc']
    cost_names = tuple(dict.fromkeys(cost_names))

    execution_started = time.perf_counter()
    initial_values = linearize(X_last, U_last, S_last)
    initial = evaluate(X_last, U_last, S_last,
                       jnp.zeros((params['n_x'], params['K'] - 1)),
                       initial_values, jnp.asarray(0.0))
    jax.block_until_ready(initial)
    initial, initial_download = to_host(initial)
    initial = dict(zip(cvx_prb._cost_names, initial))
    initial['lc'] = 0.0  # No convex trial in the original initial row.
    cost_dict = update_cost_dict({}, Ite=0, T_Ite=0., T_Disc=0.,
                                 T_SubP=0., T_J_Np=initial_download, T_Np_J=initial_upload)
    for name in cost_names:
        cost_dict = dict_append(cost_dict, name, float(initial[name]))

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

        t0_dicr = time.perf_counter()
        cvx_params = linearize(X_last, U_last, S_last)
        jax.block_until_ready(cvx_params)
        t_dicr = time.perf_counter() - t0_dicr
        t_j_n = 0.0
        t_n_j = 0.0

        if optional:
            parameters = cvx_params   # Our IPM / chain builder needs no host copy.
        else:
            # Original CVXPY expression graph is unchanged. Transfer only its
            # parameters, not the complete dense continuous-time trajectory.
            required = {key: value for key, value in cvx_params.items()
                        if key in cvx_prb.param_dict}
            parameters, t_j_n = to_host(required)
        parameter_started = time.perf_counter()
        set_parameters(cvx_prb, **parameters)
        parameter_seconds = time.perf_counter() - parameter_started

        while True:

            if previous_weight != w_tr:
                weight_value, upload = to_device(np.asarray(w_tr, dtype=np.float64))
                t_n_j += upload
                previous_weight = w_tr
            t0_cvx = time.perf_counter()
            set_parameters(cvx_prb, w_tr=weight_value if optional else w_tr)
            status, X_new, U_new, S_new, nu_new = solve_parsed_problem(cvx_prb, params)
            if optional:
                record = cvx_prb.records[-1]
                total_solution_decode_seconds += record['solution_decode_seconds']
                t_sub_prob = record['seconds'] + parameter_seconds
                t_j_n += record['jax_to_numpy_seconds']
                t_n_j += record['numpy_to_jax_seconds']
                solvers_used.add(record['solver'])
            else:
                t_sub_prob = time.perf_counter() - t0_cvx + parameter_seconds
                stats = cvx_prb.solver_stats
                solvers_used.add(str(stats.solver_name) if stats else 'unknown')
                (X_new, U_new, S_new, nu_new), upload = to_device(
                    (X_new, U_new, S_new, nu_new))
                t_n_j += upload
            total_convex_seconds += t_sub_prob
            convex_solve_count += 1

            costs = evaluate(X_new, U_new, S_new, nu_new, cvx_params, weight_value)
            jax.block_until_ready(costs)   # Nonlinear work belongs to T-Ite, NOT transfer.
            costs, download = to_host(costs)  # Only scalar cost/log values.
            costs = dict(zip(cvx_prb._cost_names, costs))
            t_j_n += download
            for name in cost_names:
                cost_dict = dict_append(cost_dict, name, float(costs[name]))
            trial_timings.append(dict(
                iteration=ite + 1, T_Disc=t_dicr, T_SubP=t_sub_prob,
                T_J_Np=t_j_n, T_Np_J=t_n_j))

            delta_J = last_cost - cost_dict['nlc'][-1]
            delta_L = last_cost - cost_dict['lc'][-1]
            rho = delta_J / delta_L
            
            if (ite == 0) or not(params['adaptive_step']):
                X_last = X_new
                U_last = U_new
                S_last = S_new

                last_cost = cost_dict['nlc'][-1]
                note = 'First'

                if delta_L < (params['ptr_term']) or (w_tr >= 1e7):
                    converged = True
                    print('delta_L', delta_L)
                    note = 'Conv'

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

                if delta_L < (params['ptr_term']) or (w_tr >= 1e7):
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
                        T_Np_J = t_n_j,
                        D_NL   = delta_J,
                        D_L    = delta_L,
                        Rho    = rho,
                        W_TR   = w_tr,
                        Note   = note,
                        Staus  = status
                    )

                    print_ite(cost_dict)
                    t0_ite = time.time()
                    t_dicr = t_j_n = t_n_j = parameter_seconds = 0.0

                else:
                    X_last = X_new
                    U_last = U_new
                    S_last = S_new
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
                T_Np_J = t_n_j,
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
        
    (X_out, U_out, S_out), final_download = to_host((X_last, U_last, S_last))
    compilation = array_compilation + getattr(cvx_prb, 'compile_seconds', 0.0) - compilation_before
    cost_dict.update(
        total_convex_seconds=total_convex_seconds,
        total_solution_decode_seconds=total_solution_decode_seconds,
        total_discretization_seconds=sum(row['T_Disc'] for row in trial_timings),
        total_jax_to_numpy_seconds=initial_download + final_download + sum(row['T_J_Np'] for row in trial_timings),
        total_numpy_to_jax_seconds=initial_upload + sum(row['T_Np_J'] for row in trial_timings),
        initial_jax_to_numpy_seconds=initial_download,
        initial_numpy_to_jax_seconds=initial_upload,
        final_jax_to_numpy_seconds=final_download,
        compilation_seconds=compilation,
        total_scp_seconds=time.perf_counter() - call_started - compilation,
        convex_solve_count=convex_solve_count, trial_timings=trial_timings,
        solvers_used=sorted(solvers_used), converged=converged,
        X_new=X_out, U_new=U_out, S_new=S_out,
    )
    return cost_dict
