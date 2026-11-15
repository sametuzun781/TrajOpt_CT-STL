import typing as T
import numpy as np
from solvers.optional_cvxpy import cp

def parse_convex_problem(params : T.Dict[str, T.Any],
                         fcn_dict : T.Dict[str, T.Any]):
       
    """Build the convex subproblem with the common positive objective scale."""

    n_x = params['n_x']
    n_u = params['n_u']
    K = params['K']

    n_dt = params['f_dt_dim']
    n_ct = params['f_ct_dim']

    n_cvx = params['f_cvx_dim']
    n_comp = params['f_comp_dim']

    var = {}
    par = {}

    var['X']  = cp.Variable((n_x, K), name='X')
    var['U']  = cp.Variable((n_u, K), name='U')
    var['nu'] = cp.Variable((n_x, K - 1), name='nu')
    var['dx'] = cp.Variable((n_x, K), name='dx')
    var['du'] = cp.Variable((n_u, K), name='du')

    par['f_bar'] = cp.Parameter((n_x, K - 1), name='f_bar')
    par['A_bar'] = cp.Parameter((n_x*n_x, K - 1), name='A_bar')
    par['B_bar'] = cp.Parameter((n_x*n_u, K - 1), name='B_bar')
    par['C_bar'] = cp.Parameter((n_x*n_u, K - 1), name='C_bar')
    par['z_bar'] = cp.Parameter((n_x, K - 1), name='z_bar')

    par['X_last'] = cp.Parameter((n_x, K), name='X_last')
    par['U_last'] = cp.Parameter((n_u, K), name='U_last')

    par['w_tr'] = cp.Parameter(nonneg=True, name='w_tr')

    par['f_dt_last'] = cp.Parameter(n_dt, name='f_dt_last')
    par['gf_dt_last'] = cp.Parameter((n_dt, n_x*K), name='gf_dt_last')

    par['f_ct_last'] = cp.Parameter(n_ct, name='f_ct_last')
    par['A_ct_last'] = cp.Parameter((n_ct, n_x*(K-1)), name='A_ct_last')
    par['B_ct_last'] = cp.Parameter((n_ct, n_u*(K-1)), name='B_ct_last')
    par['C_ct_last'] = cp.Parameter((n_ct, n_u*(K-1)), name='C_ct_last')

    # -------------------------------------------
    # --------------- prox-convex ---------------
    # -------------------------------------------
    par['gf_comp_neg_last'] = cp.Parameter((n_comp, n_x*K), nonneg=True, name='gf_comp_neg_last')
    par['gf_smth_pos_last'] = cp.Parameter((n_comp, n_cvx*K), nonneg=True, name='gf_smth_pos_last')
    par['gf_smth_pos_cvx_last'] = cp.Parameter(n_comp, name='gf_smth_pos_cvx_last')

    par['f_comp_last'] = cp.Parameter(n_comp, name='f_comp_last')
    par['gf_comp_last'] = cp.Parameter((n_comp, n_x*K), name='gf_comp_last')
    # -------------------------------------------
    # --------------- prox-convex ---------------
    # -------------------------------------------

    if params['free_final_time']:
        par['S_bar'] = cp.Parameter((n_x, K - 1), name='S_bar')
        par['S_ct_last'] = cp.Parameter((n_ct, 1*(K-1)), name='S_ct_last')

        if (params['time_dil']):
            var['S'] = cp.Variable((1, K - 1), nonneg=True, name='S')
            par['S_last'] = cp.Parameter((1, K - 1), nonneg=True, name='S_last')
            var['ds'] = cp.Variable((1, K - 1), name='ds')

            sig_matrix = np.ones( (n_x, 1) ) @ var['ds']
            ds_sum = cp.sum_squares(var['ds'][0, :] / params['t_f'] * params['w_ds'])

            sig_ct = var['ds']
        else:
            var['S'] = cp.Variable(nonneg=True, name='S')
            par['S_last'] = cp.Parameter(nonneg=True, name='S_last')
            var['ds'] = cp.Variable(nonneg=True, name='ds')

            sig_matrix = np.ones((n_x, K - 1)) * var['ds']
            ds_sum = cp.sum(cp.abs(var['ds'] / params['t_f'] * params['w_ds'])**2)

            sig_ct = np.ones((1, K - 1)) * var['ds']

        S_S = cp.multiply(par['S_bar'], sig_matrix)
        S_S_ct = (par['S_ct_last'] @ sig_ct.T).flatten()

    else:
        var['S'] = np.array(0, dtype=np.float64)
        par['S_last'] = np.array(0, dtype=np.float64)
        var['ds'] = np.array(0, dtype=np.float64)

        ds_sum = 0.
        S_S = np.zeros( (n_x, K - 1) )
        S_S_ct = np.zeros(n_ct)

    # ---------------------------------------------------------------------------------

    cost = par['w_tr'] * (cp.sum_squares(var['dx']) + cp.sum_squares(var['du']) + ds_sum) # Trust region
    cost += params['w_con_dyn'] * cp.sum(cp.abs(var['nu'][:params['n_dyn'], :]))
    cost += params['w_con_ctc'] * cp.sum(cp.abs(var['nu'][params['n_dyn']:, :]))

    # ---------------------------------------------------------------------------------

    lin_dt = par['f_dt_last'] + par['gf_dt_last'] @ cp.vec(var['dx'], order='C')

    for i_dt in range(n_dt):
        if params['pen_dt_fcn'][i_dt] == 'abs':
            cost += params['w_dt_fcn'][i_dt] * cp.abs(lin_dt[i_dt])
        elif params['pen_dt_fcn'][i_dt] == 'max':
            cost += params['w_dt_fcn'][i_dt] * cp.pos(lin_dt[i_dt])
        else:
            cost += params['w_dt_fcn'][i_dt] * lin_dt[i_dt]

    # ---------------------------------------------------------------------------------

    lin_ct = (par['f_ct_last'] 
              + (par['A_ct_last'] @ cp.vec(var['dx'][:, :K-1], order='C')) 
              + (par['B_ct_last'] @ cp.vec(var['du'][:, :K-1], order='C')) 
              + (par['C_ct_last'] @ cp.vec(var['du'][:,  1:K], order='C')) 
              + S_S_ct)

    for i_ct in range(n_ct):
        if params['pen_ct_fcn'][i_ct] == 'abs':
            cost += params['w_ct_fcn'][i_ct] * cp.abs(lin_ct[i_ct])
        elif params['pen_ct_fcn'][i_ct] == 'max':
            cost += params['w_ct_fcn'][i_ct] * cp.pos(lin_ct[i_ct])
        else:
            cost += params['w_ct_fcn'][i_ct] * lin_ct[i_ct]

    # -------------------------------------------
    # --------------- prox-convex ---------------
    # -------------------------------------------

    lin_comp = par['f_comp_last']
    if params['ncvx_solver'] == 'pl':
        lin_comp += par['gf_comp_last'] @ cp.vec(var['dx'], order='C')
    elif params['ncvx_solver'] == 'pcx':
        print("PCS IS ACTIVE")
        y_cvx = fcn_dict['ncvx_cvx_cp_fcn'](var['X'], params)
        lin_comp += par['gf_comp_neg_last'] @ cp.vec(var['dx'], order='C')
        lin_comp += par['gf_smth_pos_last'] @ cp.vec(y_cvx, order='C') - par['gf_smth_pos_cvx_last']
    else:
        raise ValueError(f"Unknown ncvx_solver: {params['ncvx_solver']}")

    for i_comp in range(n_comp):
        cost += params['w_comp_fcn'][i_comp] * lin_comp[i_comp]
    # -------------------------------------------
    # --------------- prox-convex ---------------
    # -------------------------------------------

    cost += fcn_dict['cvx_cost_fcn'](var['X'], var['U'], var['S'], 
                                     var['nu'], params, npy=False)

    # ---------------------------------------------------------------------------------

    constraints = [
        par['X_last'][:, k + 1] + var['dx'][:, k + 1] == par['f_bar'][:, k]
        + cp.reshape(par['A_bar'][:, k], (n_x, n_x), order='C') @ var['dx'][:, k]
        + cp.reshape(par['B_bar'][:, k], (n_x, n_u), order='C') @ var['du'][:, k]
        + cp.reshape(par['C_bar'][:, k], (n_x, n_u), order='C') @ var['du'][:, k+1]
        + S_S[:, k]
        + var['nu'][:, k]
        for k in range(K - 1)
    ]

    constraints += [
        var['dx'] == var['X'] - par['X_last'],
        var['du'] == var['U'] - par['U_last'],
    ]

    if params['free_final_time']:
        constraints += [
            var['ds'] == (var['S'] - par['S_last'])
        ]

    constraints += fcn_dict['cvx_cons_fcn'](var['X'], var['U'], var['S'], params)
    
    # Create the optimization problem
    scale = max(1., params['w_con_dyn'], params['w_con_ctc'],
                *params['w_dt_fcn'], *params['w_ct_fcn'])
    cvx_prb = cp.Problem(cp.Minimize(cost / scale), constraints)
    cvx_prb.objective_scale = scale

    # print(cvx_prb.is_dcp(dpp=True))

    return cvx_prb

def solve_parsed_problem(cvx_prb: T.Any, 
                         params: T.Dict[str, T.Any]):
    
    if hasattr(cvx_prb, 'solve_ipm'):
        return cvx_prb.solve_ipm()

    cvx_prb.solve(solver='QOCO', warm_start=False,
                  **params.get('qoco_options', {}))
    if cvx_prb.status != cp.OPTIMAL:
        raise RuntimeError(f'QOCO did not solve the convex subproblem: {cvx_prb.status}')

    X_new = cvx_prb.var_dict['X'].value
    U_new = cvx_prb.var_dict['U'].value
    if params['free_final_time']:
        S_new = cvx_prb.var_dict['S'].value
    else:
        S_new = np.array(0, dtype=np.float64)
    
    nu_new = cvx_prb.var_dict['nu'].value

    return cvx_prb.status, X_new, U_new, S_new, nu_new
