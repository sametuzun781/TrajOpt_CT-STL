"""JAX model linearization and objective evaluation used by the SCP loop."""
import jax
import jax.numpy as jnp

__all__ = ["linearize_trajectory", "evaluate_costs"]


def compile_array_kernels(X, U, S, params, functions, model):
    """Compile shared kernels; download cost scalars as one small array."""
    linearize = jax.jit(lambda X, U, S: linearize_trajectory(X, U, S, params, functions))
    linearize = linearize.lower(X, U, S).compile()
    sample = linearize(X, U, S)
    jax.block_until_ready(sample)
    zero = jnp.zeros((params['n_x'], params['K']-1))
    weight = jnp.asarray(params['w_ptr'], dtype=jnp.float64)
    def costs(X, U, S, nu, values, weight):
        return evaluate_costs(X, U, S, nu, values, weight, params, functions, model)
    names = tuple(sorted(jax.eval_shape(costs, X, U, S, zero, sample, weight)))
    def packed(X, U, S, nu, values, weight):
        result = costs(X, U, S, nu, values, weight)
        return jnp.stack([result[name] for name in names])
    evaluate = jax.jit(packed).lower(X, U, S, zero, sample, weight).compile()
    return linearize, evaluate, names


def linearize_trajectory(X, U, S, params, functions):
    """Use the repository's original RK4 and derivative functions, without NumPy."""
    endpoint = getattr(functions['cal_disc_jitted'], 'icore_terminal_discretizer', None)
    zero_ct = getattr(functions['ncvx_ct_fcn_grad_jitted'], 'icore_zero_continuous_cost', False)
    if endpoint is not None and zero_ct:
        terminal = endpoint(X, U, S).T
        n, k, nx, nu = params['f_ct_dim'], params['K'] - 1, params['n_x'], params['n_u']
        ct = (jnp.zeros((n,), X.dtype), jnp.zeros((n, nx*k), X.dtype),
              jnp.zeros((n, nu*k), X.dtype), jnp.zeros((n, nu*k), X.dtype),
              jnp.zeros((n, k), X.dtype))
    else:
        trajectory = functions['cal_disc_jitted'](X, U, S)
        terminal = trajectory[:, -1, :].T
        ct = functions['ncvx_ct_fcn_grad_jitted'](trajectory)
    dt = functions['ncvx_dt_fcn_grad_jitted'](X)
    inner_value, inner_jacobian = functions['ncvx_cvx_fcn_grad_jitted'](X)
    inner = functions['ncvx_cvx_fcn'](X, params)
    _, outer_jacobian = functions['ncvx_smth_fcn_grad_jitted'](inner)
    comp = functions['ncvx_comp_fcn_grad_jitted'](X)
    positive = jnp.maximum(outer_jacobian, 0.)
    values = dict(
        f_bar=terminal[params['i0']:params['i1']],
        A_bar=terminal[params['i1']:params['i2']],
        B_bar=terminal[params['i2']:params['i3']],
        C_bar=terminal[params['i3']:params['i4']],
        z_bar=terminal[params['i5']:params['i6']],
        X_last=X, U_last=U,
        f_dt_last=dt[0], gf_dt_last=dt[1],
        f_ct_last=ct[0], A_ct_last=ct[1], B_ct_last=ct[2], C_ct_last=ct[3],
        f_comp_last=comp[0], gf_comp_last=comp[1],
        gf_comp_neg_last=jnp.minimum(outer_jacobian, 0.) @ inner_jacobian,
        gf_smth_pos_last=positive,
        gf_smth_pos_cvx_last=positive @ inner_value,
    )
    if params['free_final_time']:
        values.update(S_last=S, S_bar=terminal[params['i4']:params['i5']], S_ct_last=ct[4])
    return values


def evaluate_costs(X, U, S, nu, values, weight, params, functions, model):
    """Numerical counterpart of scp_noncvx_cost; returns the same cost columns."""
    continuous = functions['int_mult_jitted'](X, U, S)
    nonlinear_defect = X[:, 1:] - continuous[:, :, -1]
    dt = functions['ncvx_dt_fcn_jitted'](X)
    ct = functions['ncvx_ct_fcn_jitted'](continuous)
    comp = functions['ncvx_comp_fcn_jitted'](X)
    dx, du = X - values['X_last'], U - values['U_last']
    dt_linear = values['f_dt_last'] + values['gf_dt_last'] @ dx.ravel()
    ct_linear = (values['f_ct_last'] + values['A_ct_last'] @ dx[:, :-1].ravel()
                 + values['B_ct_last'] @ du[:, :-1].ravel()
                 + values['C_ct_last'] @ du[:, 1:].ravel())
    comp_linear = values['f_comp_last']
    if params['ncvx_solver'] == 'pl':
        comp_linear += values['gf_comp_last'] @ dx.ravel()
    else:
        inner = functions['ncvx_cvx_fcn'](X, params).ravel()
        comp_linear += (values['gf_comp_neg_last'] @ dx.ravel()
                        + values['gf_smth_pos_last'] @ inner - values['gf_smth_pos_cvx_last'])

    convex = jnp.array(0.)
    duration_step = jnp.array(0.)
    costs = {}
    if params['free_final_time']:
        costs['S_cost'] = params['w_S'] * jnp.sum(jnp.abs(S)) / params['t_f']
        convex += costs['S_cost']
        ds = S - values['S_last']
        duration_step = jnp.sum((ds[0] / params['t_f'] * params['w_ds']) ** 2)
        ct_linear += (values['S_ct_last'] @ ds.T).ravel()
    if model == 'quad_flight':
        if params['add_inp']:
            convex += params['w_inp'] * jnp.sum(U[3] ** 2)
        if params['add_inp_trq']:
            costs['input_cost'] = params['w_inp_trq'] * jnp.sum(U[:3] ** 2)
            convex += costs['input_cost']
        if params['add_elv_rate']:
            costs['elv_rate_cost'] = params['w_elv_r'] * jnp.sum(X[5] ** 2)
            convex += costs['elv_rate_cost']
    n_dyn = params['n_dyn']
    costs['dyn_cost'] = params['w_con_dyn'] * jnp.sum(jnp.abs(nonlinear_defect[:n_dyn]))
    costs['ctc_cost'] = params['w_con_ctc'] * jnp.sum(jnp.abs(nonlinear_defect[n_dyn:]))
    nonlinear_penalty = jnp.array(0.)
    linear_penalty = jnp.array(0.)
    for actual, predicted, weights, modes, names in (
        (dt, dt_linear, params['w_dt_fcn'], params['pen_dt_fcn'], params['name_dt_fcn']),
        (ct, ct_linear, params['w_ct_fcn'], params['pen_ct_fcn'], params['name_ct_fcn']),
        (comp, comp_linear, params['w_comp_fcn'], ['none'] * params['f_comp_dim'], params['name_comp_fcn']),
    ):
        for index, mode in enumerate(modes):
            real, model_value = actual[index], predicted[index]
            if mode == 'abs':
                real, model_value = jnp.abs(real), jnp.abs(model_value)
            elif mode == 'max':
                real, model_value = jnp.maximum(real, 0.), jnp.maximum(model_value, 0.)
            costs[names[index]] = weights[index] * real
            nonlinear_penalty += weights[index] * real
            linear_penalty += weights[index] * model_value
    costs['ptr_cost'] = weight * (jnp.sum(jnp.sum(dx ** 2, axis=0) + jnp.sum(du ** 2, axis=0)) + duration_step)
    costs['lcd'] = params['w_con_dyn'] * jnp.sum(jnp.abs(nu[:n_dyn]))
    costs['lcd'] += params['w_con_ctc'] * jnp.sum(jnp.abs(nu[n_dyn:]))
    costs['nlcd'] = costs['dyn_cost'] + costs['ctc_cost']
    costs['lc'] = costs['lcd'] + convex + linear_penalty + costs['ptr_cost']
    costs['nlc'] = costs['nlcd'] + convex + nonlinear_penalty
    return costs
