"""Translate the CT-STL convex subproblem and call the selected solver."""
from functools import partial
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from solvers import ChainSolver
from solvers.chain_model import ChainQP
from ctstl.array_io import to_device


class OptionalConvexProblem:
    """Keep the parameter/solution interface consumed by cvx.py and scp.py."""

    def __init__(self, params, functions, backend='jax_icore', *,
                 tolerance=1e-8, qoco_options=None):
        if backend not in ('jax_icore', 'qoco_native', 'icore_c', 'icore_cpp'):
            raise ValueError('Unknown solver backend.')
        self.params = params.copy()
        self.backend = backend
        self.records = []
        self._build_compile_seconds = 0.0
        self._compiled_build = None
        self._compiled_decode = None
        self.solver = ChainSolver(backend, tolerance=tolerance,
                                  qoco_options=dict(qoco_options or {}))
        model = functions['cvx_cons_fcn'].__module__.split('.')[1]
        self.model = model
        if model not in ('di_until', 'di_until_aug', 'di_eventually',
                         'di_always_stc', 'quad_flight'):
            raise NotImplementedError('This model needs its own convex adapter.')
        if params['free_final_time'] and not params['time_dil']:
            raise NotImplementedError('Use fixed time or per-interval time dilation.')
        if any(mode in ('abs', 'max') for mode in params['pen_ct_fcn']):
            raise NotImplementedError('Nonlocal CT abs/hinge terms need an extended adapter.')
        if params['free_final_time'] and params['min_S'] < 0:
            raise ValueError('The time-cost adapter requires nonnegative durations.')
        self.scale = max(1., params['w_con_dyn'], params['w_con_ctc'],
                         *params['w_dt_fcn'], *params['w_ct_fcn'])
        names = ('f_bar A_bar B_bar C_bar S_bar z_bar X_last U_last S_last w_tr '
                 'f_dt_last gf_dt_last f_ct_last A_ct_last B_ct_last C_ct_last '
                 'S_ct_last f_comp_last gf_comp_last gf_comp_neg_last '
                 'gf_smth_pos_last gf_smth_pos_cvx_last').split()
        self.param_dict = {name: SimpleNamespace(value=None) for name in names}
        self.build = jax.jit(partial(build_chain_qp, params=self.params,
                                    model=model, scale=self.scale))
        self.decode = jax.jit(partial(decode_solution, params=self.params))

    @property
    def compile_seconds(self):
        return self._build_compile_seconds + self.solver.compile_seconds

    @property
    def setup_seconds(self):
        return self.solver.setup_seconds

    def build_qp(self, values):
        if self._compiled_build is None:
            started = perf_counter()
            self._compiled_build = self.build.lower(values).compile()
            self._build_compile_seconds += perf_counter() - started
        started = perf_counter()
        qp = self._compiled_build(values)
        jax.block_until_ready(qp)
        return qp, perf_counter() - started

    def prepare(self, functions):
        """Compile array construction and prepare the solver without a warm-up solve."""
        from ctstl.full_scp import compile_array_kernels
        p = self.params
        X, U, S = (jnp.asarray(p[name], dtype=jnp.float64)
                   for name in ('X_last', 'U_last', 'S_last'))
        if not hasattr(self, '_array_linearize'):
            started = perf_counter()
            self._array_linearize, self._array_costs, self._cost_names = compile_array_kernels(
                X, U, S, p, functions, self.model)
            self._build_compile_seconds += perf_counter() - started
        values = self._array_linearize(X, U, S)
        values['w_tr'] = jnp.asarray(p['w_ptr'], dtype=jnp.float64)
        jax.block_until_ready(values)
        for name, value in values.items():
            self.param_dict[name].value = value
        qp, _ = self.build_qp(values)
        self.compile_decode(qp)
        self.solver.prepare(qp)

    def compile_decode(self, qp):
        if self._compiled_decode is None:
            started = perf_counter()
            self._compiled_decode = self.decode.lower(qp.center, qp).compile()
            self._build_compile_seconds += perf_counter() - started

    def solve(self):
        values = {name: parameter.value for name, parameter in self.param_dict.items()
                  if parameter.value is not None}
        solution = run_subproblem(self, values)
        return 'optimal', *solution

    def solve_ipm(self):
        return self.solve()

    def close(self):
        self.solver.close()


def build_chain_qp(values, params, model, scale=1.):
    """Translate the original increment-form model into absolute stage variables.

    y[k] = [X[:, k], U[:, k], S[k]]; the last S is a fixed unused padding slot.
    max(0, r) = (abs(r) + r)/2 is an exact reformulation of terminal hinges.
    Constant objective offsets are omitted; all minimizers are unchanged.
    """
    n_x, n_u, nodes = params['n_x'], params['n_u'], params['K']
    free_time = params['free_final_time']
    width = n_x + n_u + int(free_time)
    X, U = values['X_last'], values['U_last']
    nominal = jnp.concatenate((X.T, U.T), axis=1)
    if free_time:
        durations = jnp.pad(values['S_last'].ravel(), (0, 1))
        nominal = jnp.column_stack((nominal, durations))

    A = values['A_bar'].T.reshape(nodes - 1, n_x, n_x)
    B = values['B_bar'].T.reshape(nodes - 1, n_x, n_u)
    C = values['C_bar'].T.reshape(nodes - 1, n_x, n_u)
    left = jnp.concatenate((-A, -B), axis=2)
    right = jnp.concatenate((jnp.broadcast_to(jnp.eye(n_x), A.shape), -C), axis=2)
    offset = (-values['f_bar'].T
              + jnp.einsum('kij,kj->ki', A, X.T[:-1])
              + jnp.einsum('kij,kj->ki', B, U.T[:-1])
              + jnp.einsum('kij,kj->ki', C, U.T[1:]))
    if free_time:
        time_jacobian = values['S_bar'].T
        left = jnp.concatenate((left, -time_jacobian[:, :, None]), axis=2)
        right = jnp.pad(right, ((0, 0), (0, 0), (0, 1)))
        offset += time_jacobian * values['S_last'].ravel()[:, None]

    # Positive proximal curvature; physical quadratic costs are added exactly.
    hessian = jnp.full_like(nominal, 2. * values['w_tr'] / scale)
    if free_time:
        hessian = hessian.at[:, -1].multiply((params['w_ds'] / params['t_f']) ** 2)
    vehicle_hessian = jnp.zeros_like(nominal)
    if model == 'quad_flight':
        if params['add_inp']:
            vehicle_hessian = vehicle_hessian.at[:, n_x + 3].set(2 * params['w_inp'] / scale)
        if params['add_inp_trq']:
            vehicle_hessian = vehicle_hessian.at[:, n_x:n_x + 3].set(2 * params['w_inp_trq'] / scale)
        if params['add_elv_rate']:
            vehicle_hessian = vehicle_hessian.at[:, 5].set(2 * params['w_elv_r'] / scale)
    hessian += vehicle_hessian
    linear = vehicle_hessian * nominal
    if free_time:
        linear = linear.at[:-1, -1].add(params['w_S'] / params['t_f'] / scale)

    # Dynamics L1 penalties keep the physical/CTCS weight split from cvx.py.
    weights = jnp.concatenate((jnp.full(params['n_dyn'], params['w_con_dyn']),
                               jnp.full(n_x - params['n_dyn'], params['w_con_ctc']))) / scale
    weights = jnp.broadcast_to(weights, offset.shape)
    dt_jacobian = values['gf_dt_last'].reshape(-1, n_x, nodes).transpose(0, 2, 1)
    terminal_left, terminal_right, terminal_offset, terminal_weights = [], [], [], []
    for index, mode in enumerate(params['pen_dt_fcn']):
        weight = params['w_dt_fcn'][index] / scale
        derivative = jnp.pad(dt_jacobian[index], ((0, 0), (0, width - n_x)))
        if mode not in ('abs', 'max'):
            linear += weight * derivative
        elif weight > 0:
            # In the supplied models, these DT residuals are terminal functions.
            # A changed model must not silently discard nonterminal derivatives.
            terminal_only = jnp.max(jnp.abs(derivative[:-1])) <= 1e-14
            derivative = jnp.where(terminal_only, derivative, jnp.nan)
            row_left = jnp.zeros((nodes - 1, 1, width))
            row_right = row_left.at[-1, 0].set(derivative[-1])
            row_offset = jnp.zeros((nodes - 1, 1)).at[-1, 0].set(
                values['f_dt_last'][index] - jnp.sum(derivative * nominal))
            if mode == 'max':
                linear += 0.5 * weight * derivative
                weight *= 0.5
            terminal_left.append(row_left)
            terminal_right.append(row_right)
            terminal_offset.append(row_offset)
            terminal_weights.append(jnp.full((nodes - 1, 1), weight))
    if terminal_left:
        left = jnp.concatenate((left, *terminal_left), axis=1)
        right = jnp.concatenate((right, *terminal_right), axis=1)
        offset = jnp.concatenate((offset, *terminal_offset), axis=1)
        weights = jnp.concatenate((weights, *terminal_weights), axis=1)

    # An unpenalized linearized CT-STL cost is a global linear term, not a
    # dense Hessian. Both adjacent-control Jacobians (FOH) are retained.
    ct_weights = jnp.asarray(params['w_ct_fcn']) / scale
    state_ct = (ct_weights @ values['A_ct_last']).reshape(n_x, nodes - 1).T
    input_left = (ct_weights @ values['B_ct_last']).reshape(n_u, nodes - 1).T
    input_right = (ct_weights @ values['C_ct_last']).reshape(n_u, nodes - 1).T
    linear = linear.at[:-1, :n_x].add(state_ct)
    linear = linear.at[:-1, n_x:n_x + n_u].add(input_left)
    linear = linear.at[1:, n_x:n_x + n_u].add(input_right)
    if free_time:
        linear = linear.at[:-1, -1].add((ct_weights @ values['S_ct_last']).ravel())
    if params['ncvx_solver'] == 'pl':
        comp_gradient = jnp.asarray(params['w_comp_fcn']) @ values['gf_comp_last'] / scale
        linear = linear.at[:, :n_x].add(comp_gradient.reshape(n_x, nodes).T)
    else:
        # All supplied PCX component functions are identically zero.
        supported = (jnp.max(jnp.abs(values['gf_smth_pos_last'])) == 0)
        comp_gradient = jnp.asarray(params['w_comp_fcn']) @ values['gf_comp_neg_last'] / scale
        linear = linear.at[:, :n_x].add(comp_gradient.reshape(n_x, nodes).T)
        linear = jnp.where(supported, linear, jnp.nan)

    free = jnp.ones_like(nominal)
    fixed = jnp.zeros_like(nominal)
    if model == 'quad_flight':
        initial_indices = np.r_[0:8, 9:24]
        final_indices = np.r_[3:8, 9:12]
        initial_values = np.asarray(params['x_init']).copy()
        initial_values[12:24] = 0.
        initial_values[[19, 21]] = 1.
    else:
        initial_indices = np.arange(n_x)
        final_indices = np.arange(6)
        initial_values = np.asarray(params['x_init']).copy()
        initial_values[6:] = 0.
        if model == 'di_until_aug':
            initial_values[7] = 1.
        if model == 'di_eventually':
            initial_values[[6, 8, 10]] = 1.
    free = free.at[0, initial_indices].set(0.).at[-1, final_indices].set(0.)
    fixed = fixed.at[0, initial_indices].set(initial_values[initial_indices])
    fixed = fixed.at[-1, final_indices].set(np.asarray(params['x_final'])[final_indices])
    free = free.at[jnp.array([0, nodes - 1]), n_x:n_x + n_u].set(0.)
    fixed = fixed.at[0, n_x:n_x + n_u].set(params['u_init'])
    fixed = fixed.at[-1, n_x:n_x + n_u].set(params['u_final'])
    if free_time:
        free = free.at[-1, -1].set(0.)

    # Continuous-time accumulator increment, unchanged at 1e-4.
    pair_left = jnp.zeros((nodes - 1, n_x - params['n_dyn'], width))
    indices = jnp.arange(n_x - params['n_dyn'])
    pair_left = pair_left.at[:, indices, params['n_dyn'] + indices].set(-1.)
    pair_right = -pair_left
    pair_offset = jnp.full(pair_left.shape[:2], -1e-4)

    # Stage bounds. Inactive padding rows are -1 <= 0.
    local = jnp.zeros((nodes, 12 if model == 'quad_flight' else 2, width))
    local_offset = -jnp.ones(local.shape[:2])
    if free_time:
        lower = params['min_S'] if params['add_minmax_sig'] else 1e-4
        local = local.at[:-1, 0, -1].set(-1.)
        local_offset = local_offset.at[:-1, 0].set(lower)
        if params['add_minmax_sig']:
            local = local.at[:-1, 1, -1].set(1.)
            local_offset = local_offset.at[:-1, 1].set(-params['max_S'])
    if model == 'quad_flight':
        local = local.at[:, 2, n_x + 3].set(1.).at[:, 3, n_x + 3].set(-1.)
        local_offset = local_offset.at[:, 2].set(-params['T_max']).at[:, 3].set(params['T_min'])
        for axis in range(3):
            local = local.at[:, 4 + 2 * axis, n_x + axis].set(1.)
            local = local.at[:, 5 + 2 * axis, n_x + axis].set(-1.)
            local_offset = local_offset.at[:, 4 + 2 * axis:6 + 2 * axis].set(-params['tau_max'])
        local = local.at[-1, 10, 0].set(1.).at[-1, 11, 0].set(-1.)
        local_offset = local_offset.at[-1, 10].set(-params['x_final'][0])
        local_offset = local_offset.at[-1, 11].set(0.8 * params['x_final'][0])

    return ChainQP(left, right, offset, pair_left, pair_right, pair_offset,
                   local, local_offset, nominal, hessian, linear, weights, free, fixed)


def decode_solution(solution, qp, params):
    """Return exactly the X, U, S, nu arrays expected by scp.prox_convex."""
    n_x, n_u = params['n_x'], params['n_u']
    X = solution[:, :n_x].T
    U = solution[:, n_x:n_x + n_u].T
    S = solution[:-1, -1][None] if params['free_final_time'] else jnp.array(0., dtype=jnp.float64)
    nu = (jnp.einsum('kij,kj->ki', qp.left[:, :n_x], solution[:-1])
          + jnp.einsum('kij,kj->ki', qp.right[:, :n_x], solution[1:])
          + qp.offset[:, :n_x]).T
    return X, U, S, nu


def run_subproblem(owner, values):
    """Keep trajectories on JAX; compiled decode accepts owned NumPy solutions."""
    values, upload_seconds = to_device(values, dtype=jnp.float64)
    compilation_before = owner.compile_seconds
    setup_before = owner.setup_seconds
    qp, build_seconds = owner.build_qp(values)
    owner.compile_decode(qp)
    y, record = owner.solver.solve(qp)
    started = perf_counter()
    # Compiled JAX functions accept NumPy inputs directly. Avoid an extra
    # device_put dispatch; any input copy is included in the decode timer.
    solution = owner._compiled_decode(y, qp)
    jax.block_until_ready(solution)
    decode_seconds = perf_counter() - started
    record = dict(record)
    record['seconds'] += build_seconds + decode_seconds
    record['solution_decode_seconds'] = decode_seconds
    record['numpy_to_jax_seconds'] += upload_seconds
    record['compilation_seconds'] = owner.compile_seconds - compilation_before
    record['setup_seconds'] = owner.setup_seconds - setup_before
    owner.records.append(record)
    return solution
