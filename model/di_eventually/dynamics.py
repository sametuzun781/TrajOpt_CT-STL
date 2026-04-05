from jax import config, jit, jacfwd
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import typing as T
from utils import rotation_matrix

def dynamics(params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
    """
    JIT-compiled dynamics and its Jacobians:
      - f_func(x,u):   state derivative
      - A_func(x,u):   ∂f/∂x
      - B_func(x,u):   ∂f/∂u
    """

    def f(x: jnp.ndarray, u: jnp.ndarray, tau: float) -> jnp.ndarray:
        """
        Double-integrator plus CTCS penalty.
        State x = [pos(3), vel(3), CTCS, Eventually]
        Input u = [T1, T2, T3]
        """

        dx = jnp.zeros_like(x)
        dx = dx.at[0:3].set(x[3:6])  # velocities

        # Rotation application for force in body z direction
        dx = dx.at[3].set(u[0])
        dx = dx.at[4].set(u[1])
        dx = dx.at[5].set(u[2] - params['g_0'])

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Window / obstacle event-state dynamics
        # ------------------------------------------------------------------------------------------------------------------------------------------------

        EVENT_EPS = 1e-6
        LOG_SCALE = 40.0
        PENALTY_SCALE = 60.0

        window_specs = (
            ("p_w",   "r_w",   6,  7),
            ("p_w_2", "r_w_2", 8,  9),
            ("p_w_3", "r_w_3", 10, 11),
        )

        position = x[:3]
        t_f = params["t_f"]

        for center_key, radius_key, log_idx, penalty_idx in window_specs:
            delta = position - params[center_key]
            dist_sq = jnp.dot(delta, delta)
            margin = params[radius_key] ** 2 - dist_sq

            log_arg = jnp.square(jnp.minimum(margin / LOG_SCALE, 0.0)) + EVENT_EPS
            penalty = jnp.square(jnp.maximum(PENALTY_SCALE * margin, 0.0))

            dx = dx.at[log_idx].set(x[log_idx] * jnp.log(log_arg) / t_f)
            dx = dx.at[penalty_idx].set(penalty / t_f)

        # ------------------------------------------------------------------------------------------------------------------------------------------------

        dx = dx.at[12].set(0)
        dx = dx.at[12].add( params['w_states_theta'] * jnp.maximum(0, u[0]**2 + u[1]**2 - ( jnp.cos(params['theta']) * u[2] )**2 )**2 )
        dx = dx.at[12].add( params['w_states_T_max'] * jnp.maximum(0, u[0]**2 + u[1]**2 + u[2]**2 - params['T_max']**2 )**2 )
        dx = dx.at[12].add( params['w_states_v_max'] * jnp.maximum(0, x[3]**2 + x[4]**2 + x[5]**2 - params['v_max']**2 )**2 )

        return dx

    params['f_func'] = jit(f)
    params['A_func'] = jit(jacfwd(f, argnums=0))
    params['B_func'] = jit(jacfwd(f, argnums=1))

    return params


