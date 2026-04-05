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
        # Speed-limited until-target event-state dynamics
        # ------------------------------------------------------------------------------------------------------------------------------------------------

        C1 = 1e-4
        C2 = 1e-4
        EVENT_EPS = 1e-12
        TAU_EPS = 1e-4
        SPEED_MARGIN = 0.06
        INTEGRAL_SCALE = 50.0
        DISTANCE_SCALE = 10.0

        t_f = params["t_f"]
        tau_safe = tau + TAU_EPS

        # Reset auxiliary states
        speed_int_idx = 6
        geo_int_idx = 7
        dx = dx.at[speed_int_idx].set(0.0)
        dx = dx.at[geo_int_idx].set(0.0)

        # Speed violation: negative means satisfied
        speed_sq = jnp.dot(x[3:6], x[3:6])
        speed_limit_sq = (params["spd_lim"] - SPEED_MARGIN) ** 2
        speed_violation = speed_sq - speed_limit_sq

        speed_integral = jnp.square(jnp.maximum(speed_violation, 0.0))
        speed_scale = t_f * INTEGRAL_SCALE
        dx = dx.at[speed_int_idx].set(speed_integral / speed_scale)

        # Smooth "always until t" term
        q1_t = jnp.sqrt(C1**2 + jnp.maximum(x[speed_int_idx], 0.0) * speed_scale / tau_safe) - C1

        position_error = x[:3] - params["p_w"]
        dist_sq = jnp.dot(position_error, position_error)
        dist_violation = DISTANCE_SCALE * (dist_sq - params["r_w"] ** 2)

        z_pos = jnp.sqrt(
            C2 + 0.5 * (
                jnp.square(jnp.maximum(dist_violation, 0.0))
                + jnp.square(jnp.maximum(q1_t, 0.0))
            )
        )
        z_t = (z_pos - jnp.sqrt(C2)) / INTEGRAL_SCALE

        geo_integrand = jnp.square(jnp.maximum(z_t, 0.0)) + EVENT_EPS
        dx = dx.at[geo_int_idx].set(x[geo_int_idx] * jnp.log(geo_integrand) / t_f)

        # ------------------------------------------------------------------------------------------------------------------------------------------------

        dx = dx.at[8].set(0)
        dx = dx.at[8].add( params['w_states_theta'] * jnp.maximum(0, u[0]**2 + u[1]**2 - ( jnp.cos(params['theta']) * u[2] )**2 )**2 )
        dx = dx.at[8].add( params['w_states_T_max'] * jnp.maximum(0, u[0]**2 + u[1]**2 + u[2]**2 - params['T_max']**2 )**2 )
        dx = dx.at[8].add( params['w_states_v_max'] * jnp.maximum(0, x[3]**2 + x[4]**2 + x[5]**2 - params['v_max']**2 )**2 )

        return dx

    params['f_func'] = jit(f)
    params['A_func'] = jit(jacfwd(f, argnums=0))
    params['B_func'] = jit(jacfwd(f, argnums=1))

    return params


