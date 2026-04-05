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
        Quadrotor continuous‐time dynamics plus CTCS penalty.
        State x = [pos(3), vel(3), euler(3), omega(3), CTCS]
        Input u = [τ₁, τ₂, τ₃, thrust]
        """

        rot = rotation_matrix(x[6:9])
        
        dx = jnp.zeros_like(x)
        dx = dx.at[0:3].set(x[3:6])  # velocities

        # Rotation application for force in body z direction
        dx = dx.at[3].set(u[3] * rot[0, 2])
        dx = dx.at[4].set(u[3] * rot[1, 2])
        dx = dx.at[5].set(u[3] * rot[2, 2] - params['g_0'])

        # Angular velocities
        dx = dx.at[6].set(
            x[9] +
            jnp.sin(x[6]) * jnp.tan(x[7]) * x[10] +
            jnp.cos(x[6]) * jnp.tan(x[7]) * x[11]
        )
        dx = dx.at[7].set(jnp.cos(x[6]) * x[10] - jnp.sin(x[6]) * x[11])
        dx = dx.at[8].set(
            jnp.sin(x[6]) / jnp.cos(x[7]) * x[10] +
            jnp.cos(x[6]) / jnp.cos(x[7]) * x[11]
        )

        # Torques
        dx = dx.at[9].set((params['I_y'] - params['I_z']) / params['I_x'] * x[10] * x[11] + u[0] / params['I_x'])
        dx = dx.at[10].set((params['I_z'] - params['I_x']) / params['I_y'] * x[9] * x[11] + u[1] / params['I_y'])
        dx = dx.at[11].set((params['I_x'] - params['I_y']) / params['I_z'] * x[9] * x[10] + u[2] / params['I_z'])

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Time-varying STC barrier states
        # ------------------------------------------------------------------------------------------------------------------------------------------------

        STC_SCALE = 1e8
        TIME_SCALE = 10.0
        TIME_RATE = 0.1

        # Reset STC states
        for idx in range(12, 18):
            dx = dx.at[idx].set(0.0)

        time = TIME_SCALE * x[18]
        x_pos = x[0]
        y_pos = x[1]

        delta_p = params["y_stc_range"] * (jnp.sin(time - 0.5 * jnp.pi) + 1.0)


        def rect_barrier_sq(x_val, y_val, x_low, x_high, y_boundary, upper_halfspace):
            x_left = jnp.minimum(0.0, x_low - x_val)
            x_right = jnp.minimum(0.0, x_val - x_high)

            if upper_halfspace:
                y_term = jnp.minimum(0.0, y_val - y_boundary)
            else:
                y_term = jnp.minimum(0.0, -(y_val - y_boundary))

            return jnp.square(x_left) * jnp.square(x_right) * jnp.square(y_term)


        dist_1 = rect_barrier_sq(
            x_pos, y_pos,
            params["x_stc_1"], params["x_stc_2"],
            params["y_stc_1"] - delta_p,
            upper_halfspace=True,
        )

        dist_2 = rect_barrier_sq(
            x_pos, y_pos,
            params["x_stc_3"], params["x_stc_4"],
            params["y_stc_2"] - delta_p,
            upper_halfspace=False,
        )

        dist_3 = rect_barrier_sq(
            x_pos, y_pos,
            params["x_stc_5"], params["x_stc_6"],
            params["y_stc_3"] + delta_p,
            upper_halfspace=True,
        )

        dist_4 = rect_barrier_sq(
            x_pos, y_pos,
            params["x_stc_7"], params["x_stc_8"],
            params["y_stc_4"] - delta_p,
            upper_halfspace=False,
        )

        dist_5 = rect_barrier_sq(
            x_pos, y_pos,
            params["x_stc_9"], params["x_stc_10"],
            params["y_stc_5"] - delta_p,
            upper_halfspace=True,
        )

        dist_6 = rect_barrier_sq(
            x_pos, y_pos,
            params["x_stc_11"], params["x_stc_12"],
            params["y_stc_6"] + delta_p,
            upper_halfspace=False,
        )

        dx = dx.at[12].set(STC_SCALE * dist_1)
        dx = dx.at[13].set(STC_SCALE * dist_2)
        dx = dx.at[14].set(STC_SCALE * dist_3)
        dx = dx.at[15].set(STC_SCALE * dist_4)
        dx = dx.at[16].set(STC_SCALE * dist_5)
        dx = dx.at[17].set(STC_SCALE * dist_6)

        # Time state
        dx = dx.at[18].set(TIME_RATE)

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Moving window event states
        # ------------------------------------------------------------------------------------------------------------------------------------------------

        EVENT_EPS = 1e-12
        LOG_SCALE = 4.0
        PENALTY_SCALE = 60.0
        PENALTY_GAIN = 1e5

        for idx in range(19, 23):
            dx = dx.at[idx].set(0.0)

        t_f = params["t_f"]

        y_loc_1 = params["pw1_base"] + 0.5 * params["pw_range"] * (
            jnp.sin((jnp.pi / 5.655) * time - 0.5 * jnp.pi) + 1.0
        )
        y_loc_2 = params["pw2_base"] - 0.5 * params["pw_range"] * (
            jnp.sin((jnp.pi / 4.38) * time - 0.5 * jnp.pi) + 1.0
        )

        center_1 = jnp.array([params["p_w_1"][0], y_loc_1, params["p_w_1"][2]])
        center_2 = jnp.array([params["p_w_2"][0], y_loc_2, params["p_w_2"][2]])

        dist_margin_1 = params["r_w_1"]**2 - jnp.dot(x[:3] - center_1, x[:3] - center_1)
        dist_margin_2 = params["r_w_2"]**2 - jnp.dot(x[:3] - center_2, x[:3] - center_2)

        log_arg_1 = jnp.square(jnp.minimum(0.0, LOG_SCALE * dist_margin_1)) + EVENT_EPS
        log_arg_2 = jnp.square(jnp.minimum(0.0, LOG_SCALE * dist_margin_2)) + EVENT_EPS

        penalty_1 = jnp.square(jnp.maximum(0.0, PENALTY_SCALE * dist_margin_1))
        penalty_2 = jnp.square(jnp.maximum(0.0, PENALTY_SCALE * dist_margin_2))

        dx = dx.at[19].set(x[19] * jnp.log(log_arg_1) / t_f)
        dx = dx.at[20].set(PENALTY_GAIN * penalty_1 / t_f)

        dx = dx.at[21].set(x[21] * jnp.log(log_arg_2) / t_f)
        dx = dx.at[22].set(PENALTY_GAIN * penalty_2 / t_f)

        # ------------------------------------------------------------------------------------------------------------------------------------------------

        dx = dx.at[23].set(0)
        
        spd = jnp.linalg.norm(x[3:6] + 1e-8)
        dx = dx.at[23].add( params['w_states_spd'] * jnp.maximum(0, spd - params['vehicle_v_max'])**2 )

        dx = dx.at[23].add( params['w_states_alt'] * jnp.maximum(0, x[2] - params['max_alt'])**2 )
        dx = dx.at[23].add( params['w_states_alt'] * jnp.maximum(0, -x[2] + params['min_alt'])**2 )
        
        dx = dx.at[23].add( params['w_states_phi'] * jnp.maximum(0,  x[6] - params['phi_bd'])**2 )
        dx = dx.at[23].add( params['w_states_phi'] * jnp.maximum(0, -x[6] - params['phi_bd'])**2 )

        dx = dx.at[23].add( params['w_states_tht'] * jnp.maximum(0,  x[7] - params['theta_bd'])**2 )
        dx = dx.at[23].add( params['w_states_tht'] * jnp.maximum(0, -x[7] - params['theta_bd'])**2 )

        dx = dx.at[23].add( params['w_states_p'] * jnp.maximum(0,  x[9] - params['phi_rate'])**2 )
        dx = dx.at[23].add( params['w_states_p'] * jnp.maximum(0, -x[9] - params['phi_rate'])**2 )

        dx = dx.at[23].add( params['w_states_q'] * jnp.maximum(0,  x[10] - params['theta_rate'])**2 )
        dx = dx.at[23].add( params['w_states_q'] * jnp.maximum(0, -x[10] - params['theta_rate'])**2 )

        dx = dx.at[23].add( params['w_states_r'] * jnp.maximum(0, x[11] - params['yaw_rate'])**2 )
        dx = dx.at[23].add( params['w_states_r'] * jnp.maximum(0, -x[11] - params['yaw_rate'])**2 )

        ## - CTCS Updates Params: x_init and n_x + Dyn + CTCS Cons + f_obj

        return dx

    params['f_func'] = jit(f)
    params['A_func'] = jit(jacfwd(f, argnums=0))
    params['B_func'] = jit(jacfwd(f, argnums=1))

    return params
