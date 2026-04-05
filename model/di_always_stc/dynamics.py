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

        dx = dx.at[6].set(0)
        dx = dx.at[7].set(0)

        dist_1 = jnp.minimum(0, params['x_stc_1']-x[0] )**2 * jnp.minimum(0, -params["x_stc_2"]+x[0] )**2 * jnp.minimum(0, -x[1]+params['y_stc_1'] )**2
        dist_2 = jnp.minimum(0, params['x_stc_3']-x[0] )**2 * jnp.minimum(0, -params['x_stc_4']+x[0] )**2 * jnp.minimum(0,  x[1]-params['y_stc_2'] )**2
        dx = dx.at[6].add( 1000*dist_1 )
        dx = dx.at[7].add( 1000*dist_2 )

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


