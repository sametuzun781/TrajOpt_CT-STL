"""Shared array boundaries, timing, plotting and geometry helpers."""
import jax
from time import perf_counter
from jax import config, jacfwd, vmap, Array
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import glob
import numbers
import numpy as np
from PIL import Image

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

ARRAY_FLOW_VERSION = "compiled-solution-input-r2"


def to_host(values):
    """Fetch a ready pytree; CPU views are read-only and the wait is not timed."""
    jax.block_until_ready(values)
    started = perf_counter()
    result = (jax.tree.map(np.asarray, values) if jax.default_backend() == 'cpu'
              else jax.device_get(values))
    return result, perf_counter() - started


def to_device(values, *, dtype=None):
    """Keep device arrays; time any required upload or explicit dtype conversion."""
    leaves = jax.tree.leaves(values)
    ready = all(isinstance(value, jax.Array) and
                (dtype is None or value.dtype == dtype) for value in leaves)
    if ready:
        return values, 0.0
    started = perf_counter()
    result = (jax.device_put(values) if dtype is None else
              jax.tree.map(lambda value: jnp.asarray(value, dtype=dtype), values))
    jax.block_until_ready(result)
    return result, perf_counter() - started


### Profiling
def profile_with_cprofile(func, *args, filename="profiling_results.prof", **kwargs):
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    pr.disable()
    pr.dump_stats(filename)
    return result

### Animation maker
def make_anim(load_from, save_to, duration, fps, anim_format):

    if anim_format == 'gif':
        frames = [Image.open(image) for image in sorted(glob.glob(f"{load_from}/*"+".png"))][::2]
        frame_one = frames[0]
        frame_one.save(save_to + ".gif", format="GIF", 
                       append_images=frames, save_all=True, 
                       duration=duration, loop=0)
    
    elif anim_format == 'mp4':
        import cv2

        # Get the list of image frames
        frames = sorted(glob.glob(f"{load_from}/*.png"))

        # Read the first frame to get frame dimensions
        first_frame = cv2.imread(frames[0])
        height, width, layers = first_frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify MP4 codec
        video = cv2.VideoWriter(save_to + ".mp4", fourcc, fps, (width, height))

        # Add all frames to the video
        for frame in frames:
            img = cv2.imread(frame)
            video.write(img)

        # Release the video writer
        video.release()

### Print
def print_run_summary(result, params, functions):
    """Show totals and per-call timings in ms, separating one-time boundaries."""
    trials = result['trial_timings']
    calls = result['convex_solve_count']
    linearizations = len({trial['iteration'] for trial in trials})
    print('Discretization:', functions['cal_disc_jitted'].discretization_backend)
    print('Solver:', result['solvers_used'], '| Converged:', result['converged'])
    print('Linearizations / convex calls:', linearizations, '/', calls)
    rows = (
        ('Discretization / linearization', result['total_discretization_seconds'], linearizations),
        ('Convex calls (incl. solution input/decode)', result['total_convex_seconds'], calls),
        ('Explicit JAX -> NumPy (loop)', sum(t['T_J_Np'] for t in trials), calls),
        ('Explicit NumPy -> JAX (loop)', sum(t['T_Np_J'] for t in trials), calls),
    )
    for label, seconds, count in rows:
        print(f'{label}: {1000 * seconds:.3f} ms total; {1000 * seconds / max(count, 1):.4f} ms/call')
    startup = result['initial_jax_to_numpy_seconds'] + result['initial_numpy_to_jax_seconds']
    print(f"Startup / final transitions [ms]: {1000 * startup:.3f} / {1000 * result['final_jax_to_numpy_seconds']:.3f}")
    duration = np.sum(result['S_new']) if params['free_final_time'] else params['t_f']
    print('Flight time [s]:', float(duration))


def dict_append(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

    return dict

def update_cost_dict(d, **kwargs):
    key_map = {
        'T_Ite': 'T-Ite',
        'T_Disc': 'T-Disc',
        'T_SubP': 'T-SubP',
        'T_J_Np': 'T-J_Np',
        'T_Np_J': 'T-Np_J',
        'D_NL': 'D-NL',
        'D_L': 'D-L',
        'W_TR': 'W-TR',
        'Note': 'Note',
        'Staus': 'Staus',
        'Rho': 'Rho',
        'Ite': 'Ite',
    }
    for key, val in kwargs.items():
        d = dict_append(d, key_map.get(key, key), val)
    return d

def print_ite(log_ite_all):
    """Timing records are seconds; display is milliseconds."""
    columns = list(log_ite_all)
    labels = [key + " [ms]" if key.startswith("T-") else key for key in columns]
    width = max(14, max(map(len, labels), default=0) + 2)
    if log_ite_all.get("Note", [None])[-1] == "Start":
        print("".join(label.ljust(width) for label in labels))
    values = []
    for key in columns:
        value = log_ite_all[key]
        if isinstance(value, list):
            value = value[-1]
        if hasattr(value, "shape") and np.size(value) == 1:
            value = float(value)
        if isinstance(value, numbers.Real):
            if key.startswith("T-"):
                value = f"{1000.0 * float(value):.3f}"
            elif not isinstance(value, numbers.Integral):
                value = f"{value:.2f}"
        values.append(str(value).ljust(width))
    print("".join(values))


def rotation_matrix(angle):
    phi, theta, psi = angle
    R_roll = jnp.array([[1, 0, 0],
                        [0, jnp.cos(phi), -jnp.sin(phi)],
                        [0, jnp.sin(phi), jnp.cos(phi)]])
    
    R_pitch = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                            [0, 1, 0],
                            [-jnp.sin(theta), 0, jnp.cos(theta)]])
    
    R_yaw = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                        [jnp.sin(psi), jnp.cos(psi), 0],
                        [0, 0, 1]])
    
    return jnp.dot(R_yaw, jnp.dot(R_pitch, R_roll))

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def norm_fcn_jax(x):
    return jnp.linalg.norm(x)

def f_visit_wp_jax(H, pos_old, tgt_vec, r):
    def single_term(pos_k):
        ada = norm_fcn_jax(H @ (pos_k - tgt_vec))
        return 0.5 * (r**2 - ada**2)

    # Vectorized mapping across columns of pos_old
    f_list = vmap(single_term, in_axes=1)(pos_old)
    return f_list  # This is already a 1D array

def f_cos_sim_jax(U_P_all, alpha, V_P_all):
    def single_f(U_P, V_P):
        return -jnp.dot(U_P, V_P) + jnp.cos(alpha) * norm_fcn_jax(U_P) * norm_fcn_jax(V_P)

    f_vec = vmap(single_f, in_axes=(1, 1))(U_P_all, V_P_all)
    return f_vec

def f_cos_sim_pos_angle_all_jax(U_P_all, angle_P_all, alpha, los_dir):
    def f_single(U_P, angle_P):
        V_P = (rotation_matrix(angle_P) @ los_dir)[:, 0]
        return -jnp.dot(U_P, V_P) + jnp.cos(alpha) * jnp.linalg.norm(U_P) * jnp.linalg.norm(V_P)

    return vmap(f_single, in_axes=(1, 1))(U_P_all, angle_P_all)

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def gf_visit_wp_jacobian_jax(H, pos_old, tgt_vec, r):
    def single_term(pos_k):
        ada = norm_fcn_jax(H @ (pos_k - tgt_vec))
        return 0.5 * (r**2 - ada**2)

    f_vector = lambda pos_old_: vmap(single_term, in_axes=1)(pos_old_)
    grad_fn = jacfwd(f_vector)
    return grad_fn(pos_old)  # shape (M, N, K)

def gf_cos_sim_jacobian_jax(U_P_all, V_P_all, alpha):
    def single_f(U_P, V_P):
        return -jnp.dot(U_P, V_P) + jnp.cos(alpha) * norm_fcn_jax(U_P) * norm_fcn_jax(V_P)

    grad_f = jacfwd(single_f, argnums=(0, 1))

    def jacobian_at_k(U_P, V_P):
        df_dU_P, df_dV_P = grad_f(U_P, V_P)
        return df_dU_P, df_dV_P  # each ∈ ℝ^N

    return vmap(jacobian_at_k, in_axes=(1, 1))(U_P_all, V_P_all)  # shapes: (K, N), (K, N)

def gf_cos_sim_pos_angle_jacobian_jax(U_P_all, angle_P_all, alpha, los_dir):
    def single_f(U_P, angle_P):
        V_P = (rotation_matrix(angle_P) @ los_dir)[:, 0]
        return -jnp.dot(U_P, V_P) + jnp.cos(alpha) * jnp.linalg.norm(U_P) * jnp.linalg.norm(V_P)

    grad_f = jacfwd(single_f, argnums=(0, 1))

    def jacobian_at_k(U_P, angle_P):
        df_dU_P, df_dangle_P = grad_f(U_P, angle_P)
        return df_dU_P, df_dangle_P  # shapes: (N,), (A,)

    return vmap(jacobian_at_k, in_axes=(1, 1))(U_P_all, angle_P_all)  # shapes: (K, N), (K, A)

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
