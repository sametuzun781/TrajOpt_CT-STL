from jax import config, jacfwd, vmap, Array
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import glob
from PIL import Image

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

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

    col_width = int(1 * max(len(ele) for ele in log_ite_all.keys() if ele is not None)) + 4

    if log_ite_all['Note'][-1] == 'Start':
        all_dum = ''
        for key in log_ite_all.keys():
            all_dum += key.ljust(col_width)
        print(all_dum)

    all_dum = ''
    for key in log_ite_all.keys():
        if type(log_ite_all[key]) == list:
            dum = log_ite_all[key][-1]
            if isinstance(dum, Array):
                dum = float(dum)
            if isinstance(dum, float):
                dum = round(dum, 4)
                dum = min(dum, 999999.99)

        all_dum += str(dum).ljust(col_width)
    print(all_dum)

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

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
