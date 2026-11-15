from jax import config, lax
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import typing as T

def conjunction(c: float,
                 p: int,
                 weights: jnp.ndarray,
                 *args: T.Tuple[T.Any]) -> T.Any:
    '''
    Input: The values of the functions to be connected with conjunction -> (f, g, h, ...)
    Output: gmsr_and function's value
    '''
    
    fcn_vals = args[0]
    
    # Masks for positive and negative elements
    pos_mask = fcn_vals > 0
    # neg_mask = ~pos_mask
    neg_mask = fcn_vals <= 0

    # Values and weights partitioned by positive and negative masks
    pos_vals = jnp.where(pos_mask, fcn_vals, 0.0)
    neg_vals = jnp.where(neg_mask, fcn_vals, 0.0)
    
    pos_w = jnp.where(pos_mask, weights, 0.0)
    neg_w = jnp.where(neg_mask, weights, 0.0)
    
    sum_w = weights.sum()

    # Conditional computation using cond
    def compute_neg_case(ada):
        sums = jnp.sum(neg_w * (neg_vals**(2 * p)))
        Mp = (c**p + (sums / sum_w))**(1/p)
        return c**(1/2) - Mp**(1/2)
    
    def compute_pos_case(ada):
        # mult = jnp.prod((pos_vals)**(2 * pos_w))
        mult = jnp.exp(jnp.sum(2 * pos_w * jnp.log(pos_vals)))
        M0 = (c**sum_w + mult)**(1/sum_w)
        return M0**(1/2) - c**(1/2)

        # k = 10
        # num = jnp.prod(pos_vals, axis=-1)
        # den = jnp.linalg.norm(pos_vals, ord=k, axis=-1)
        # M0  = num/den + c
        # return jnp.sqrt(M0) - jnp.sqrt(c)
    
    # Choose which branch to execute
    h_and = lax.cond(jnp.any(neg_mask), compute_neg_case, compute_pos_case, operand=None)

    return h_and

def disjunction(c: float, p: int, weights: jnp.ndarray, *args: T.Tuple[T.Any]) -> T.Any:
    '''
    Input: The values of the functions to be connected with disjunction
    Output: gmsr_or function's value
    '''
    
    neg_args = -jnp.array(args[0])
    h_mor = conjunction(c, p, weights, neg_args)
    return -h_mor

def UNTIL(
    c: float,
    p: int,
    w_f: jnp.ndarray,
    w_g: jnp.ndarray,
    w_fg: jnp.ndarray,
    f: jnp.ndarray,
    g: jnp.ndarray,
) -> jnp.ndarray:
    """
    JAX‐jit‐compatible version of UNTIL. Uses a Python loop over K (static)
    so that slicing w_f[:i+1] and f[:i+1] always sees a concrete Python int.
    """
    K = f.shape[0]
    s_list = []

    # Python loop unrolls at trace‐time (K is static), so slicing is OK
    for i in range(int(K)):
        # build prefix‐arrays of length (i+1)
        w_pref = w_f[: i + 1]
        f_pref = f[: i + 1]

        # “rolling conjunction” over the prefix
        y_i = conjunction(c, p, w_pref, f_pref)

        # then conjunction with g[i]
        pair = jnp.array([y_i, g[i]])
        s_i  = conjunction(c, p, w_fg[:2], pair)

        s_list.append(s_i)

    # build full s‐vector and disjunction‐reduce it
    s = jnp.stack(s_list)           # shape (K,)
    z = disjunction(c, p, w_g, s)  # scalar

    return z
