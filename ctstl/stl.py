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

def _conjunction_from_sums(c, p, weight, negative_power, log_product, any_negative):
    # Inactive geometric branches must stay finite for reverse-mode AD.
    safe_weight = jnp.where(any_negative, 1., weight)
    safe_log = jnp.where(any_negative, 0., log_product)
    positive = jnp.sqrt((c**safe_weight + jnp.exp(safe_log))**(1./safe_weight)) - jnp.sqrt(c)
    negative = jnp.sqrt(c) - jnp.sqrt((c**p + negative_power/weight)**(1./p))
    return jnp.where(any_negative, negative, positive)


def UNTIL(
    c: float,
    p: int,
    w_f: jnp.ndarray,
    w_g: jnp.ndarray,
    w_fg: jnp.ndarray,
    f: jnp.ndarray,
    g: jnp.ndarray,
) -> jnp.ndarray:
    """Original weighted GMSR UNTIL, with linear-time prefix reductions."""
    negative = f <= 0
    prefix = _conjunction_from_sums(
        c, p, jnp.cumsum(w_f),
        jnp.cumsum(w_f*jnp.where(negative, f, 0.)**(2*p)),
        jnp.cumsum(2*w_f*jnp.log(jnp.where(negative, 1., f))),
        jnp.cumsum(negative.astype(jnp.int32)) > 0)
    pair = jnp.stack((prefix, g), axis=1)
    negative = pair <= 0
    weights = jnp.asarray(w_fg[:2])
    values = _conjunction_from_sums(
        c, p, weights.sum(),
        jnp.sum(weights*jnp.where(negative, pair, 0.)**(2*p), axis=1),
        jnp.sum(2*weights*jnp.log(jnp.where(negative, 1., pair)), axis=1),
        jnp.any(negative, axis=1))
    return disjunction(c, p, w_g, values)
