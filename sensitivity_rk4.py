"""RK4 sensitivity transport without materializing continuous-time Jacobians.

The RK4 grid, input interpolation and time arguments are identical to the
reference integrator. All calculations use the input dtype (normally float64).
No finite-difference approximation, thresholded sparsity or step reduction is used.
"""
import jax
import jax.numpy as jnp
from jax import lax


def make_discretizer(params):
    nx, nu, nodes = params['n_x'], params['n_u'], params['K']
    steps = int(params['rk4_steps'])
    if steps < 2:
        raise ValueError('RK4 needs at least two grid points.')
    free_time, dilation = params['free_final_time'], params['time_dil']
    foh = params['inp_param'] == 'FOH'
    if params['inp_param'] not in ('FOH','ZOH'):
        raise ValueError('Unknown input interpolation.')
    span = (1.0 if dilation else 1.0/(nodes-1)) if free_time else params['t_scp']
    h = span/(steps-1)
    f = params['f_func']
    autodiff = params.get('dynamics_autodiff', 'forward')
    if autodiff not in ('forward', 'reverse'):
        raise ValueError('dynamics_autodiff must be forward or reverse')

    def interval(x0, u0, u1, scale, start):
        identity = jnp.eye(nx,dtype=x0.dtype)
        eye_u = jnp.eye(nu,dtype=x0.dtype)
        tangent0 = jnp.pad(identity, ((0,0),(0,2*nu+1)))
        def rhs(state,t):
            x, tangent = state
            beta = t/span if foh else jnp.zeros_like(t)
            alpha = 1.0-beta
            u = u0+beta*(u1-u0)
            def field(x,u):
                return f(x,u,start+t)
            control = jnp.concatenate((jnp.zeros((nu,nx),dtype=x.dtype),
                                       alpha*eye_u,beta*eye_u,
                                       jnp.zeros((nu,1),dtype=x.dtype)),axis=1)
            if autodiff == 'reverse':
                value = field(x, u)
                A, B = jax.jacrev(field, argnums=(0, 1))(x, u)
                direction = A @ tangent + B @ control
            else:
                value, push = jax.linearize(field, x, u)
                direction = jax.vmap(push,in_axes=(1,1),out_axes=1)(tangent,control)
            derivative = scale*direction
            derivative = derivative.at[:,-1].add(value)
            return scale*value, derivative
        def add(y,k,a):
            return tuple(v+a*d for v,d in zip(y,k))
        def step(state, index):
            t=index*h
            k1 = jax.tree.map(lambda v:h*v,rhs(state,t))
            k2 = jax.tree.map(lambda v:h*v,rhs(add(state,k1,.5),t+.5*h))
            k3 = jax.tree.map(lambda v:h*v,rhs(add(state,k2,.5),t+.5*h))
            k4 = jax.tree.map(lambda v:h*v,rhs(add(state,k3,1.),t+h))
            nxt = jax.tree.map(lambda v,a,b,c,d:v+(a+2*b+2*c+d)/6.,state,k1,k2,k3,k4)
            return nxt, nxt
        _,history = lax.scan(step,(x0,tangent0),jnp.arange(steps-1))
        def unpack(x, tangent):
            A=tangent[...,:nx];B=tangent[...,nx:nx+nu];C=tangent[...,nx+nu:nx+2*nu];S=tangent[...,-1]
            z=x-jnp.einsum('...ij,j->...i',A,x0)-jnp.einsum('...ij,j->...i',B,u0)-jnp.einsum('...ij,j->...i',C,u1)
            if free_time:z=z-S*scale
            return A,B,C,S,z
        x=jnp.concatenate((x0[None],history[0]),axis=0)
        tangent=jnp.concatenate((tangent0[None],history[1]),axis=0)
        A,B,C,S,z=unpack(x,tangent)
        return jnp.concatenate((x,A.reshape(steps,-1),B.reshape(steps,-1),
                                C.reshape(steps,-1),S,z),axis=1)

    def discretize(X,U,S):
        if free_time:
            scales=S[0] if dilation else jnp.broadcast_to(S,(nodes-1,))
            starts=jnp.cumsum(jnp.concatenate((jnp.zeros(1,dtype=X.dtype),S[0])))[:-1] if dilation else jnp.arange(nodes-1)*S/(nodes-1)
        else:
            scales=jnp.ones(nodes-1,dtype=X.dtype)
            starts=jnp.arange(nodes-1)*params['t_scp']
        return jax.vmap(interval)(X[:,:-1].T,U[:,:-1].T,U[:,1:].T,scales,starts)
    return jax.jit(discretize)
