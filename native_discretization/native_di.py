"""Certified FP64 double-integrator RK4 through JAX's CPU FFI."""
import ctypes as ct
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'di_always_stc': (0, 9, 'f66a3bcefd20356a45f7542446ec3c1ae124e798f1b9f05fcba8f0293e2e9b31'),
    'di_eventually': (1, 13, '3fc83c436a7aefebbc5c0258d69f17c492c0ea6902555c75209fb93c0772304c'),
    'di_until': (2, 7, '983e72127605ec0d394c9fa8c0553f85ec37f08c6b0c74c9d523658dc6371c90'),
    'di_until_aug': (3, 9, '29136962ea1a547918a20d63f75140b6a62461c57ed74e39f2432a4b7661b73d'),
}
_LIBRARIES = {}


def eligibility(params):
    field = params.get('f_func')
    if field is None or params.get('_native_di_field') is not field:
        return False, 'unrecognized or replaced double-integrator dynamics'
    module = getattr(field, '__module__', '').split('.')
    model = module[1] if len(module) == 3 and module[0] == 'model' else None
    if model not in MODELS or (params['n_x'], params['n_u']) != (MODELS[model][1], 3):
        return False, 'unsupported double-integrator model or dimensions'
    source = (ROOT / 'model' / model / 'dynamics.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != MODELS[model][2]:
        return False, 'model source changed; use JAX or update the native field'
    if jax.default_backend() != 'cpu' or not jax.config.x64_enabled:
        return False, 'native double-integrator requires CPU and JAX x64'
    if params['inp_param'] not in ('FOH', 'ZOH'):
        return False, 'unsupported input interpolation'
    return True, model


@lru_cache(maxsize=1)
def registered_target():
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError('Native discretization requires a C++ compiler (macOS: xcode-select --install).')
    arch = platform.machine().lower()
    flags = ['-march=native'] if arch in ('x86_64', 'amd64') else ['-mcpu=native'] if arch in ('arm64', 'aarch64') else []
    if os.environ.get('ICORE_DISCRETIZATION_ARCH', 'native') == 'portable':
        flags = []
    source = ROOT / 'native_discretization/di_rk4.cc'
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    key = hashlib.sha256(source.read_bytes() + repr((compiler, version, flags, sys.platform, arch, jaxlib.__version__)).encode()).hexdigest()[:20]
    cache = ROOT / '.build/di_discretization'
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / ('libdi_' + key + ('.dylib' if sys.platform == 'darwin' else '.so'))
    if not output.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            target = Path(temporary) / output.name
            command = [*compiler, '-O3', *flags, '-std=c++17', '-fPIC',
                       '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                       '-I' + jax.ffi.include_dir(), str(source), '-o', str(target), '-lm']
            result = subprocess.run(command, capture_output=True, text=True)
            (cache / 'build.log').write_text(' '.join(command) + '\n' + result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError('Native double-integrator build failed:\n' + result.stderr)
            os.replace(target, output)
    name = 'icore_di_rk4_' + key
    if name not in _LIBRARIES:
        library = ct.CDLL(str(output))
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(library.ICOREDoubleIntegratorRk4), platform='cpu')
        _LIBRARIES[name] = library
    return name


def parameter_vector(params, model):
    keys = ['g_0', 'theta', 'T_max', 'v_max', 'w_states_theta', 'w_states_T_max', 'w_states_v_max', 't_f']
    if model == 'di_always_stc':
        keys += ['x_stc_1', 'x_stc_2', 'y_stc_1', 'x_stc_3', 'x_stc_4', 'y_stc_2']
    elif model == 'di_eventually':
        keys += ['p_w', 'r_w', 'p_w_2', 'r_w_2', 'p_w_3', 'r_w_3']
    elif model == 'di_until_aug':
        keys += ['p_w', 'r_w', 'spd_lim']
    values = []
    for key in keys:
        value = np.asarray(params[key], dtype=np.float64)
        if value.shape != ((3,) if key.startswith('p_w') else ()) or not np.isfinite(value).all():
            raise ValueError('Invalid native DI parameter: ' + key)
        values.extend(value.ravel())
    if params['t_f'] <= 0:
        raise ValueError('Nominal final time must be positive.')
    return jnp.asarray(values, dtype=jnp.float64)


def make_native_discretizer(params, *, terminal_only=False, states_only=False):
    valid, model = eligibility(params)
    if not valid:
        raise ValueError('Native double-integrator unavailable: ' + model)
    model_id, nx, _ = MODELS[model]
    nodes, points = int(params['K']), int(params['rk4_steps'])
    if nodes < 2 or points < 2:
        raise ValueError('At least two nodes and two RK4 points are required.')
    free, dilation = params['free_final_time'], params['time_dil']
    span = (1. if dilation else 1./(nodes-1)) if free else float(params['t_scp'])
    mode = 2 if states_only else 0 if terminal_only else 1
    shape = (nodes-1, points, nx) if states_only else (nodes-1, nx*(nx+9)) if terminal_only else (nodes-1, points, nx*(nx+9))
    config = jnp.asarray([model_id, nodes, points, int(params['inp_param']=='FOH'), int(free), mode], dtype=jnp.int32)
    physical = parameter_vector(params, model)
    call = jax.ffi.ffi_call(registered_target(), jax.ShapeDtypeStruct(shape, np.float64), vmap_method='sequential')

    def discretize(X, U, S):
        if X.shape != (nx, nodes) or U.shape != (3, nodes) or X.dtype != jnp.float64 or U.dtype != jnp.float64:
            raise ValueError('Native double-integrator requires matching float64 trajectories.')
        if free:
            if dilation:
                if S.shape != (1, nodes-1):
                    raise ValueError('Time-dilation S must have shape (1,K-1).')
                scales = durations = S[0]
            else:
                if S.shape != ():
                    raise ValueError('Global duration must be scalar.')
                scales = jnp.broadcast_to(S, (nodes-1,))
                durations = scales/(nodes-1)
        else:
            scales = jnp.ones(nodes-1, dtype=jnp.float64)
            durations = scales*params['t_scp']
        if states_only or (free and dilation):
            starts = jnp.cumsum(jnp.concatenate((jnp.zeros(1, dtype=X.dtype), durations)))[:-1]
        else:
            starts = jnp.arange(nodes-1)*S/(nodes-1) if free else jnp.arange(nodes-1)*params['t_scp']
        result = call(X.T, U.T, durations if states_only else scales, starts, physical, config, jnp.asarray(span, dtype=jnp.float64))
        return result.transpose(2, 0, 1) if states_only else result
    return jax.jit(discretize)
