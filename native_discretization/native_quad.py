"""Native FP64 quadrotor RK4, integrated into JAX via its typed CPU FFI.

This is a model-specialized CPU implementation, not a general replacement for
arbitrary JAX dynamics. Use discretization_mode='jax' for custom models, GPU, or
higher derivatives of the discretizer. The reference fixed RK4 grid is retained.
"""
from functools import lru_cache
from pathlib import Path
import ast
import ctypes as ct
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'native_discretization'
_LIBRARIES={}

def source_signature():
    source=(ROOT/'model/quad_flight/dynamics.py').read_text()
    utility=(ROOT/'ctstl/utils.py').read_text()
    rotation=next(n for n in ast.parse(utility).body if isinstance(n,ast.FunctionDef) and n.name=='rotation_matrix')
    return hashlib.sha256((source+'\n'+ast.get_source_segment(utility,rotation)).encode()).hexdigest()

def eligibility(params):
    if (params.get('n_x'),params.get('n_u'))!=(24,4):return False,'different state/input dimensions'
    if params.get('_native_quad_field') is not params.get('f_func'):return False,'unrecognized or replaced dynamics callback'
    layout=json.loads((SOURCE/'field_layout.json').read_text())
    if source_signature()!=layout['source_sha256']:return False,'model source changed; regenerate native field or use JAX'
    if jax.default_backend()!='cpu':return False,'native implementation supports the CPU backend only'
    if not jax.config.x64_enabled:return False,'native implementation requires JAX x64'
    if params['inp_param'] not in ('FOH','ZOH'):return False,'unsupported interpolation'
    return True,'exact model/source certificate'

@lru_cache(maxsize=8)
def build_library(cc_spec=None,cxx_spec=None):
    cc=shlex.split(cc_spec or os.environ.get('CC','cc'))
    cxx=shlex.split(cxx_spec or os.environ.get('CXX','c++'))
    for compiler in (cc,cxx):
        if not compiler or not shutil.which(compiler[0]):raise RuntimeError('Native discretization requires C/C++ compilers (macOS: xcode-select --install).')
    arch=['-march=native'] if platform.machine().lower() in ('x86_64','amd64') else ['-mcpu=native'] if platform.machine().lower() in ('arm64','aarch64') else []
    if os.environ.get('ICORE_DISCRETIZATION_ARCH','native')=='portable': arch=[]
    versions=[subprocess.check_output([*c,'--version'],text=True).splitlines()[0] for c in (cc,cxx)]
    files=['quad_rk4.c','quad_field.c','sensitivity_plan.h','quad_ffi.cc']
    key=hashlib.sha256(b''.join((SOURCE/f).read_bytes() for f in files)+repr((cc,cxx,versions,sys.platform,platform.machine(),jaxlib.__version__,arch)).encode()).hexdigest()[:20]
    cache=ROOT/'.build/quad_discretization';cache.mkdir(parents=True,exist_ok=True)
    output=cache/('libquad_'+key+('.dylib' if sys.platform=='darwin' else '.so'))
    if not output.exists():
        with tempfile.TemporaryDirectory(dir=cache) as d:
            d=Path(d);obj=d/'quad.o';temporary=d/output.name
            commands=[[*cc,'-O3',*arch,'-std=c11','-fPIC','-c',str(SOURCE/'quad_rk4.c'),'-o',str(obj)],
                      [*cxx,'-O3','-std=c++17','-fPIC','-dynamiclib' if sys.platform=='darwin' else '-shared',
                       '-I'+jax.ffi.include_dir(),str(SOURCE/'quad_ffi.cc'),str(obj),'-o',str(temporary),'-lm']]
            log=[]
            for command in commands:
                result=subprocess.run(command,text=True,capture_output=True)
                log.append(' '.join(command)+'\n'+result.stdout+result.stderr)
                (cache/'build.log').write_text('\n'.join(log))
                if result.returncode:raise RuntimeError('Native quadrotor build failed:\n'+result.stderr)
            os.replace(temporary,output)
    return output,key

def registered_target():
    path,key=build_library();name='icore_quad_rk4_'+key
    if name not in _LIBRARIES:
        lib=ct.CDLL(str(path));jax.ffi.register_ffi_target(name,jax.ffi.pycapsule(lib.ICOREQuadRk4),platform='cpu')
        _LIBRARIES[name]=lib
    return name

def parameter_vector(params):
    layout=json.loads((SOURCE/'field_layout.json').read_text())
    vector=np.empty(layout['parameter_count'],np.float64)
    for row in layout['parameters']:
        value=np.asarray(params[row['key']],dtype=np.float64)
        if list(value.shape)!=row['shape'] or not np.isfinite(value).all():raise ValueError('Invalid parameter '+row['key'])
        vector[row['offset']:row['offset']+row['size']]=value.ravel()
    if any(params[k]<=0 for k in ('I_x','I_y','I_z','t_f')):raise ValueError('Inertias and nominal final time must be positive.')
    return vector

def make_native_discretizer(params, *, terminal_only=False):
    valid,why=eligibility(params)
    if not valid:raise ValueError('Native quadrotor field is not applicable: '+why)
    nodes=int(params['K']);points=int(params['rk4_steps'])
    if nodes<2 or points<2:raise ValueError('At least two nodes and two grid points are required.')
    free=bool(params['free_final_time']);dilation=bool(params['time_dil'])
    span=(1.0 if dilation else 1.0/(nodes-1)) if free else float(params['t_scp'])
    physical=jnp.asarray(parameter_vector(params))
    config=jnp.asarray([nodes,points,int(params['inp_param']=='FOH'),int(free),int(not terminal_only),physical.size],dtype=jnp.int32)
    shape=(nodes-1,840) if terminal_only else (nodes-1,points,840)
    call=jax.ffi.ffi_call(registered_target(),jax.ShapeDtypeStruct(shape,np.float64),vmap_method='sequential')
    def discretize(X,U,S):
        if X.shape!=(24,nodes) or U.shape!=(4,nodes):raise ValueError('Trajectory shape mismatch.')
        if X.dtype!=jnp.float64 or U.dtype!=jnp.float64:raise ValueError('Native quadrotor requires float64 trajectories.')
        if free:
            if dilation:
                if S.shape!=(1,nodes-1):raise ValueError('Time-dilation S must have shape (1,K-1).')
                scales=S[0]
            else:
                if S.shape!=():raise ValueError('Global duration S must be scalar.')
                scales=jnp.broadcast_to(S,(nodes-1,))
        else:scales=jnp.ones(nodes-1,dtype=jnp.float64)
        return call(X.T,U.T,scales,physical,config,jnp.asarray(span,dtype=jnp.float64))
    return jax.jit(discretize)
