// JAX CPU FFI boundary. JAX owns the buffers; no Python callback or host copy.
#include <cmath>
#include <cstdint>
#include <limits>
#include "xla/ffi/api/ffi.h"
namespace ffi=xla::ffi;
extern "C" int icore_quad_discretize(int,int,int,int,int,double,const double*,const double*,const double*,const double*,double*);
static ffi::Error QuadRk4(ffi::BufferR2<ffi::F64> x,ffi::BufferR2<ffi::F64> u,
    ffi::BufferR1<ffi::F64> scales,ffi::BufferR1<ffi::F64> parameters,
    ffi::BufferR1<ffi::S32> config,ffi::BufferR0<ffi::F64> span,
    ffi::ResultBuffer<ffi::F64> result) {
    if(config.element_count()!=6) return ffi::Error::InvalidArgument("Quadrotor RK4 configuration must have six entries");
    const int32_t *c=config.typed_data();
    const int nodes=c[0],points=c[1],foh=c[2],free_time=c[3],history=c[4],nparam=c[5];
    if(nodes<2 || points<2 || (foh!=0&&foh!=1) || (free_time!=0&&free_time!=1) || (history!=0&&history!=1) ||
       x.dimensions()[0]!=nodes || x.dimensions()[1]!=24 || u.dimensions()[0]!=nodes || u.dimensions()[1]!=4 ||
       scales.element_count()!=nodes-1 || parameters.element_count()!=nparam || nparam!=50 ||
       result->element_count()!=int64_t(nodes-1)*(history?points:1)*840)
        return ffi::Error::InvalidArgument("Quadrotor RK4 buffer shape mismatch");
    for(int64_t i=0;i<x.element_count();++i) if(!std::isfinite(x.typed_data()[i])) return ffi::Error::InvalidArgument("Nonfinite state");
    for(int64_t i=0;i<u.element_count();++i) if(!std::isfinite(u.typed_data()[i])) return ffi::Error::InvalidArgument("Nonfinite input");
    for(int64_t i=0;i<parameters.element_count();++i) if(!std::isfinite(parameters.typed_data()[i])) return ffi::Error::InvalidArgument("Nonfinite model parameter");
    int error=icore_quad_discretize(nodes,points,foh,free_time,history,span.typed_data()[0],x.typed_data(),u.typed_data(),scales.typed_data(),parameters.typed_data(),result->typed_data());
    if(error) return ffi::Error::InvalidArgument("Quadrotor RK4 rejected a nonfinite value or invalid duration");
    return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(ICOREQuadRk4,QuadRk4,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::F64>>().Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>().Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());
