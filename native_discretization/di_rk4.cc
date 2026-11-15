// FP64 RK4 for the four certified double-integrator models. The reference
// sample grid, interpolation, time arguments, and augmented states are retained.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

template<int M, int N>
static void field(const double *x, const double *u, double tau, const double *p,
                  double *f, double *A, double *B) {
    for (int i=0; i<3; ++i) {
        f[i]=x[i+3]; f[i+3]=u[i];
        if (A) { A[i*N+i+3]=1.; B[(i+3)*3+i]=1.; }
    }
    f[5]-=p[0];
    if constexpr (M==0) { // Always avoid the two state-triggered regions.
        for (int row=6; row<8; ++row) {
            const int o=8+3*(row-6);
            const double sign=row==6?-1.:1.;
            const double a=std::min(0.,p[o]-x[0]);
            const double b=std::min(0.,x[0]-p[o+1]);
            const double c=std::min(0.,sign*(x[1]-p[o+2]));
            f[row]=1000.*a*a*b*b*c*c;
            if (A) {
                A[row*N]=2000.*(-a*b*b+a*a*b)*c*c;
                A[row*N+1]=2000.*a*a*b*b*c*sign;
            }
        }
    }
    if constexpr (M==1) { // Eventually visit three windows.
        for (int w=0; w<3; ++w) {
            const int o=8+4*w, row=6+2*w;
            double delta[3], distance=0.;
            for (int j=0; j<3; ++j) { delta[j]=x[j]-p[o+j]; distance+=delta[j]*delta[j]; }
            const double margin=p[o+3]*p[o+3]-distance;
            const double negative=std::min(margin/40.,0.);
            const double log_arg=negative*negative+1e-6;
            const double positive=std::max(60.*margin,0.);
            f[row]=x[row]*std::log(log_arg)/p[7];
            f[row+1]=positive*positive/p[7];
            if (A) {
                A[row*N+row]=std::log(log_arg)/p[7];
                for (int j=0; j<3; ++j) {
                    A[row*N+j]=-x[row]*negative*delta[j]/(10.*log_arg*p[7]);
                    A[(row+1)*N+j]=-240.*positive*delta[j]/p[7];
                }
            }
        }
    }
    if constexpr (M==3) { // Speed-limited until-target event states.
        const double speed_scale=p[7]*50., tau_safe=tau+1e-4;
        const double limit=p[12]-.06;
        double speed=0., distance=0., delta[3];
        for (int j=0; j<3; ++j) {
            speed+=x[j+3]*x[j+3]; delta[j]=x[j]-p[8+j]; distance+=delta[j]*delta[j];
        }
        const double violation=std::max(speed-limit*limit,0.);
        f[6]=violation*violation/speed_scale;
        const double q=std::max(x[6]*speed_scale/tau_safe,0.);
        const double d=std::max(10.*(distance-p[11]*p[11]),0.);
        const double zpos=std::sqrt(1e-4+.5*(d*d+q*q));
        const double z=(zpos-std::sqrt(1e-4))/50.;
        const double integrand=z*z+1e-12;
        f[7]=x[7]*std::log(integrand)/p[7];
        if (A) {
            const double factor=x[7]*z/(50.*p[7]*integrand*zpos);
            for (int j=0; j<3; ++j) {
                A[6*N+j+3]=4.*violation*x[j+3]/speed_scale;
                A[7*N+j]=factor*d*20.*delta[j];
            }
            A[7*N+6]=factor*q*speed_scale/tau_safe;
            A[7*N+7]=std::log(integrand)/p[7];
        }
    }
    // Common tilt, thrust, and speed penalties in the final augmented state.
    const int row=N-1;
    const double cosine=std::cos(p[1]), c2=cosine*cosine;
    const double tilt=std::max(0.,u[0]*u[0]+u[1]*u[1]-c2*u[2]*u[2]);
    const double thrust=std::max(0.,u[0]*u[0]+u[1]*u[1]+u[2]*u[2]-p[2]*p[2]);
    const double speed=std::max(0.,x[3]*x[3]+x[4]*x[4]+x[5]*x[5]-p[3]*p[3]);
    f[row]=p[4]*tilt*tilt+p[5]*thrust*thrust+p[6]*speed*speed;
    if (A) {
        for (int j=0; j<3; ++j) {
            A[row*N+j+3]=4.*p[6]*speed*x[j+3];
            B[row*3+j]=4.*u[j]*(p[5]*thrust+p[4]*tilt*(j==2?-c2:1.));
        }
    }
}

template<int M, int N>
static void rhs(const double *v, const double *u0, const double *u1, double scale,
                double beta, double tau, const double *p, bool sensitivities, double *dv) {
    constexpr int T=N+7, NV=N*(T+1);
    double u[3], f[N]={}, A[N*N]={}, B[N*3]={};
    for (int j=0; j<3; ++j) u[j]=u0[j]+beta*(u1[j]-u0[j]);
    field<M,N>(v,u,tau,p,f,sensitivities?A:nullptr,sensitivities?B:nullptr);
    for (int i=0; i<N; ++i) dv[i]=scale*f[i];
    if (!sensitivities) return;
    std::fill(dv+N,dv+NV,0.);
    for (int i=0; i<N; ++i) {
        double *row=dv+N+i*T;
        for (int j=0; j<N; ++j) {
            const double a=scale*A[i*N+j];
            if (a!=0.) { // Exact zeros only; no sparsity threshold.
                const double *tangent=v+N+j*T;
                for (int c=0; c<T; ++c) row[c]+=a*tangent[c];
            }
        }
        for (int j=0; j<3; ++j) {
            row[N+j]+=scale*B[i*3+j]*(1.-beta);
            row[N+3+j]+=scale*B[i*3+j]*beta;
        }
        row[T-1]+=f[i];
    }
}

template<int N>
static void unpack(const double *v, const double *x0, const double *u0, const double *u1,
                   double scale, bool free_time, double *out) {
    constexpr int T=N+7, IA=N, IB=IA+N*N, IC=IB+N*3, IS=IC+N*3, IZ=IS+N;
    std::copy(v,v+N,out);
    for (int i=0; i<N; ++i) {
        const double *row=v+N+i*T;
        double z=v[i];
        for (int j=0; j<N; ++j) { out[IA+i*N+j]=row[j]; z-=row[j]*x0[j]; }
        for (int j=0; j<3; ++j) {
            out[IB+i*3+j]=row[N+j]; out[IC+i*3+j]=row[N+3+j];
            z-=row[N+j]*u0[j]; z-=row[N+3+j]*u1[j];
        }
        out[IS+i]=row[T-1];
        out[IZ+i]=free_time?z-row[T-1]*scale:z;
    }
}

template<int M, int N>
static bool integrate(int nodes, int points, bool foh, bool free_time, int mode, double span,
                      const double *X, const double *U, const double *scales,
                      const double *starts, const double *p, double *out) {
    constexpr int T=N+7, NV=N*(T+1), OUT=N*(N+9);
    const bool sensitivity=mode!=2, history=mode!=0;
    const int nv=sensitivity?NV:N, width=sensitivity?OUT:N;
    for (int interval=0; interval<nodes-1; ++interval) {
        double v[NV]={}, tmp[NV], k1[NV], k2[NV], k3[NV], k4[NV];
        const double *x0=X+interval*N, *u0=U+interval*3, *u1=u0+3;
        const double scale=sensitivity?scales[interval]:1.;
        const double duration=sensitivity?span:scales[interval];
        const double h=duration/(points-1), start=starts[interval];
        double *output=out+int64_t(interval)*(history?points:1)*width;
        std::copy(x0,x0+N,v);
        if (sensitivity) for (int i=0; i<N; ++i) v[N+i*T+i]=1.;
        auto store=[&](double *dest) {
            if (sensitivity) unpack<N>(v,x0,u0,u1,scale,free_time,dest);
            else std::copy(v,v+N,dest);
        };
        if (history) store(output);
        for (int step=0; step<points-1; ++step) {
            const double t=step*h;
            rhs<M,N>(v,u0,u1,scale,foh?t/duration:0.,start+t,p,sensitivity,k1);
            for (int j=0; j<nv; ++j) { k1[j]*=h; tmp[j]=v[j]+.5*k1[j]; }
            rhs<M,N>(tmp,u0,u1,scale,foh?(t+.5*h)/duration:0.,start+t+.5*h,p,sensitivity,k2);
            for (int j=0; j<nv; ++j) { k2[j]*=h; tmp[j]=v[j]+.5*k2[j]; }
            rhs<M,N>(tmp,u0,u1,scale,foh?(t+.5*h)/duration:0.,start+t+.5*h,p,sensitivity,k3);
            for (int j=0; j<nv; ++j) { k3[j]*=h; tmp[j]=v[j]+k3[j]; }
            rhs<M,N>(tmp,u0,u1,scale,foh?(t+h)/duration:0.,start+t+h,p,sensitivity,k4);
            for (int j=0; j<nv; ++j) {
                k4[j]*=h; v[j]+=(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6.;
                if (!std::isfinite(v[j])) return false;
            }
            if (history) store(output+int64_t(step+1)*width);
        }
        if (!history) store(output);
    }
    return true;
}

static ffi::Error DoubleIntegratorRk4(ffi::BufferR2<ffi::F64> x, ffi::BufferR2<ffi::F64> u,
    ffi::BufferR1<ffi::F64> scales, ffi::BufferR1<ffi::F64> starts,
    ffi::BufferR1<ffi::F64> parameters, ffi::BufferR1<ffi::S32> config,
    ffi::BufferR0<ffi::F64> span, ffi::ResultBuffer<ffi::F64> result) {
    if (config.element_count()!=6) return ffi::Error::InvalidArgument("DI RK4 config needs six entries");
    const int32_t *c=config.typed_data();
    const int model=c[0], nodes=c[1], points=c[2], foh=c[3], free_time=c[4], mode=c[5];
    const int dimensions[]={9,13,7,9}, parameter_counts[]={14,20,8,13};
    if (model<0 || model>3 || nodes<2 || points<2 || (foh!=0&&foh!=1) ||
        (free_time!=0&&free_time!=1) || mode<0 || mode>2)
        return ffi::Error::InvalidArgument("Invalid DI RK4 configuration");
    const int nx=dimensions[model], width=mode==2?nx:nx*(nx+9);
    if (x.dimensions()[0]!=nodes || x.dimensions()[1]!=nx || u.dimensions()[0]!=nodes ||
        u.dimensions()[1]!=3 || scales.element_count()!=nodes-1 || starts.element_count()!=nodes-1 ||
        parameters.element_count()!=parameter_counts[model] ||
        result->element_count()!=int64_t(nodes-1)*(mode==0?1:points)*width)
        return ffi::Error::InvalidArgument("DI RK4 buffer shape mismatch");
    auto finite=[](const double *buffer, int64_t count) {
        for (int64_t i=0; i<count; ++i) if (!std::isfinite(buffer[i])) return false;
        return true;
    };
    if (!finite(x.typed_data(),x.element_count()) || !finite(u.typed_data(),u.element_count()) ||
        !finite(parameters.typed_data(),parameters.element_count()))
        return ffi::Error::InvalidArgument("Nonfinite DI input");
    const double length=span.typed_data()[0];
    if (!(length>0.) || !std::isfinite(length) || !(parameters.typed_data()[7]>0.))
        return ffi::Error::InvalidArgument("Invalid DI integration duration");
    for (int i=0; i<nodes-1; ++i)
        if (!(scales.typed_data()[i]>0.) || !std::isfinite(scales.typed_data()[i]) || !std::isfinite(starts.typed_data()[i]))
            return ffi::Error::InvalidArgument("Invalid DI interval time");
    bool ok=false;
    #define RUN(M,N) integrate<M,N>(nodes,points,foh,free_time,mode,length,x.typed_data(),u.typed_data(),scales.typed_data(),starts.typed_data(),parameters.typed_data(),result->typed_data())
    switch (model) { case 0:ok=RUN(0,9);break; case 1:ok=RUN(1,13);break;
                     case 2:ok=RUN(2,7);break; case 3:ok=RUN(3,9);break; }
    #undef RUN
    return ok?ffi::Error::Success():ffi::Error::InvalidArgument("Nonfinite DI RK4 result");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ICOREDoubleIntegratorRk4,DoubleIntegratorRk4,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::F64>>().Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>().Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>().Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR0<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>());
