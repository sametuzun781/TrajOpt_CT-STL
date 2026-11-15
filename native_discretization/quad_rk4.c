/* FP64 fixed-grid RK4 with exact sparse analytic Jacobians and compressed
 * structural sensitivities. All physical parameters are runtime data.
 * No reduced step count, fast-math, thresholds, finite differences or allocation.
 */
#include <math.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif
#include "quad_field.c"
#include "sensitivity_plan.h"

int icore_quad_discretize(int nodes,int points,int foh,int free_time,int history,
                         double span,const double *X,const double *U,
                         const double *scales,const double *raw_parameters,double *output) {
    if(nodes<2 || points<2 || !(span>0) || !isfinite(span)) return 1;
    double parameters[QUAD_NPARAM+QUAD_NINVARIANT];
    memcpy(parameters,raw_parameters,QUAD_NPARAM*sizeof(double));
    quad_prepare(raw_parameters,parameters+QUAD_NPARAM);
    const double h=span/(points-1);
    for(int interval=0;interval<nodes-1;++interval) {
        double v[NV],tmp[NV],k1[NV],k2[NV],k3[NV],k4[NV];
        const double *x0=X+interval*NX,*u0=U+interval*NU,*u1=u0+NU;
        const double scale=scales[interval];
        double *out=output+(size_t)interval*(history?points:1)*OUT;
        if(!isfinite(scale) || scale<=0) return 4;
        initialize(v,x0);
        if(history) unpack(v,x0,u0,u1,scale,free_time,out);
        for(int step=0;step<points-1;++step) {
            double t=step*h;
            int error=rhs(v,u0,u1,scale,foh?t/span:0.,parameters,k1);
            if(error) return error;
            for(int j=0;j<NV;++j) {k1[j]*=h;tmp[j]=v[j]+0.5*k1[j];}
            error=rhs(tmp,u0,u1,scale,foh?(t+.5*h)/span:0.,parameters,k2);
            if(error) return error;
            for(int j=0;j<NV;++j) {k2[j]*=h;tmp[j]=v[j]+0.5*k2[j];}
            error=rhs(tmp,u0,u1,scale,foh?(t+.5*h)/span:0.,parameters,k3);
            if(error) return error;
            for(int j=0;j<NV;++j) {k3[j]*=h;tmp[j]=v[j]+k3[j];}
            error=rhs(tmp,u0,u1,scale,foh?(t+h)/span:0.,parameters,k4);
            if(error) return error;
            for(int j=0;j<NV;++j) {k4[j]*=h;v[j]+=(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6.;}
            int invalid=0;
            for(int j=0;j<NV;++j) invalid|=!isfinite(v[j]);
            if(invalid) return 5;
            if(history) unpack(v,x0,u0,u1,scale,free_time,out+(size_t)(step+1)*OUT);
        }
        if(!history) unpack(v,x0,u0,u1,scale,free_time,out);
    }
    return 0;
}
