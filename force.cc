#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include "simd.h"

//----------------------------------------------------------------------
const int LI = 30;
//const int LI = 2;
const int N = LI*LI*LI;
const double L = (double)LI;
const int D = 3;
const int X = 0;
const int Y = 1;
const int Z = 2;
double q[N+1][D],p[N][D];
const double dt = 0.001;
const double C2 = 0.1;
//----------------------------------------------------------------------
double
myrand(void){
  return static_cast<double>(rand())/(static_cast<double>(RAND_MAX));
}
//----------------------------------------------------------------------
double
myclock(void){
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec*1e-6;
}
//----------------------------------------------------------------------
void
calcforce(void){
  for(int i=0;i<N-1;i++){
    for(int j=i+1;j<N;j++){
      double dx = q[j][X] - q[i][X];
      double dy = q[j][Y] - q[i][Y];
      double dz = q[j][Z] - q[i][Z];
      double r2 = (dx*dx + dy*dy + dz*dz);
      double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2)+C2*8.0)*dt;
#ifdef DEBUG
      printf("%d %d %f\n",i,j,df);
#endif
      p[i][X] += df*dx;
      p[i][Y] += df*dy;
      p[i][Z] += df*dz;
      p[j][X] -= df*dx;
      p[j][Y] -= df*dy;
      p[j][Z] -= df*dz;
    }
  }
}
//----------------------------------------------------------------------
// place i-particle on register
//----------------------------------------------------------------------
void
calcforce2(void){
  for(int i=0;i<N-1;i++){
    const double qix = q[i][X];
    const double qiy = q[i][Y];
    const double qiz = q[i][Z];
    double pix = p[i][X];
    double piy = p[i][Y];
    double piz = p[i][Z];
    for(int j=i+1;j<N;j++){
      double dx = q[j][X] - qix;
      double dy = q[j][Y] - qiy;
      double dz = q[j][Z] - qiz;
      double r2 = (dx*dx + dy*dy + dz*dz);
      double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2)+C2*8.0)*dt;
      pix += df*dx;
      piy += df*dy;
      piz += df*dz;
      p[j][X] -= df*dx;
      p[j][Y] -= df*dy;
      p[j][Z] -= df*dz;
    }
    p[i][X] = pix;
    p[i][Y] = piy;
    p[i][Z] = piz;
  }
}
//----------------------------------------------------------------------
// software pipelining 
//----------------------------------------------------------------------
void
calcforce_sp(void){
  for(int i=0;i<N-1;i++){
    const double qix = q[i][X];
    const double qiy = q[i][Y];
    const double qiz = q[i][Z];
    double pix = p[i][X];
    double piy = p[i][Y];
    double piz = p[i][Z];
    double dxt = q[i+1][X] - qix;
    double dyt = q[i+1][Y] - qiy;
    double dzt = q[i+1][Z] - qiz;
    for(int j=i+1;j<N;j++){
      double dx = dxt;
      double dy = dyt;
      double dz = dzt;
      double r2 = (dx*dx + dy*dy + dz*dz);
      double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2)+C2*8.0)*dt;
#ifdef DEBUG
      printf("%f\n",df);
#endif
      dxt = q[j+1][X] - qix;
      dyt = q[j+1][Y] - qiy;
      dzt = q[j+1][Z] - qiz;
      pix += df*dx;
      piy += df*dy;
      piz += df*dz;
      p[j][X] -= df*dx;
      p[j][Y] -= df*dy;
      p[j][Z] -= df*dz;
    }
    p[i][X] = pix;
    p[i][Y] = piy;
    p[i][Z] = piz;
  }
}
//----------------------------------------------------------------------
// software pipelining and loop unrolling
//----------------------------------------------------------------------
void
calcforce_sp_unroll(void){
  for(int i=0;i<N-1;i++){
    const double qix = q[i][X];
    const double qiy = q[i][Y];
    const double qiz = q[i][Z];
    double pix = p[i][X];
    double piy = p[i][Y];
    double piz = p[i][Z];
    double dxt_a = q[i+1][X] - qix;
    double dyt_a = q[i+1][Y] - qiy;
    double dzt_a = q[i+1][Z] - qiz;
    if(i+1 == N-1){
      double dx_a = dxt_a;
      double dy_a = dyt_a;
      double dz_a = dzt_a;
      double r2_a = (dx_a*dx_a + dy_a*dy_a + dz_a*dz_a);
      double r6_a = r2_a*r2_a*r2_a;
      double df_a = ((24.0*r6_a-48.0)/(r6_a*r6_a*r2_a)+C2*8.0)*dt;
#ifdef DEBUG
      printf("%d %d %f\n",i,i+1,df_a);
#endif
      p[i+1][X] -= df_a*dx_a;
      p[i+1][Y] -= df_a*dy_a;
      p[i+1][Z] -= df_a*dz_a;
      p[i][X] += df_a*dx_a;
      p[i][Y] += df_a*dy_a;
      p[i][Z] += df_a*dz_a;
      continue;
    }
    double dxt_b = q[i+2][X] - qix;
    double dyt_b = q[i+2][Y] - qiy;
    double dzt_b = q[i+2][Z] - qiz;
    for(int j=i+1;j<N-1;j+=2){
      double dx_a = dxt_a;
      double dy_a = dyt_a;
      double dz_a = dzt_a;
      double dx_b = dxt_b;
      double dy_b = dyt_b;
      double dz_b = dzt_b;
      double r2_a = (dx_a*dx_a + dy_a*dy_a + dz_a*dz_a);
      double r2_b = (dx_b*dx_b + dy_b*dy_b + dz_b*dz_b);
      double r6_a = r2_a*r2_a*r2_a;
      double r6_b = r2_b*r2_b*r2_b;
      double df_a = ((24.0*r6_a-48.0)/(r6_a*r6_a*r2_a)+C2*8.0)*dt;
      double df_b = ((24.0*r6_b-48.0)/(r6_b*r6_b*r2_b)+C2*8.0)*dt;
#ifdef DEBUG
      printf("%d %d %f\n",i,j,df_a);
      printf("%d %d %f\n",i,j+1,df_b);
#endif
      dxt_a = q[j+2][X] - qix;
      dyt_a = q[j+2][Y] - qiy;
      dzt_a = q[j+2][Z] - qiz;
      dxt_b = q[j+3][X] - qix;
      dyt_b = q[j+3][Y] - qiy;
      dzt_b = q[j+3][Z] - qiz;
      pix += df_a*dx_a;
      piy += df_a*dy_a;
      piz += df_a*dz_a;
      pix += df_b*dx_b;
      piy += df_b*dy_b;
      piz += df_b*dz_b;
      p[j][X] -= df_a*dx_a;
      p[j][Y] -= df_a*dy_a;
      p[j][Z] -= df_a*dz_a;
      p[j+1][X] -= df_b*dx_b;
      p[j+1][Y] -= df_b*dy_b;
      p[j+1][Z] -= df_b*dz_b;
    }
    if ((N-i+1)%2==1){
      double dx_a = dxt_a;
      double dy_a = dyt_a;
      double dz_a = dzt_a;
      double r2_a = (dx_a*dx_a + dy_a*dy_a + dz_a*dz_a);
      double r6_a = r2_a*r2_a*r2_a;
      double df_a = ((24.0*r6_a-48.0)/(r6_a*r6_a*r2_a)+C2*8.0)*dt;
#ifdef DEBUG
      printf("%d %d %f\n",i,N-1,df_a);
#endif
      p[N-1][X] -= df_a*dx_a;
      p[N-1][Y] -= df_a*dy_a;
      p[N-1][Z] -= df_a*dz_a;
      p[i][X] += df_a*dx_a;
      p[i][Y] += df_a*dy_a;
      p[i][Z] += df_a*dz_a;
    }

    p[i][X] = pix;
    p[i][Y] = piy;
    p[i][Z] = piz;
  }
}
//----------------------------------------------------------------------
// software pipelining and loop unrolling and SIMDized
//----------------------------------------------------------------------
void
calcforce_sp_unroll_simd(void){
  for(int i=0;i<N-1;i++){
    const double qix = q[i][X];
    const double qiy = q[i][Y];
    const double qiz = q[i][Z];
    double pix = 0.0;
    double piy = 0.0;
    double piz = 0.0;
    double dxt_a = q[i+1][X] - qix;
    double dyt_a = q[i+1][Y] - qiy;
    double dzt_a = q[i+1][Z] - qiz;
    if(i+1 == N-1){
      double dx_a = dxt_a;
      double dy_a = dyt_a;
      double dz_a = dzt_a;
      double r2_a = (dx_a*dx_a + dy_a*dy_a + dz_a*dz_a);
      double r6_a = r2_a*r2_a*r2_a;
      double df_a = ((24.0*r6_a-48.0)/(r6_a*r6_a*r2_a)+C2*8.0)*dt;
#ifdef DEBUG
      printf("%d %d %f\n",i,i+1,df_a);
#endif
      p[i+1][X] -= df_a*dx_a;
      p[i+1][Y] -= df_a*dy_a;
      p[i+1][Z] -= df_a*dz_a;
      p[i][X] += df_a*dx_a;
      p[i][Y] += df_a*dy_a;
      p[i][Z] += df_a*dz_a;
      continue;
    }
    double dxt_b = q[i+2][X] - qix;
    double dyt_b = q[i+2][Y] - qiy;
    double dzt_b = q[i+2][Z] - qiz;
    v2df pixv(pix,pix);
    v2df piyv(piy,piy);
    v2df pizv(piz,piz);
    v2df qixv(qix,qix);
    v2df qiyv(qiy,qiy);
    v2df qizv(qiz,qiz);
    v2df dxtv(dxt_a,dxt_b);
    v2df dytv(dyt_a,dyt_b);
    v2df dztv(dzt_a,dzt_b);
    for(int j=i+1;j<N-1;j+=2){
      v2df dx = dxtv;
      v2df dy = dytv;
      v2df dz = dztv;
      v2df c24(24.0,24.0);
      v2df c48(48.0,48.0);
      v2df c2_8(C2*8.0,C2*8.0);
      v2df dtv(dt,dt);
      v2df r2 = dx * dx + dy*dy + dz*dz;
      v2df r6 = r2*r2*r2;
      v2df df = ((c24*r6-c48)/(r6*r6*r2)+c2_8)*dtv;
      double df_a = df[0];
      double df_b = df[1];
      v2df dfx = df*dx;
      v2df dfy = df*dy;
      v2df dfz = df*dz;
#ifdef DEBUG
      printf("%d %d %f\n",i,j,df_a);
      printf("%d %d %f\n",i,j+1,df_b);
#endif
      v2df qjxv(q[j+2][X],q[j+3][X]);
      v2df qjyv(q[j+2][Y],q[j+3][Y]);
      v2df qjzv(q[j+2][Z],q[j+3][Z]);
      dxtv = qjxv - qixv;
      dytv = qjyv - qiyv;
      dztv = qjzv - qizv;
      pixv += dfx;
      piyv += dfy;
      pizv += dfz;
      p[j][X] -= dfx[0];
      p[j][Y] -= dfy[0];
      p[j][Z] -= dfz[0];
      p[j+1][X] -= dfx[1];
      p[j+1][Y] -= dfy[1];
      p[j+1][Z] -= dfz[1];
    }
    pix = pixv[0] + pixv[1];
    piy = piyv[0] + piyv[1];
    piz = pizv[0] + pizv[1];
    if ((N-i+1)%2==1){
      double dx_a = dxtv[0];
      double dy_a = dytv[0];
      double dz_a = dztv[0];
      double r2_a = (dx_a*dx_a + dy_a*dy_a + dz_a*dz_a);
      double r6_a = r2_a*r2_a*r2_a;
      double df_a = ((24.0*r6_a-48.0)/(r6_a*r6_a*r2_a)+C2*8.0)*dt;
#ifdef DEBUG
      printf("%d %d %f\n",i,N-1,df_a);
#endif
      p[N-1][X] -= df_a*dx_a;
      p[N-1][Y] -= df_a*dy_a;
      p[N-1][Z] -= df_a*dz_a;
      p[i][X] += df_a*dx_a;
      p[i][Y] += df_a*dy_a;
      p[i][Z] += df_a*dz_a;
    }

    p[i][X] += pix;
    p[i][Y] += piy;
    p[i][Z] += piz;
  }
}
//----------------------------------------------------------------------
void
init(void){
  srand(12345);
  for(int i=0;i<N;i++){
    p[i][X] = 0.0;
    p[i][Y] = 0.0;
    p[i][Z] = 0.0;
  } 
  for (int ix=0;ix<LI;ix++){
    for (int iy=0;iy<LI;iy++){
      for (int iz=0;iz<LI;iz++){
        const int i = ix*LI*LI + iy*LI + iz;
        q[i][X] = (double)ix+myrand()*0.1;
        q[i][Y] = (double)iy+myrand()*0.1;
        q[i][Z] = (double)iz+myrand()*0.1;
      }
    } 
  }
}
//----------------------------------------------------------------------
void
measure(void(*pfunc)(),const char *name){
  init();
  double st = myclock();
  pfunc();
  double t = myclock()-st;
  printf("N=%d, %s %f [sec]\n",N,name,t);
}
//----------------------------------------------------------------------
int
main(void){
  measure(&calcforce,"calcforce");
  measure(&calcforce2,"calcforce2");
  measure(&calcforce_sp,"calcforce_sp");
  measure(&calcforce_sp_unroll,"calcforce_sp_unrool");
  measure(&calcforce_sp_unroll_simd,"calcforce_sp_unrool_simd");
/*
  for(int i=0;i<N;i++){
    printf("%f %f %f\n",p[i][X],p[i][Y],p[i][Z]);
  }
*/
}
//----------------------------------------------------------------------
