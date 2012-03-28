//----------------------------------------------------------------------
#ifndef simd_h
#define simd_h
#include <emmintrin.h>
//----------------------------------------------------------------------
class v2df {
  public:
    __m128d value;
    v2df(const double &v1, const double &v2) {value = _mm_set_pd(v2,v1);}
    v2df(const __m128d& v) {value = v;}
    v2df(const v2df& v0){value = v0.value;}
    v2df operator+(const v2df& v0) {return v2df(_mm_add_pd(value, v0.value));}
    v2df operator-(const v2df& v0) {return v2df(_mm_sub_pd(value, v0.value));}
    v2df operator*(const v2df& v0) {return v2df(_mm_mul_pd(value, v0.value));}
    v2df operator/(const v2df& v0) {return v2df(_mm_div_pd(value, v0.value));}
    v2df operator+=(const v2df& v0) {value = _mm_add_pd(value, v0.value); return *this;}
    v2df operator-=(const v2df& v0) {value = _mm_sub_pd(value, v0.value); return *this;}
    double operator[] (const int i){double *p = (double*)(&value);return p[i];};
    v2df max(const v2df& v0) {return v2df(_mm_max_pd(value, v0.value));}
    v2df min(const v2df& v0) {return v2df(_mm_min_pd(value, v0.value));}
};
//----------------------------------------------------------------------
#endif
