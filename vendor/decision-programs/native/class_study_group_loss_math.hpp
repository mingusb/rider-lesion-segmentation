#pragma once
// Shared scalar arithmetic for the versioned group-mean CE/Dice objective.
#include "class_study_train_loss.hpp"
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define CLASS_GROUP_LOSS_HD __host__ __device__
#else
#define CLASS_GROUP_LOSS_HD
#endif

namespace class_study::group_loss {
using u32 = std::uint32_t;
using u64 = unsigned long long;
enum Error : u32 {
  ok=0, bad_config=1u, bad_shape=2u, bad_group=4u, bad_label=8u,
  nonfinite_margin=16u, bad_mass=32u, bad_retention=64u,
  arithmetic_failure=128u, probability_failure=256u, fp32_range=512u
};
struct Config {
  double beta=0.5, eta=0.1, max_delta_step=0.25;
  // Explicit objective multiplier C. Use retained FIT row count for parity.
  double loss_scale=0;
  u32 classes=2;
  bool fit_only=true, require_all_positives=true;
  TrainingCurvaturePolicy curvature_policy=TrainingCurvaturePolicy::uniform_v1;
};
using GroupSpec=FitGroupSpec;
struct GroupStats {
  u64 retained[2]{0,0};
  double intersection=0, prediction_mass=0, ce_sum=0;
  double denominator=0, dice=0, ce=0, loss=0;
};
struct RowState { double positive=0, negative=0, slope=0, ce=0, weight=0; };
struct OutputPair { float g0=0,g1=0,h0=0,h1=0; };

CLASS_GROUP_LOSS_HD inline bool finite(double v) { return std::isfinite(v); }
CLASS_GROUP_LOSS_HD inline double radius(const Config& c) { return 2*c.eta*c.max_delta_step; }
CLASS_GROUP_LOSS_HD inline u32 validate_config(const Config& c, u64 rows, u32 groups) {
  if(!rows||!groups||rows>u64(SIZE_MAX/2/sizeof(float))) return bad_shape;
  if(!c.fit_only||c.classes!=2||!finite(c.beta)||c.beta<0||c.beta>1||
     !finite(c.eta)||c.eta<=0||c.eta>1||!finite(c.max_delta_step)||
     c.max_delta_step<=0||!finite(c.loss_scale)||c.loss_scale<=0||
     (c.curvature_policy!=TrainingCurvaturePolicy::uniform_v1&&
      c.curvature_policy!=TrainingCurvaturePolicy::class_row_sum_v2))
    return bad_config;
  const double r=radius(c);
  if(!finite(r)||!finite(::exp(2*r))||!finite(c.loss_scale/double(groups)))
    return bad_config;
  return ok;
}
CLASS_GROUP_LOSS_HD inline u32 validate_group(const GroupSpec& m,const GroupStats& s,
                                  const Config& c) {
  // Counts above 2^53 do not have exact integer representations in FP64.
  constexpr u64 exact=9007199254740992ULL;
  if(m.full_negative>exact||m.full_positive>exact||m.outside_positive>exact||
     m.full_negative>exact-m.full_positive||m.full_positive>exact-m.outside_positive||
     !finite(m.common_scale)||m.common_scale<=0)
    return bad_mass;
  const double q=double(m.full_negative+m.full_positive);
  const double t=double(m.full_positive)+double(m.outside_positive);
  if(!(q>0)||!finite(q)||!finite(t)) return bad_mass;
  if(s.retained[0]>m.full_negative||s.retained[1]>m.full_positive||
     (bool(s.retained[0])!=bool(m.full_negative))||
     (bool(s.retained[1])!=bool(m.full_positive))||
     (c.require_all_positives&&s.retained[1]!=m.full_positive)) return bad_retention;
  return ok;
}
CLASS_GROUP_LOSS_HD inline u32 row_state(float margin0,float margin1,u32 y,
                             const GroupSpec& m,const GroupStats& s,RowState& r) {
  if(y>1) return bad_label;
  if(!finite(double(margin0))||!finite(double(margin1))) return nonfinite_margin;
  // Cast each raw margin before subtracting; do not subtract in FP32.
  const double z=double(margin1)-double(margin0), e=::exp(-::fabs(z));
  const double d=1+e;
  if(!(e>0)||!finite(z)) return arithmetic_failure;
  r.positive=z>=0?1/d:e/d;
  r.negative=z>=0?e/d:1/d;
  // Independent of the rounded-to-one positive probability.
  r.slope=e/(d*d);
  const double yz=y?-z:z;
  r.ce=(yz>0?yz:0)+::log1p(e);
  const u64 full=y?m.full_positive:m.full_negative;
  if(!s.retained[y]||!full) return bad_retention;
  r.weight=double(full)/double(s.retained[y]);
  if(!(r.weight>0)||!finite(r.weight)||!(r.slope>0)||!finite(r.ce))
    return arithmetic_failure;
  return ok;
}
CLASS_GROUP_LOSS_HD inline u32 finalize_group(const GroupSpec& m,GroupStats& s,
                                  const Config& c) {
  const double q=double(m.full_negative+m.full_positive);
  const double t=double(m.full_positive)+double(m.outside_positive);
  s.denominator=s.prediction_mass+t;
  if(!finite(s.intersection)||!finite(s.prediction_mass)||!finite(s.ce_sum)||
     !(s.denominator>0)||!finite(s.denominator)) return arithmetic_failure;
  s.dice=m.full_positive?2*s.intersection/s.denominator:0;
  s.ce=s.ce_sum/q;
  if(!finite(s.dice)||s.dice<0||s.dice>1||!finite(s.ce)||s.ce<0)
    return probability_failure;
  const double beta=m.full_positive?c.beta:0;
  s.loss=(1-beta)*s.ce+beta*(1-s.dice);
  return finite(s.loss)?ok:arithmetic_failure;
}
CLASS_GROUP_LOSS_HD inline bool normal_or_zero(float f,double source) {
  // Refuse subnormal exports: downstream CUDA FTZ behavior is not assumed.
  return finite(double(f))&&(source==0?f==0:(f!=0&&::fabs(double(f))>=FLT_MIN));
}
CLASS_GROUP_LOSS_HD inline u32 row_output(const RowState& r,u32 y,const GroupSpec& m,
                              const GroupStats& s,const Config& c,u32 groups,
                              OutputPair& out) {
  const double q=double(m.full_negative+m.full_positive);
  const double beta=m.full_positive?c.beta:0, R=radius(c);
  // Do not evaluate inactive branches: 0*an-overflow is not a valid zero.
  const double gce=beta<1?(r.weight/q)*(y?-r.negative:r.positive):0;
  const double hce=beta<1?2*::exp(R)*(r.weight/q)*r.slope:0;
  const double gd=beta>0?r.weight*r.slope*(s.dice-2*double(y))/s.denominator:0;
  double hd=beta>0?8*::exp(2*R)*r.weight*r.slope/s.denominator:0;
  if(beta>0&&c.curvature_policy==TrainingCurvaturePolicy::class_row_sum_v2){
    // Real-box absolute Hessian row bounds, including both native logits:
    // y=0: 2 exp(2R) q/S Dhi(4-2Dhi); y=1: 2 exp(2R) q/S (4-2Dlo).
    // Q0<=P-A and Q1<=A give these without extra group reductions. The
    // ordinary FP64 operations/FP32 export are not a rounding certificate.
    const double expanded=::exp(2*R)*s.dice;
    const double upper=expanded<1?expanded:1,lower=::exp(-2*R)*s.dice;
    const double factor=y?1-lower/2:upper*(1-upper/2);
    if(!finite(factor)||!(factor>0)||factor>1)return arithmetic_failure;
    hd*=factor;
    // Refuse loss of a positive Dice curvature even if a CE term would hide it.
    if(!finite(hd)||!(hd>0))return arithmetic_failure;
  }
  const double mult=c.loss_scale/double(groups);
  const double g=mult*((1-beta)*gce+beta*gd);
  const double h=mult*((1-beta)*hce+beta*hd);
  // For this finite-logit hard-label contract every active term has the same
  // strict sign. A zero g is numerical loss, not cancellation or an optimum.
  if(!finite(g)||!finite(h)||!(h>0)||(y?!(g<0):!(g>0))) return arithmetic_failure;
  const float gf=float(g),hf=float(h);
  if(!normal_or_zero(gf,g)||!normal_or_zero(hf,h)) return fp32_range;
  // Deliberately ordinary FP32 cast, not a claim of guarded outward rounding.
  out={-gf,gf,hf,hf};
  return ok;
}
} // namespace class_study::group_loss
#undef CLASS_GROUP_LOSS_HD
