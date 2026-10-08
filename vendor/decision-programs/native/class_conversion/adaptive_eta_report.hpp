#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>

// Reporting metadata only. Samples are reduced on CUDA by adaptive_work_summary.
// This arithmetic converts those observations and measured construction work to
// a conditional forecast; it neither predicts classes nor authorizes pruning.
namespace class_conversion_adaptive::eta {
struct Sample {
  std::uint64_t paths=0,done=0,censored=0,errors=0,overflows=0;
  bool valid_total=false,standard_error_available=false;
  double mean=0,standard_error=0,sampling_seconds=0;
};
class Reporter {
  struct Window { std::uint64_t work; double seconds; };
  std::deque<Window> windows_;
  Sample sample_;
  bool sampled_=false,observed_=false;
  std::uint64_t previous_work_=0;
  double previous_seconds_=0;
  std::string refusal_status_="waiting_for_sample",refusal_reason_="Bounded work sampling has not started.";
 public:
  void refuse(std::string status,std::string reason) {
    refusal_status_=std::move(status);refusal_reason_=std::move(reason);
  }
  void set_sample(const Sample& sample) {
    sample_=sample;sampled_=true;
    if(sample.censored)refuse("sampling_censored","Unfinished sampled paths prevent a total-work forecast.");
    else if(sample.errors||sample.overflows)refuse("sampling_invalid","Sample errors or weight overflow prevent a forecast.");
    else if(!sample.valid_total||!sample.paths||sample.done!=sample.paths||
        !std::isfinite(sample.mean)||sample.mean<0||!std::isfinite(sample.standard_error)||sample.standard_error<0)
      refuse("sampling_invalid","The CUDA work summary is not a finite completed total-work sample.");
    else if(sample.paths<2||!sample.standard_error_available)
      refuse("insufficient_samples","At least two completed paths and a finite observed standard error are required.");
    else refuse("warming_throughput","Waiting for compatible committed work and measured construction time.");
  }
  void observe(std::uint64_t work,double seconds) {
    if(!std::isfinite(seconds)||seconds<0)return;
    if(observed_&&work>=previous_work_&&seconds>previous_seconds_) {
      const auto delta=work-previous_work_;
      if(delta) {
        windows_.push_back({delta,seconds-previous_seconds_});
        if(windows_.size()>4)windows_.pop_front();
        previous_work_=work;previous_seconds_=seconds;
      }
    } else if(observed_&&(work<previous_work_||seconds<previous_seconds_)) {
      windows_.clear();previous_work_=work;previous_seconds_=seconds;
    }
    if(!observed_) {previous_work_=work;previous_seconds_=seconds;observed_=true;}
  }
  nlohmann::json report(std::uint64_t completed,bool complete,bool compatible,
                        const std::string& incompatible_reason="") const {
    using J=nlohmann::json;
    J out={{"available",false},{"seconds",nullptr},{"lower_seconds",nullptr},{"upper_seconds",nullptr},
      {"status",refusal_status_},{"reason",refusal_reason_},
      {"scope","conditional unshared-work forecast; descriptive variation, not a calibrated confidence interval or completion guarantee"},
      {"work_unit","committed split vertices + class prunes + native terminals + terminal-cache hits"},
      {"completed_decisions",completed},{"sample_count",sample_.paths},{"sampled_done",sample_.done},
      {"sampled_censored",sample_.censored},{"sampled_errors",sample_.errors},{"sampled_overflows",sample_.overflows},
      {"sampling_seconds",sample_.sampling_seconds},{"rate_windows",windows_.size()},
      {"standard_error_multiplier",sample_.standard_error_available?J(2):J(nullptr)},
      {"range_kind",sample_.standard_error_available?"descriptive two-standard-error work band combined with observed throughput range":"sampling variation unavailable"},
      {"future_cache_sharing_modeled",false},{"future_memory_growth_modeled",false}};
    if(complete) {out.update({{"available",true},{"seconds",0.},{"lower_seconds",0.},{"upper_seconds",0.},
        {"status","complete"},{"reason","Construction is complete; remaining construction time is zero."}});return out;}
    if(!compatible) {out["status"]="policy_mismatch";out["reason"]=incompatible_reason;return out;}
    if(!sampled_||!sample_.valid_total||sample_.censored||sample_.errors||sample_.overflows||
       sample_.paths<2||!sample_.standard_error_available||sample_.done!=sample_.paths||!std::isfinite(sample_.mean)||sample_.mean<0||
       !std::isfinite(sample_.standard_error)||sample_.standard_error<0)return out;
    out["estimated_total_decisions"]=sample_.mean;
    out["estimated_total_standard_error"]=sample_.standard_error_available?J(sample_.standard_error):J(nullptr);
    if(sample_.mean<=double(completed)) {out["status"]="contradicted";
      out["reason"]="Sampled total work does not exceed already committed work; no zero-time prediction is inferred.";return out;}
    if(windows_.empty())return out;
    double work=0,seconds=0,minimum_rate=0,maximum_rate=0;
    for(const auto& window:windows_) {
      const auto rate=double(window.work)/window.seconds;
      if(!std::isfinite(rate)||rate<=0)return out;
      work+=double(window.work);seconds+=window.seconds;
      minimum_rate=minimum_rate?std::min(minimum_rate,rate):rate;maximum_rate=std::max(maximum_rate,rate);
    }
    const auto rate=work/seconds;
    const double uncertainty=sample_.standard_error_available?2*sample_.standard_error:0;
    const double lower_work=std::max(double(completed),sample_.mean-uncertainty);
    const double upper_work=sample_.mean+uncertainty;
    const double point=(sample_.mean-double(completed))/rate;
    const double lower=(lower_work-double(completed))/maximum_rate;
    const double upper=(upper_work-double(completed))/minimum_rate;
    if(!std::isfinite(rate)||rate<=0||!std::isfinite(point)||!std::isfinite(lower)||!std::isfinite(upper)) {
      out["status"]="forecast_overflow";out["reason"]="The remaining-work or time conversion overflowed.";return out;
    }
    out.update({{"available",true},{"seconds",point},{"lower_seconds",lower},{"upper_seconds",upper},
      {"status","conditional_estimate"},
      {"reason","Frozen compatible unshared work, recent measured throughput; cache sharing, future growth and throughput drift may change completion time."},
      {"estimated_remaining_decisions",sample_.mean-double(completed)},
      {"observed_decisions_per_second",rate},{"minimum_observed_decisions_per_second",minimum_rate},
      {"maximum_observed_decisions_per_second",maximum_rate},{"rate_window_seconds",seconds},
      {"standard_error_available",sample_.standard_error_available}});
    return out;
  }
};
} // namespace class_conversion_adaptive::eta
