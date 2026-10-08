#include "class_study_train.hpp"
#include "class_study_data.hpp"
#include "class_study_group_loss_cuda.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using J=nlohmann::json;
void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
  T*p=nullptr;std::size_t n=0;
  explicit Device(std::size_t size):n(size){ck(cudaMalloc(&p,n*sizeof(T)));}
  explicit Device(const std::vector<T>&v):Device(v.size()){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
  ~Device(){if(p)cudaFree(p);}
  Device(const Device&)=delete;Device&operator=(const Device&)=delete;
};
std::vector<float>copy(const float*p,std::size_t n){std::vector<float>v(n);ck(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}
bool bits_equal(const std::vector<float>&a,const std::vector<float>&b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin(),[](float x,float y){return std::bit_cast<std::uint32_t>(x)==std::bit_cast<std::uint32_t>(y);});}
struct Source {
  Device<float>x,weights;Device<std::uint32_t>y;
  Source(const std::vector<float>&values,const std::vector<std::uint32_t>&labels,const std::vector<float>&w):x(values),weights(w),y(labels){}
};
struct Groups {
  Device<std::uint32_t>ids;Device<class_study::FitGroupSpec>spec;
  Groups(const std::vector<std::uint32_t>&i,const std::vector<class_study::FitGroupSpec>&s):ids(i),spec(s){}
};
class_study::ResidentDataView view(const std::shared_ptr<Source>&s,std::uint32_t F=1){
  class_study::ResidentDataView v;v.values=s->x.p;v.labels=s->y.p;v.sample_weights=s->weights.p;
  v.rows=s->y.n;v.row_stride=F;v.features=F;v.classes=2;v.fit_rows=v.rows;v.owner=s;
  v.binding={{"format","synthetic-group-loss-fixture-1"},{"ordered_row_ids",J::array({41,7,93,12})},{"FIT_only",true}};return v;
}
class_study::FitGroupView group_view(const class_study::ResidentDataView&v,const std::shared_ptr<Groups>&g){
  return {g->ids.p,g->spec.p,v.rows,std::uint32_t(g->spec.n),v.binding,v.owner,g};
}
J hp(unsigned rounds=1){return {{"rounds",rounds},{"max_depth",1},{"eta",0.2},{"max_delta_step",0.25},{"lambda",0},{"alpha",0},{"gamma",1e6},{"min_child_weight",0},{"max_bin",16},{"seed",19},{"subsample",1}};}
unsigned curvature_scalar_checks(){
  using namespace class_study;using namespace class_study::group_loss;
  unsigned checks=0;
  const std::array<std::array<float,8>,2> margins={std::array<float,8>{0,0,0,0,0,0,0,0},
    std::array<float,8>{-1,1,1,-1,-4,4,4,-4}};
  for(unsigned kind=0;kind<4;++kind)for(const auto&margin:margins){
    const std::array<u32,4> labels=kind==1?std::array<u32,4>{1,1,1,1}:
      kind>=2?std::array<u32,4>{0,0,0,0}:std::array<u32,4>{0,0,0,1};
    GroupSpec spec=kind==1?GroupSpec{0,4,2,1}:kind>=2?GroupSpec{80,0,kind==3?9ULL:0ULL,1}:GroupSpec{300,1,7,1};
    GroupStats stats;for(auto y:labels)++stats.retained[y];
    Config config;config.loss_scale=7;
    need(validate_config(config,4,2)==ok&&validate_group(spec,stats,config)==ok,"scalar fixture configuration differs");
    std::array<RowState,4> rows;
    for(unsigned i=0;i<4;++i){
      need(row_state(margin[2*i],margin[2*i+1],labels[i],spec,stats,rows[i])==ok,"scalar row state differs");
      const auto&r=rows[i];stats.intersection+=r.weight*labels[i]*r.positive;
      stats.prediction_mass+=r.weight*r.positive;stats.ce_sum+=r.weight*r.ce;
    }
    for(double beta:std::array<double,3>{0,.5,1}){
      config.beta=beta;need(finalize_group(spec,stats,config)==ok,"scalar group finalization differs");
      auto tighter=config;tighter.curvature_policy=TrainingCurvaturePolicy::class_row_sum_v2;
      for(unsigned i=0;i<4;++i){
        OutputPair old{},now{};need(row_output(rows[i],labels[i],spec,stats,config,2,old)==ok&&
          row_output(rows[i],labels[i],spec,stats,tighter,2,now)==ok,"valid curvature output refused");
        need(std::bit_cast<u32>(old.g0)==std::bit_cast<u32>(now.g0)&&
          std::bit_cast<u32>(old.g1)==std::bit_cast<u32>(now.g1),"curvature changed gradient words");
        if(beta==0||spec.full_positive==0){
          need(std::bit_cast<u32>(old.h0)==std::bit_cast<u32>(now.h0)&&
            std::bit_cast<u32>(old.h1)==std::bit_cast<u32>(now.h1),"CE curvature words changed");
        }else{
          // Independent long-double formula for the approved row bound. The
          // private AD oracle separately establishes the dense-box inequality.
          const long double z=static_cast<long double>(margin[2*i+1])-margin[2*i];
          const long double e=std::exp(-std::fabs(z)),slope=e/((1+e)*(1+e));
          const long double R=2*static_cast<long double>(config.eta)*config.max_delta_step;
          const long double E=std::exp(2*R),D=stats.dice,S=stats.denominator;
          const long double lo=std::exp(-2*R)*D,hi=std::min(1.L,E*D),q=rows[i].weight*slope;
          const long double hd=2*E*q/S*(labels[i]?4-2*lo:hi*(4-2*hi));
          const long double hc=2*std::exp(R)*q/(spec.full_negative+spec.full_positive);
          const long double expected=(config.loss_scale/2)*((1-beta)*hc+beta*hd);
          need(std::fabs(static_cast<long double>(now.h1)-expected)<=4*FLT_EPSILON*expected,
            "class row-sum curvature differs from oracle");
          need(now.h0==now.h1&&now.h1<=old.h1,"class row-sum curvature widened");
        }
        ++checks;
      }
    }
  }
  Config config;config.loss_scale=2;
  config.curvature_policy=static_cast<TrainingCurvaturePolicy>(99);
  need(validate_config(config,2,1)==bad_config,"unknown scalar curvature accepted");++checks;
  // A CE component must not conceal a vanished positive Dice curvature.
  // These moments describe one near-zero positive probability and fixed,
  // large outside mass; only the ordinary background row is evaluated here.
  config.curvature_policy=TrainingCurvaturePolicy::uniform_v1;config.beta=.5;
  GroupSpec spec{1,1,9007199254740000ULL,1};
  GroupStats stats;stats.retained[0]=stats.retained[1]=1;stats.denominator=double(spec.outside_positive)+2;
  stats.dice=2e-320;RowState row{.5,.5,.25,std::log(2.),1};OutputPair out;
  need(row_output(row,0,spec,stats,config,1,out)==ok,"uniform underflow-control fixture differs");
  config.curvature_policy=TrainingCurvaturePolicy::class_row_sum_v2;
  need(row_output(row,0,spec,stats,config,1,out)==arithmetic_failure,"vanished Dice curvature accepted");++checks;
  return checks;
}
unsigned curvature_device_checks(){
  using namespace class_study;using namespace class_study::group_loss;
  Device<float>margin(std::vector<float>{0,0,-.5f,.5f,1,-2,2,-1});
  Device<float>labels(std::vector<float>{0,1,0,0});
  Device<u32>ids(std::vector<u32>{0,0,1,1});
  Device<GroupSpec>specs(std::vector<GroupSpec>{{4,1,2,1},{2,0,3,1}});
  Device<float>gradients(8),hessians(8);Device<GroupStats>stats(2);Device<u32>status(1);
  const DeviceInputs in{margin.p,labels.p,ids.p,specs.p,4,2};
  const DeviceOutputs out{gradients.p,hessians.p,stats.p,status.p};
  unsigned checks=0;
  for(double beta:std::array<double,3>{0,.5,1}){
    Config config;config.beta=beta;config.loss_scale=4;
    ck(enqueue_objective(in,out,config,nullptr));ck(cudaDeviceSynchronize());
    u32 error=0;ck(cudaMemcpy(&error,status.p,4,cudaMemcpyDeviceToHost));need(error==ok,"uniform device curvature fixture refused");
    const auto old_g=copy(gradients.p,8),old_h=copy(hessians.p,8);
    config.curvature_policy=TrainingCurvaturePolicy::class_row_sum_v2;
    ck(enqueue_objective(in,out,config,nullptr));ck(cudaDeviceSynchronize());
    ck(cudaMemcpy(&error,status.p,4,cudaMemcpyDeviceToHost));need(error==ok,"class row-sum device fixture refused");
    const auto new_g=copy(gradients.p,8),new_h=copy(hessians.p,8);
    need(bits_equal(old_g,new_g),"device curvature changed gradient words");
    for(unsigned row=0;row<4;++row){
      if(beta==0||row>=2)need(std::bit_cast<u32>(old_h[2*row])==std::bit_cast<u32>(new_h[2*row]),"device CE curvature words changed");
      else need(new_h[2*row]>0&&new_h[2*row]<old_h[2*row],"device class row-sum bound did not tighten");
      need(new_h[2*row]==new_h[2*row+1],"device two-logit curvature columns differ");++checks;
    }
  }
  return checks;
}
unsigned host_checks(){
  unsigned refused=0;class_study::TrainingLossSpec loss;loss.loss_scale=4;
  const auto good=class_study::ResidentTrainer::validate_loss(hp(),loss,4,1,2);
  need(good.at("owned_objective_CUDA_bytes")==24*4+sizeof(class_study::group_loss::GroupStats)+36,"workspace accounting differs");
  need(good.at("curvature_policy_version")=="uniform_v1"&&
    good.at("margin_prediction_mode")=="incremental_native_proxy"&&
    good.at("default_margin_prediction_mode")=="incremental_native_proxy"&&
    good.at("moment_reduction_policy")==class_study::group_loss::moment_reduction_policy&&
    good.at("known_native_total_additional_bytes")==32*4,"default custom loss policy differs");
  auto tighter=loss;tighter.curvature_policy=class_study::TrainingCurvaturePolicy::class_row_sum_v2;
  need(class_study::ResidentTrainer::validate_loss(hp(),tighter,4,1,2).at("curvature_policy_version")=="class_row_sum_v2","class row-sum policy receipt differs");
  auto refusal=[&](auto fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}need(failed,"invalid custom loss accepted");++refused;};
  for(const auto&candidate:std::vector<J>{J{{"max_delta_step",0}},J{{"max_delta_step",-1}},J{{"subsample",.5}},J{{"eta",1e-300}}}){auto p=hp();p.update(candidate);refusal([&]{class_study::ResidentTrainer::validate_loss(p,loss,4,1,2);});}
  auto p=hp();p.erase("max_delta_step");refusal([&]{class_study::ResidentTrainer::validate_loss(p,loss,4,1,2);});
  for(auto change:std::array<unsigned,5>{0,1,2,3,4}){auto bad=loss;if(change==0)bad.version=2;if(change==1)bad.beta=-.1;if(change==2)bad.beta=1.1;if(change==3)bad.loss_scale=0;if(change==4)bad.loss_scale=INFINITY;refusal([&]{class_study::ResidentTrainer::validate_loss(hp(),bad,4,1,2);});}
  refusal([&]{class_study::ResidentTrainer::validate_loss(hp(),loss,4,1,3);});
  refusal([&]{class_study::ResidentTrainer::validate_loss(hp(),loss,4,0,2);});
  refusal([&]{class_study::ResidentTrainer::validate_loss(hp(),loss,4,5,2);});
  auto invalid_mode=loss;invalid_mode.margin_mode=static_cast<class_study::TrainingMarginMode>(99);
  refusal([&]{class_study::ResidentTrainer::validate_loss(hp(),invalid_mode,4,1,2);});
  auto invalid_policy=loss;invalid_policy.curvature_policy=static_cast<class_study::TrainingCurvaturePolicy>(99);
  refusal([&]{class_study::ResidentTrainer::validate_loss(hp(),invalid_policy,4,1,2);});
  auto invalid_check=loss;invalid_check.margin_mode=class_study::TrainingMarginMode::full_ensemble;invalid_check.verify_incremental_margins=true;
  refusal([&]{class_study::ResidentTrainer::validate_loss(hp(),invalid_check,4,1,2);});
  auto checked=loss;checked.margin_mode=class_study::TrainingMarginMode::incremental_native_proxy;checked.verify_incremental_margins=true;
  const auto incremental=class_study::ResidentTrainer::validate_loss(hp(),checked,4,1,2);
  need(incremental.at("known_native_total_additional_bytes")==32*4&&incremental.at("owned_objective_CUDA_bytes")==32*4+sizeof(class_study::group_loss::GroupStats)+36,"incremental proxy/verification accounting differs");
  auto default_hp=hp();default_hp.erase("max_delta_step");
  need(!class_study::ResidentTrainer::validate_hyperparameters(default_hp,2).contains("max_delta_step"),"omitted native configuration changed");
  return refused;
}
J gpu_checks(const J&library){
  need(curvature_device_checks()==12,"device curvature coverage differs");
  J plan=library;plan["VALID_read"]=false;plan["TEST_read"]=false;
  // Synthetic audit inputs isolate acceptance/rejection from native training.
  Device<float>zero(std::vector<float>{0,0});Device<class_study::group_loss::StepAudit>audit(1);Device<std::uint32_t>status(1);
  const float audit_eta=.2f,audit_cap=.25f,endpoint=audit_eta*audit_cap;
  unsigned audit_refusals=0;
  for(float after:std::array<float,3>{endpoint,std::nextafter(endpoint,INFINITY),NAN}){
    Device<float>next(std::vector<float>{after,0});ck(cudaMemset(audit.p,0,sizeof(class_study::group_loss::StepAudit)));ck(cudaMemset(status.p,0,4));
    ck(class_study::group_loss::enqueue_step_audit(zero.p,next.p,1,audit_eta,audit_cap,2*double(audit_eta)*audit_cap,audit.p,status.p,nullptr));ck(cudaDeviceSynchronize());
    std::uint32_t error=0;ck(cudaMemcpy(&error,status.p,4,cudaMemcpyDeviceToHost));
    if(after==endpoint)need(error==0,"exact FP32 audit endpoint rejected");else{need(error!=0,"invalid FP32 audit step accepted");++audit_refusals;}
  }

  auto source=std::make_shared<Source>(std::vector<float>{0,0,0,0},std::vector<std::uint32_t>{0,0,0,1},std::vector<float>{1,1,1,1});
  auto weighted=std::make_shared<Source>(std::vector<float>{0,0,0,0},std::vector<std::uint32_t>{0,0,0,1},std::vector<float>{1,2,3,11});
  auto v=view(source),wv=view(weighted);auto g=std::make_shared<Groups>(std::vector<std::uint32_t>{0,0,0,0},std::vector<class_study::FitGroupSpec>{{3,1,0,1}});
  auto groups=group_view(v,g),wg=group_view(wv,g);
  class_study::ResidentTrainer trainer(plan,v),weighted_trainer(plan,wv,trainer);
  bool refused=false;try{trainer.predict(v.values,v.rows,true);}catch(...){refused=true;}need(refused,"public prediction before completed FIT accepted");
  class_study::TrainingLossSpec loss;loss.beta=0;loss.loss_scale=4;
  auto a=trainer.train(hp(),loss,groups);auto margin=copy(trainer.predict(v.values,4,true),8);
  auto b=weighted_trainer.train(hp(),loss,wg);auto weighted_margin=copy(weighted_trainer.predict(wv.values,4,true),8);
  need(bits_equal(margin,weighted_margin),"supplied g/h reweighted by DMatrix weights");
  // At zero margins beta0 gives sum g1=1 and sum h1=2 exp(R).
  // Its uncapped leaf magnitude exceeds .25, so each class must take eta*.25.
  const float cap=float(hp().at("eta").get<double>())*.25f;
  for(unsigned row=0;row<4;++row){need(margin[2*row]==cap&&margin[2*row+1]==-cap,"analytic capped leaf update differs");}
  need(a.metrics.at("trial")==1&&b.metrics.at("trial")==1,"custom FIT counted per round");
  need(a.metrics.at("training_loss").at("initial_margin_words_verified_zero")==true,"initializer audit absent");
  need(a.metrics.at("training_loss").at("step_audit").at("rows_checked")==4,"final update audit absent");
  auto probs=copy(trainer.predict_probabilities(v.values,4),8);
  class_study::ResidentPredictor restored(plan,1,2,trainer);restored.restore_model(a.model_json,a.model_sha256);
  need(bits_equal(margin,copy(restored.predict(v.values,4,true),8)),"saved native margins differ");
  need(bits_equal(probs,copy(restored.predict_probabilities(v.values,4),8)),"saved native probabilities differ");
  auto parent=trainer.train(hp(3),loss,groups);trainer.retain_prefix_source(parent.model_json,parent.model_sha256);
  std::vector<float>previous(8,0);
  for(unsigned rounds=1;rounds<=3;++rounds){
    trainer.slice_retained_prefix(rounds);const auto current=copy(trainer.predict(v.values,4,true),8);
    for(unsigned word=0;word<8;++word){const float lo=previous[word]-cap,hi=previous[word]+cap;need(current[word]>=lo&&current[word]<=hi,"actual native class step exceeds FP32 addition envelope");}
    previous=current;
  }
  auto ranged=copy(trainer.predict_retained_prefix_probabilities(v.values,4,1),8);
  auto prefix=trainer.slice_retained_prefix(1);need(prefix.metrics.at("FIT_performed")==false,"slice counted as FIT");
  need(bits_equal(ranged,copy(trainer.predict_probabilities(v.values,4),8)),"prefix range versus native slice differs");
  auto invalid=groups;invalid.rows=3;refused=false;try{trainer.train(hp(),loss,invalid);}catch(...){refused=true;}need(refused,"wrong group row shape accepted");
  invalid=groups;invalid.fit_owner=weighted;refused=false;try{trainer.train(hp(),loss,invalid);}catch(...){refused=true;}need(refused,"different FIT owner accepted");
  invalid=groups;invalid.fit_binding["FIT_only"]=false;refused=false;try{trainer.train(hp(),loss,invalid);}catch(...){refused=true;}need(refused,"different FIT binding accepted");
  auto bad_groups=std::make_shared<Groups>(std::vector<std::uint32_t>{0,0,0,1},std::vector<class_study::FitGroupSpec>{{3,1,0,1}});
  auto bad=group_view(v,bad_groups);refused=false;try{trainer.train(hp(),loss,bad);}catch(...){refused=true;}need(refused,"out-of-range group ID accepted");
  refused=false;try{trainer.predict(v.values,4,true);}catch(...){refused=true;}need(refused,"failed partial FIT became public model");
  auto recovered=trainer.train(hp(),loss,groups);need(recovered.metrics.at("trial")==3,"failed FIT incremented completed counter");
  for(double beta:std::array<double,2>{.5,1.0}){
    auto mixed=loss;mixed.beta=beta;const auto model=weighted_trainer.train(hp(2),mixed,wg);
    need(model.metrics.at("training_loss").at("step_audit").at("updates_checked")==2,"mixed-class loss failed final step audit");
  }
  // Same-booster differential checks: genuine split candidates, two groups,
  // three features, missing values, all beta controls and every round. This
  // does not assume separately trained models are universally bit-identical.
  std::vector<float>ix(36),iw(12);std::vector<std::uint32_t>iy(12),ig(12);
  for(unsigned row=0;row<12;++row){
    ix[3*row]=float(row%4);ix[3*row+1]=row==8?NAN:float(row/4);ix[3*row+2]=float((row*7)%11);
    iy[row]=row%3==0;ig[row]=row%2;iw[row]=iy[row]?.6f:1.2f;
  }
  auto incremental_source=std::make_shared<Source>(ix,iy,iw);auto iv=view(incremental_source,3);
  iv.binding["ordered_row_ids"]=J::array();for(unsigned row=0;row<12;++row)iv.binding["ordered_row_ids"].push_back(7+17*row);
  auto incremental_groups=std::make_shared<Groups>(ig,std::vector<class_study::FitGroupSpec>{{8,2,1,1},{8,2,3,1}});
  auto gv=group_view(iv,incremental_groups);class_study::ResidentTrainer incremental_trainer(plan,iv,trainer);
  auto ihp=hp(8);ihp["gamma"]=0;ihp["lambda"]=1;ihp["max_depth"]=2;
  std::string beta0_model;std::vector<float> beta0_margins;
  for(auto policy:std::array<class_study::TrainingCurvaturePolicy,2>{class_study::TrainingCurvaturePolicy::uniform_v1,class_study::TrainingCurvaturePolicy::class_row_sum_v2})
  for(double beta:std::array<double,3>{0,.5,1}){
    auto checked=loss;checked.beta=beta;checked.loss_scale=12;checked.curvature_policy=policy;
    checked.margin_mode=class_study::TrainingMarginMode::incremental_native_proxy;checked.verify_incremental_margins=true;
    const auto model=incremental_trainer.train(ihp,checked,gv);const auto&receipt=model.metrics.at("training_loss");
    need(receipt.at("incremental_margin_words_compared")==192&&receipt.at("native_incremental_margin_predictions")==8&&receipt.at("native_full_margin_predictions")==9,"incremental differential coverage differs");
    need(receipt.at("step_audit").at("updates_checked")==8&&receipt.at("training_DMatrix_base_margin_modified")==false,"incremental final audit/FIT isolation differs");
    need(receipt.at("curvature_policy_version")==
      (policy==class_study::TrainingCurvaturePolicy::uniform_v1?"uniform_v1":"class_row_sum_v2"),"trained curvature receipt differs");
    if(beta==0){
      const auto current=copy(incremental_trainer.predict(iv.values,iv.rows,true),iv.rows*2);
      if(policy==class_study::TrainingCurvaturePolicy::uniform_v1){beta0_model=model.model_sha256;beta0_margins=current;}
      else need(beta0_model==model.model_sha256&&bits_equal(beta0_margins,current),"curvature policy changed beta0 model/margin words");
    }
  }
  auto long_checked=loss;long_checked.beta=.5;long_checked.loss_scale=12;
  long_checked.margin_mode=class_study::TrainingMarginMode::incremental_native_proxy;long_checked.verify_incremental_margins=true;
  auto long_hp=ihp;long_hp["rounds"]=128;
  const auto long_model=incremental_trainer.train(long_hp,long_checked,gv);
  need(long_model.metrics.at("training_loss").at("incremental_margin_words_compared")==12*2*128&&
    long_model.metrics.at("training_loss").at("step_audit").at("updates_checked")==128,
    "128-round incremental differential or final audit coverage differs");
  auto fast=loss;fast.loss_scale=12;fast.margin_mode=class_study::TrainingMarginMode::incremental_native_proxy;
  const auto fast_model=incremental_trainer.train(ihp,fast,gv);
  need(fast_model.metrics.at("training_loss").at("native_full_margin_predictions")==1&&fast_model.metrics.at("training_loss").at("incremental_margin_words_compared")==0,"unchecked incremental path replays full ensemble");
  // No-inside-positive and all-positive FITs exercise actual zero-round native
  // initialization and supported single-class gradient paths without a floor.
  for(unsigned label:std::array<unsigned,2>{0,1}){
    auto one=std::make_shared<Source>(std::vector<float>{0,0,0,0},std::vector<std::uint32_t>(4,label),std::vector<float>{1,2,3,4});auto ov=view(one);
    auto og=std::make_shared<Groups>(std::vector<std::uint32_t>{0,0,0,0},std::vector<class_study::FitGroupSpec>{{label?0ULL:4ULL,label?4ULL:0ULL,2,1}});
    class_study::ResidentTrainer t(plan,ov,trainer);auto group=group_view(ov,og);auto pure=loss;pure.beta=1;pure.curvature_policy=class_study::TrainingCurvaturePolicy::class_row_sum_v2;pure.margin_mode=class_study::TrainingMarginMode::incremental_native_proxy;pure.verify_incremental_margins=true;const auto model=t.train(hp(),pure,group);
    need(model.metrics.at("training_loss").at("initial_margin_words_verified_zero")==true,"single-class initializer differs");
  }
  // Existing path remains callable and normalizes an omitted cap identically.
  auto native_hp=hp();native_hp.erase("max_delta_step");auto native=trainer.train(native_hp);
  need(!native.metrics.contains("training_loss")&&native.metrics.at("trial")==4,"default train path/counter changed");
  return {{"complete",true},{"CUDA_executed",true},{"step_audit_refusals",audit_refusals},{"incremental_long_horizon_rounds_checked",128},{"incremental_same_booster_every_round_words_equal",true},{"incremental_FIT_DMatrix_unchanged",true},{"weight_once_leaf_fixture",true},{"explicit_zero_initializer",true},{"single_class_initialization",true},{"borrowed_output_shapes",true},{"restore_and_prefix_bits_equal",true},{"failure_does_not_complete_FIT",true},{"heldout_data_read",false}};
}
struct FitBenchmarkOptions {
  std::uint64_t rows=1048576;
  std::uint32_t features=54,groups=26,rounds=128;
  bool incremental_first=false,verify=false;
};
J fit_benchmark(const J&library,const FitBenchmarkOptions&o){
  need(o.rows>=1024&&o.rows<=2097152&&o.features>=3&&o.features<=128&&o.groups>=2&&o.groups<=64&&o.rows>=2*o.groups&&o.rounds>0&&o.rounds<=512,"FIT benchmark extent outside supported fixture bounds");
  using Clock=std::chrono::steady_clock;const auto all_started=Clock::now();
  auto seconds=[](auto start){return std::chrono::duration<double>(Clock::now()-start).count();};
  const auto setup_started=Clock::now();
  std::vector<float>x(o.rows*o.features),weights(o.rows);
  std::vector<std::uint32_t>labels(o.rows),ids(o.rows);
  std::vector<class_study::FitGroupSpec>specs(o.groups);
  std::uint64_t row=0;
  for(std::uint32_t g=0;g<o.groups;++g){
    const std::uint64_t count=o.rows/o.groups+(g<o.rows%o.groups);
    for(std::uint64_t k=0;k<count;++k,++row){
      ids[row]=g;labels[row]=g!=0&&k%97==0;
      if(labels[row])++specs[g].full_positive;else++specs[g].full_negative;
      for(std::uint32_t f=0;f<o.features;++f){
        // Deterministic feature words, generated from coordinates rather than
        // reading the target. The first two supply useful split structure.
        float value=f==0?float(k%97)/96.0f:f==1?float(g)/float(o.groups-1):
          float((k*(2*f+1)+g*17+f*13)%1021)/1020.0f;
        if(f==o.features-1&&k%4093==0)value=NAN;
        x[row*o.features+f]=value;
      }
    }
    specs[g].full_negative*=32;specs[g].outside_positive=g?g%5:11;
  }
  need(row==o.rows,"FIT benchmark group extent differs");
  for(std::uint64_t r=0;r<o.rows;++r){
    const auto g=ids[r];const double full=double(specs[g].full_negative+specs[g].full_positive);
    const double inverse=labels[r]?1.0:32.0;
    weights[r]=float((double(o.rows)/o.groups)*inverse/full);
  }
  auto source=std::make_shared<Source>(x,labels,weights);auto group_source=std::make_shared<Groups>(ids,specs);
  // Release large host staging before FIT; both timed modes retain this exact
  // immutable CUDA input and the same DMatrix/cut cache throughout.
  std::vector<float>().swap(x);std::vector<float>().swap(weights);
  std::vector<std::uint32_t>().swap(labels);std::vector<std::uint32_t>().swap(ids);
  class_study::ResidentDataView v;v.values=source->x.p;v.labels=source->y.p;v.sample_weights=source->weights.p;
  v.rows=o.rows;v.row_stride=o.features;v.features=o.features;v.classes=2;v.fit_rows=o.rows;v.owner=source;
  v.binding={{"format","synthetic-shared-custom-FIT-benchmark-1"},{"FIT_only",true},{"rows",o.rows},{"features",o.features},{"groups",o.groups},{"group_layout","contiguous"},{"generation","coordinate-feature-v1; shared immutable CUDA words for every mode"}};
  auto groups=group_view(v,group_source);J plan=library;plan["VALID_read"]=false;plan["TEST_read"]=false;
  class_study::ResidentTrainer trainer(plan,v);const auto setup_seconds=seconds(setup_started);
  J parameters={{"rounds",o.rounds},{"max_depth",3},{"eta",.1},{"max_delta_step",.25},{"lambda",1},{"alpha",0},{"gamma",0},{"min_child_weight",1},{"max_bin",128},{"seed",19},{"subsample",1}};
  class_study::TrainingLossSpec loss;loss.beta=.5;loss.loss_scale=double(o.rows);
  auto warm_parameters=parameters;warm_parameters["rounds"]=1;
  J warmups=J::array();
  for(auto mode:{class_study::TrainingMarginMode::full_ensemble,class_study::TrainingMarginMode::incremental_native_proxy}){
    loss.margin_mode=mode;const auto started=Clock::now();auto warm=trainer.train(warm_parameters,loss,groups);
    warmups.push_back({{"mode",mode==class_study::TrainingMarginMode::full_ensemble?"full_ensemble":"incremental_native_proxy"},{"rounds",1},{"FIT_seconds",seconds(started)}});
  }
  J timed=J::array();
  const auto full=class_study::TrainingMarginMode::full_ensemble,incremental=class_study::TrainingMarginMode::incremental_native_proxy;
  const std::array<class_study::TrainingMarginMode,2>order=o.incremental_first?std::array{incremental,full}:std::array{full,incremental};
  for(auto mode:order){
    loss.margin_mode=mode;loss.verify_incremental_margins=false;
    const auto started=Clock::now();auto model=trainer.train(parameters,loss,groups);const double wall=seconds(started);
    const auto&provenance=model.metrics.at("training_loss");
    timed.push_back({{"mode",mode==full?"full_ensemble":"incremental_native_proxy"},{"rounds",o.rounds},{"FIT_seconds",wall},{"amortized_seconds_per_round",wall/o.rounds},{"trainer_reported_FIT_seconds",model.metrics.at("seconds_including_buffer_export")},{"source_model_sha256",model.model_sha256},{"native_full_margin_predictions",provenance.at("native_full_margin_predictions")},{"native_incremental_margin_predictions",provenance.at("native_incremental_margin_predictions")},{"step_audit",provenance.at("step_audit")},{"verification_enabled",false}});
  }
  J verification=nullptr;
  if(o.verify){
    loss.margin_mode=incremental;loss.verify_incremental_margins=true;
    const auto started=Clock::now();auto model=trainer.train(parameters,loss,groups);
    verification={{"FIT_seconds_excluded_from_timed_pair",seconds(started)},{"rounds",o.rounds},{"margin_words_compared",model.metrics.at("training_loss").at("incremental_margin_words_compared")},{"passed",true},{"same_booster_each_round",true}};
  }
  return {{"complete",true},{"CUDA_executed",true},{"synthetic_data",true},{"real_cohort_rows_read",false},{"mode","shared-custom-FIT-benchmark"},{"rows",o.rows},{"features",o.features},{"groups",o.groups},{"group_layout","contiguous"},{"rounds",o.rounds},{"hyperparameters",parameters},{"beta",.5},{"loss_scale",o.rows},{"data_cut_weights","equal group inverse-retention; mean one subject to FP32 rounding"},{"same_immutable_weighted_DMatrix",true},{"cut_cache_prewarmed",true},{"fresh_native_booster_per_FIT",true},{"objective_and_proxy_paths_prewarmed",true},{"setup_seconds",setup_seconds},{"warmup_FITs",warmups},{"timed_order",o.incremental_first?"incremental-first":"full-first"},{"timed_FITs",timed},{"optional_untimed_verification",verification},{"total_wall_seconds",seconds(all_started)},{"timing_scope","Each timed train call includes objective allocation/reductions, native updates, margin predictions, step audits and export/hash; excludes dataset staging, DMatrix creation, warmups and optional equality verification."},{"separate_FIT_bitwise_identity_claim",false},{"per_round_files_written",0}};
}

}
int main(int argc,char**argv){try{
  const auto checks=host_checks(),curvature_checks=curvature_scalar_checks();
  if(argc==1){std::cout<<J{{"complete",true},{"CUDA_executed",false},{"host_refusals",checks},{"curvature_scalar_checks",curvature_checks}}.dump()<<'\n';return 0;}
  need(argc>=3,"usage: loss-checks --gpu-check plan.json | --gpu-fit-benchmark plan.json [--rows N --features F --groups G --rounds R --order full-first|incremental-first --verify]");
  std::ifstream input(argv[2]);need(bool(input),"library plan missing");J plan;input>>plan;
  J result;
  if(std::string(argv[1])=="--gpu-check"){need(argc==3,"unexpected GPU check arguments");result=gpu_checks(plan);result["curvature_device_checks"]=12;result["curvature_policies_verified"]={"uniform_v1","class_row_sum_v2"};result["beta0_curvature_model_and_margins_equal"]=true;}
  else{
    need(std::string(argv[1])=="--gpu-fit-benchmark","unknown check mode");FitBenchmarkOptions options;
    for(int i=3;i<argc;++i){
      const std::string key=argv[i];if(key=="--verify"){options.verify=true;continue;}
      need(i+1<argc,"benchmark option missing value");const std::string value=argv[++i];
      if(key=="--rows")options.rows=std::stoull(value);
      else if(key=="--features"||key=="--groups"||key=="--rounds"){
        const auto n=std::stoull(value);need(n<=UINT32_MAX,"benchmark integer capacity");
        if(key=="--features")options.features=std::uint32_t(n);else if(key=="--groups")options.groups=std::uint32_t(n);else options.rounds=std::uint32_t(n);
      }else if(key=="--order"){need(value=="full-first"||value=="incremental-first","benchmark order differs");options.incremental_first=value=="incremental-first";}
      else throw std::runtime_error("unknown benchmark option");
    }
    result=fit_benchmark(plan,options);
  }
  result["host_refusals"]=checks;result["curvature_scalar_checks"]=curvature_checks;std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
