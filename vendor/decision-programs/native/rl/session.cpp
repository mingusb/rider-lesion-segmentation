#include "streaming_file_hash.hpp"
#include "session.hpp"
#include "class_identity.hpp"
#include "class_rank_gpu_class_apply_source_receipt.hpp"
#include "class_rank_gpu_binary_grid_validator_receipt.hpp"
#include "class_rank_gpu_score_pool_identity.hpp"
#include <unistd.h>
#include <mutex>
#include <chrono>
#include <set>
#include <map>

namespace rl_qualified_session {
namespace {
using J=dpnative::json;namespace fs=std::filesystem;
namespace ca=rank_gpu_class_apply;namespace lp=rank_gpu_leaf_partitions;
namespace source=rank_gpu_score_source_apply;namespace codec=rank_regional_model_export;
namespace grid=rank_gpu_binary_grid_validator;
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
void same(const codec::Model&a,const codec::Model&b){
  need(a.source_sha256==b.source_sha256&&a.rank_sha256==b.rank_sha256&&
       a.rank_cut_bits==b.rank_cut_bits&&a.scope.lo==b.scope.lo&&a.scope.hi==b.scope.hi&&
       a.scope.allowed==b.scope.allowed&&a.nodes==b.nodes&&a.terms==b.terms&&a.root==b.root,
       "session codec/readback correspondence differs");
}
std::string hash_words(const std::vector<U>&words){rank_gpu_score_identity::Hash h;
  h.text("CUDA-full-class-graph-words-v1");for(auto w:words)h.word(w,8);return h.finish();}
std::string graph_metadata(const ca::Result&r,const lp::Result&p){
  return J{{"class",apply_receipt::describe(r)},{"source_sha256",p.source_sha256},
           {"library_sha256",p.library_sha256},{"rank_sha256",p.rank_sha256},
           {"source_partition_binding",p.source.binding},{"domain",leaf_receipt::box(p.domain)}}.dump();
}
J trace_payload(const rl_category_lowering::Result&r){J traces=J::array();
  for(const auto&t:r.traces)if(t.dimension==10||t.dimension==11){J tests=J::array();
    for(unsigned k=0;k<t.count;++k)tests.push_back(t.test_order[k]);
    traces.push_back(J{{"source_node",t.source_node},{"dimension",t.dimension},
      {"selected_mask",t.selected_mask},{"fallback_arc",t.fallback_arc},
      {"policy_version",t.policy_version},{"test_order",tests}});
  }
  return J{{"injected",r.order_injected},{"trace_audited",r.trace_audited},
           {"policy_version",r.policy_version},{"order44",r.selected_order},{"traces",traces}};
}
}

struct QualifiedProposal::Data {
  std::shared_ptr<const void> owner;
  U bytes=0,nodes=0,policy_version=0;fs::path directory;
  fs::path runtime_file,certificate_file,trace_file;
  std::string runtime_sha,certificate_sha,session_binding;
  std::string trace_sha;
  std::string certificate_bytes,trace_bytes;
  ProposalTimings timings;
  bool order_injected=false;std::array<unsigned,44> order{};
  bool sample_bound=false;U sampled_seed=0,sampled_episode=0;
  bool cache_hit=false;U cache_id=UINT64_MAX;std::string effective_key_sha;
};
QualifiedProposal::QualifiedProposal(std::shared_ptr<const Data>d):p_(std::move(d)){}
U QualifiedProposal::encoded_bytes()const{return p_->bytes;}
U QualifiedProposal::nodes()const{return p_->nodes;}
U QualifiedProposal::policy_version()const{return p_->policy_version;}
const std::string&QualifiedProposal::runtime_sha256()const{return p_->runtime_sha;}
const std::string&QualifiedProposal::certificate_sha256()const{return p_->certificate_sha;}
const fs::path&QualifiedProposal::directory()const{return p_->directory;}
const fs::path&QualifiedProposal::runtime_path()const{return p_->runtime_file;}
bool QualifiedProposal::cache_hit()const{return p_->cache_hit;}
const std::string&QualifiedProposal::certificate_record()const{return p_->certificate_bytes;}
ProposalTimings QualifiedProposal::timings()const{return p_->timings;}

struct Session::Impl {
  struct Graph {const ca::Result classes;const lp::Result partitions;
    Graph(ca::Result r,lp::Result p):classes(std::move(r)),partitions(std::move(p)){} };
  std::unique_ptr<const Graph> graph;
  std::unique_ptr<detail::ClassIdentity> identity;
  std::unique_ptr<detail::EffectiveCache> cache;
  std::vector<std::shared_ptr<const QualifiedProposal::Data>>cache_proofs;
  std::set<std::string>issued_outputs;
  std::shared_ptr<const void> owner=std::make_shared<const unsigned char>(0);
  std::unique_ptr<grid::Validator> native_validator;
  Options options;int device;U pid;fs::path teacher,library,base,frozen;
  std::string teacher_bytes,library_sha,metadata_identity;
  J build,base_certificate;Summary status;mutable std::mutex mutex;
  std::vector<std::shared_ptr<const QualifiedProposal::Data>> published;
  bool resident_sealed=false;
  bool resident_created=false;

  Impl(const std::string&lib,const std::string&model,const std::string&expected,
       fs::path fresh,fs::path sources,Options o,const Stop&stop,int gpu)
      :options(o),device(gpu),pid(getpid()),teacher(fs::absolute(model)),library(fs::absolute(lib)),
       base(fs::absolute(fresh)),frozen(fs::absolute(sources)) {
    const auto setup_start=Clock::now();
    need(options.maximum_device_bytes&&options.maximum_validation_cells&&options.native_batch_rows&&options.maximum_runtime_bytes,"session zero capacity");
    need(!fs::exists(base)&&fs::create_directories(base/"audit")&&fs::create_directories(base/"runtime"),"session needs fresh base directory");
    J pending{{"format","opaque-source-qualified-class-session-1"},{"qualified",false},
              {"process_id",pid},{"whole_source_conversion_complete",false}};
    try {
      build=source_apply_receipt::build(frozen);
      const auto validator_sha=public_rl_build::source_sha256("class_rank_gpu_binary_grid_validator.cu");
      teacher_bytes=dpnative::read_text(teacher);need(dpnative::sha256(teacher_bytes)==expected,"session teacher SHA differs");
      library_sha=dp_streaming::sha256_file(library);
      need(library_sha=="462aa6331ecb178df8d10f16612c865ae70f571c248138e50c8a9eb4bc007dd4","session native library differs");
      need(!stop||!stop(),"cancelled_before_session_construction");
      dpnative::atomic_text(base/"audit/source.json",teacher_bytes);
      lp::Result partitions;ca::Result classes;J construction;
      {
        lp::Bridge bridge(library.string(),(base/"audit/source.json").string(),expected,{},device);
        partitions=bridge.extract(stop);need(partitions.complete,"session source bridge incomplete");
        need(partitions.owned_device_resident_bytes<options.maximum_device_bytes,"session bridge device budget");
        U work_cap=options.maximum_device_bytes-partitions.owned_device_resident_bytes;
        source::Options so;so.maximum_device_bytes=work_cap;
        auto folded=source::fold(partitions,so,stop,device);
        need(folded.complete&&folded.accepted&&folded.accepted->matches(partitions),"session ordered score fold incomplete");
        ca::Options co;co.maximum_device_bytes=work_cap;
        classes=ca::construct(*folded.accepted,partitions,library.string(),(base/"audit/source.json").string(),co,stop,device);
        need(classes.complete&&classes.local_semantics_audited,"session class graph incomplete");
        construction={{"bridge",leaf_receipt::describe(partitions)},
                      {"score_fold",source_apply_receipt::describe(folded)},
                      {"class_graph",apply_receipt::describe(classes)}};
      } // Release bridge GPU storage before persistent identity/validation.
      graph=std::make_unique<const Graph>(std::move(classes),std::move(partitions));
      identity=std::make_unique<detail::ClassIdentity>(graph->classes,graph->partitions.rank_cut_bits,options.maximum_device_bytes,device);
      metadata_identity=graph_metadata(graph->classes,graph->partitions);
      status.class_word_sha256=hash_words(identity->words());
      status.identity_retained_device_bytes=identity->retained_device_bytes();
      need(status.identity_retained_device_bytes<options.maximum_device_bytes,"session retained identity budget");
      U stage_cap=options.maximum_device_bytes-status.identity_retained_device_bytes;
      auto lower=rl_category_lowering::lower_binary(graph->classes,{},1048576,stage_cap,stop,device);
      need(lower.lowered.complete,"session default lowering incomplete");
      auto canonical=apply_receipt::canonical(graph->classes,graph->partitions,std::move(lower.lowered),base.string(),stage_cap);
      auto encoded=codec::encode(canonical.model);need(encoded.size()<=options.maximum_runtime_bytes,"session runtime byte cap");
      auto decoded=codec::decode(encoded);same(canonical.model,decoded);
      dpnative::atomic_text(base/"runtime/model.bin",encoded);auto runtime_sha=dpnative::sha256(encoded);
      auto disk=codec::read_model(base/"runtime/model.bin",runtime_sha,options.maximum_runtime_bytes);same(decoded,disk);
      grid::Options go;go.maximum_cells=options.maximum_validation_cells;go.native_batch_rows=options.native_batch_rows;
      go.maximum_device_bytes=stage_cap;
      native_validator=std::make_unique<grid::Validator>(library.string(),(base/"audit/source.json").string(),go);
      auto profile=native_validator->profile_only();
      need(profile.fits_uint64&&profile.within_cell_budget,"session baseline native grid capacity exceeded");
      auto audit=native_validator->audit(disk,stop);
      need(audit.complete&&audit.CUDA_executed&&audit.candidate_source_cell_constancy,"session independent default native grid incomplete");
      auto audit_json=binary_grid_receipt::describe(audit);
      dpnative::atomic_json(base/"audit/native-grid.json",audit_json);
      status.source_sha256=expected;status.library_sha256=library_sha;status.rank_sha256=graph->partitions.rank_sha256;
      status.baseline_runtime_sha256=runtime_sha;status.baseline_bytes=encoded.size();status.baseline_nodes=disk.nodes.size();
      status.base_native_cells=audit.cells;status.base_native_audits=1;status.class_builds=1;status.process_id=pid;
      status.base_native_audit_sha256=dp_streaming::sha256_file(base/"audit/native-grid.json");
      status.build_binding_sha256=dpnative::sha256(build.dump());
      need(!stop||!stop(),"cancelled_before_session_qualification");
      identity->audit(graph->classes,graph->partitions.rank_cut_bits);
      need(source_apply_receipt::build(frozen)==build&&dpnative::read_text(teacher)==teacher_bytes&&
           dpnative::read_text(base/"audit/source.json")==teacher_bytes&&
           dp_streaming::sha256_file(library)==library_sha,"session immutable source/build changed");
      same(decoded,codec::read_model(base/"runtime/model.bin",runtime_sha,options.maximum_runtime_bytes));
      base_certificate={{"format","opaque-source-qualified-class-session-base-1"},{"process_id",pid},
        {"source_sha256",expected},{"library_sha256",library_sha},{"rank_sha256",status.rank_sha256},
        {"class_word_sha256",status.class_word_sha256},{"class_metadata_sha256",dpnative::sha256(metadata_identity)},
        {"baseline_runtime_sha256",runtime_sha},{"baseline_bytes",status.baseline_bytes},
        {"baseline_nodes",status.baseline_nodes},{"native_cells",audit.cells},
        {"base_native_audit_sha256",status.base_native_audit_sha256},{"frozen_build",build},
        {"validator_source_sha256",validator_sha},{"collector_input_sha256",canonical.collector_input_sha256},
        {"collector_output_sha256",canonical.collector_output_sha256},{"construction",construction},
        {"domain",leaf_receipt::box(graph->partitions.domain)},
        {"native_configuration",J::parse(audit.native_configuration)},
        {"native_contract",J::parse(audit.native_contract)},
        {"domain_scope","finite FP32 numeric10 including signed zeros; exactly-one wilderness4/soil40"},
        {"native_implementation_and_CUDA_auditors_trusted",true},{"formal_CUDA_refinement",false},
        {"archive_import_authority",false},{"future_process_reuse_authority",false}};
      status.binding=dpnative::sha256(base_certificate.dump());base_certificate["binding_sha256"]=status.binding;
      base_certificate["qualified"]=true;base_certificate["whole_source_conversion_complete"]=true;
      dpnative::atomic_json(base/"session-base.json",base_certificate);
      status.qualified=true;status.CUDA_executed=true;
      // The complete baseline now supplies the immutable native-source premise.
      // Free native validator's GPU buffers; proposals perform no native calls.
      native_validator.reset();
      if(options.maximum_cache_entries){
        need(identity->owned_device_peak_bytes()<options.maximum_device_bytes,"cache leaves no identity audit budget");
        cache=std::make_unique<detail::EffectiveCache>(*identity,options.maximum_cache_entries,
                options.maximum_device_bytes-identity->owned_device_peak_bytes(),device);
        auto key=cache->normalize(nullptr);auto entry=std::make_shared<QualifiedProposal::Data>();
        entry->owner=owner;entry->session_binding=status.binding;entry->bytes=status.baseline_bytes;entry->nodes=status.baseline_nodes;
        entry->runtime_file=base/"runtime/model.bin";entry->runtime_sha=status.baseline_runtime_sha256;
        entry->certificate_file=base/"session-base.json";entry->certificate_sha=dp_streaming::sha256_file(entry->certificate_file);
        entry->certificate_bytes=dpnative::read_text(entry->certificate_file);
        entry->directory=base;entry->effective_key_sha=key.key_sha256;entry->cache_id=cache->admit();
        need(entry->cache_id==0,"baseline cache admission incomplete");cache_proofs.push_back(entry);
        status.cache_retained_device_bytes=cache->retained_device_bytes();
      }
      status.setup_seconds=elapsed(setup_start);status.publication_identity_current=true;
    }catch(const std::exception&e){pending["error"]=e.what();dpnative::atomic_json(base/"incomplete-session.json",pending);throw;}
  }

  void audit_graph()const {
    need(status.qualified&&U(getpid())==pid,"session not qualified in current process");
    identity->audit(graph->classes,graph->partitions.rank_cut_bits);
  }
  void sealed(const QualifiedProposal::Data&v)const{
    need(v.owner==owner&&v.session_binding==status.binding&&
         dpnative::sha256(v.certificate_bytes)==v.certificate_sha&&
         (v.trace_sha.empty()||dpnative::sha256(v.trace_bytes)==v.trace_sha),
         "private sealed qualification changed");
  }
  void external_artifact(const QualifiedProposal::Data&v)const{
    sealed(v);
    need(dp_streaming::sha256_file(v.runtime_file)==v.runtime_sha&&
         (v.certificate_file.empty()||dp_streaming::sha256_file(v.certificate_file)==v.certificate_sha)&&
         (v.trace_file.empty()||dp_streaming::sha256_file(v.trace_file)==v.trace_sha),
         "private qualified artifact changed");
  }
  void artifact(const QualifiedProposal::Data&v)const{
    // Immutable private records authorize steady-state cost reuse. External
    // changes reject final publication; opt-in diagnostics reject immediately.
    if(options.per_trial_diagnostics)external_artifact(v);else sealed(v);
  }
  void audit_external()const {
    audit_graph();
    need(source_apply_receipt::build(frozen)==build&&dpnative::read_text(teacher)==teacher_bytes&&
         dpnative::read_text(base/"audit/source.json")==teacher_bytes&&
         dp_streaming::sha256_file(library)==library_sha&&
         graph_metadata(graph->classes,graph->partitions)==metadata_identity,
         "qualified session source/build/metadata changed");
    need(dp_streaming::sha256_file(base/"runtime/model.bin")==status.baseline_runtime_sha256&&
         dp_streaming::sha256_file(base/"audit/native-grid.json")==status.base_native_audit_sha256&&
         J::parse(dpnative::read_text(base/"session-base.json"))==base_certificate,
         "qualified baseline artifacts changed");
    std::map<fs::path,std::string> retained;
    auto unique_artifact=[&](const QualifiedProposal::Data&v){
      sealed(v);
      for(const auto&[path,hash]:std::array<std::pair<fs::path,std::string>,3>{{
          {v.runtime_file,v.runtime_sha},{v.certificate_file,v.certificate_sha},{v.trace_file,v.trace_sha}}}){
        if(path.empty())continue;
        auto [it,inserted]=retained.emplace(path,hash);
        need(it->second==hash,"retained artifact identity conflict");
        if(inserted)need(dp_streaming::sha256_file(path)==hash,"retained artifact changed before publication");
      }
    };
    for(const auto&v:published)unique_artifact(*v);
    for(const auto&v:cache_proofs)unique_artifact(*v);
  }
};

Session::Session(const std::string&lib,const std::string&teacher,const std::string&sha,
                 fs::path base,fs::path frozen,Options o,const Stop&stop,int device)
    :p_(std::make_unique<Impl>(lib,teacher,sha,std::move(base),std::move(frozen),o,stop,device)){}
Session::~Session()=default;
Summary Session::summary()const{std::lock_guard lock(p_->mutex);return p_->status;}
void Session::finalize(){std::lock_guard lock(p_->mutex);auto before=Clock::now();
  p_->status.publication_identity_current=false;p_->audit_external();
  p_->status.finalization_seconds=elapsed(before);p_->status.publication_identity_current=true;
}
QualifiedCostTable Session::seal_resident_table(){
  std::lock_guard lock(p_->mutex);p_->status.publication_identity_current=false;
  need(!p_->resident_sealed&&p_->cache&&p_->cache->entries()==p_->cache_proofs.size(),"resident table requires complete private cache admissions");
  p_->audit_external();auto keys=p_->cache->snapshot_keys();std::vector<ResidentArtifact>artifacts;
  for(U e=0;e<p_->cache_proofs.size();++e){const auto&v=*p_->cache_proofs[e];p_->sealed(v);
    need(v.cache_id==e,"resident cache/proof order differs");artifacts.push_back({e,v.bytes,v.nodes,v.runtime_sha,v.certificate_sha,v.runtime_file});}
  U occupied=p_->identity->retained_device_bytes()+p_->cache->retained_device_bytes();need(occupied<p_->options.maximum_device_bytes,"resident seal leaves no device capacity");
  auto table=QualifiedCostTable::seal(p_->owner,p_->pid,p_->device,1,p_->status.baseline_bytes,p_->status.binding,
      dpnative::sha256(p_->build.dump()+"|resident-policy1-full44-exact-effective-key-1"),p_->identity->words(),keys,std::move(artifacts),p_->options.maximum_device_bytes-occupied);
  p_->resident_sealed=true;return table;
}
std::unique_ptr<ResidentTraining> Session::make_resident(const QualifiedCostTable&table,
    const rl_category_policy::Policy&initial,U rate,ResidentSchedule plan,ResidentOptions options){
  std::lock_guard lock(p_->mutex);need(p_->resident_sealed&&!p_->resident_created,"sealed Session permits exactly one resident engine");
  table.require_owner(p_->owner,p_->pid,p_->status.binding);p_->audit_graph();
  auto engine=std::unique_ptr<ResidentTraining>(new ResidentTraining(table,initial.logit_words(),initial.version(),rate,plan,options));
  p_->resident_created=true;return engine;
}
void Session::finalize_resident(const QualifiedCostTable&table,const ResidentTraining&training){
  std::lock_guard lock(p_->mutex);p_->status.publication_identity_current=false;
  table.require_owner(p_->owner,p_->pid,p_->status.binding);training.require_complete(table);
  const auto before=Clock::now();p_->audit_external();p_->status.finalization_seconds=elapsed(before);p_->status.publication_identity_current=true;
}

QualifiedProposal Session::propose(rl_category_lowering::DeviceOrder action,fs::path output,const Stop&stop){
  return propose_impl(action,nullptr,std::move(output),stop);
}
QualifiedProposal Session::propose_impl(rl_category_lowering::DeviceOrder action,const rl_category_policy::OrderView*sample,fs::path output,const Stop&stop){
  std::lock_guard lock(p_->mutex);output=fs::absolute(output);
  need(!p_->resident_sealed,"Session cache admissions are frozen by resident sealing");
  if(sample)need(sample->order44&&sample->traces&&sample->version&&sample->device==p_->device&&
                 sample->baseline_bytes==p_->status.baseline_bytes,"sample has wrong session baseline/device");
  need(!fs::exists(output)&&p_->issued_outputs.insert(output.string()).second,"proposal needs fresh logical output");
  J pending{{"format","source-qualified-session-proposal-1"},{"complete",false},{"source_equivalent",false},{"session_binding",p_->status.binding}};
  ProposalTimings timing;const auto whole_start=Clock::now();
  p_->status.publication_identity_current=false;
  try {
    auto before=Clock::now();p_->audit_graph();timing.identity_before=elapsed(before);
    need(!stop||!stop(),"cancelled_before_session_proposal");
    detail::EffectiveAction effective;
    if(p_->cache){effective=p_->cache->normalize(action.bits);
      if(effective.hit!=UINT64_MAX){
        need(effective.hit<p_->cache_proofs.size(),"cache proof index");const auto&entry=*p_->cache_proofs[effective.hit];
        need(entry.effective_key_sha==effective.key_sha256,"cache exact-key/hash provenance differs");p_->artifact(entry);
        need(!stop||!stop(),"cancelled_before_cache_certificate");p_->cache->audit_order(action.bits);p_->audit_graph();p_->artifact(entry);
        J traces{{"format","effective-order-qualified-cache-hit-1"},{"injected",action.bits!=nullptr},{"policy_version",action.policy_version},{"order44",effective.order},{"effective_key_sha256",effective.key_sha256},{"cache_entry",effective.hit},{"qualified_trace_sha256",entry.trace_sha},{"trace_equivalence","CUDA exact complete per-node source/fixedfallback/selectedmask/test-order key"}};
        auto data=std::make_shared<QualifiedProposal::Data>();data->owner=p_->owner;data->session_binding=p_->status.binding;
        data->directory=output;data->runtime_file=entry.runtime_file;data->runtime_sha=entry.runtime_sha;data->bytes=entry.bytes;data->nodes=entry.nodes;
        data->cache_hit=true;data->cache_id=effective.hit;data->effective_key_sha=effective.key_sha256;
        data->policy_version=action.policy_version;data->order_injected=action.bits!=nullptr;data->order=effective.order;
        if(sample){data->sample_bound=true;data->sampled_seed=sample->seed;data->sampled_episode=sample->episode;}
        data->trace_bytes=traces.dump(2)+"\n";data->trace_sha=dpnative::sha256(data->trace_bytes);
        J certificate{{"format","source-qualified-effective-cache-hit-1"},{"complete",true},{"source_equivalent",true},{"whole_source_conversion_complete",true},{"session_binding",p_->status.binding},{"process_id",p_->pid},{"source_sha256",p_->status.source_sha256},{"rank_sha256",p_->status.rank_sha256},{"class_word_sha256",p_->status.class_word_sha256},{"cache_entry",effective.hit},{"effective_key_sha256",effective.key_sha256},{"qualified_runtime_path",entry.runtime_file.string()},{"runtime_sha256",entry.runtime_sha},{"runtime_bytes",entry.bytes},{"runtime_nodes",entry.nodes},{"qualified_parent_certificate_sha256",entry.certificate_sha},{"order_trace_sha256",data->trace_sha},{"policy_version",action.policy_version},{"current_full_order44",effective.order},{"universal_lowering_reused",true},{"collector_codec_reused",true},{"fresh_native_grid_audits",0},{"score_refolds",0},{"class_rebuilds",0},{"runtime_serializations",0},{"cache_key_authority","complete exact CUDA word equality; hash only identity"},{"qualification_source","captured private source graph; external publication identity rechecked at finalize"},{"sampling_or_reward_confers_class_authority",false},{"future_process_reuse_authority",false}};
        if(sample){certificate["sampled_seed"]=sample->seed;certificate["sampled_episode"]=sample->episode;certificate["baseline_bytes_frozen_before_sample"]=sample->baseline_bytes;}
        data->certificate_bytes=certificate.dump(2)+"\n";data->certificate_sha=dpnative::sha256(data->certificate_bytes);
        if(p_->options.per_trial_diagnostics){fs::create_directories(output/"audit");data->trace_file=output/"audit/category-order.json";data->certificate_file=output/"certificate.json";
          dpnative::atomic_text(data->trace_file,data->trace_bytes);dpnative::atomic_text(data->certificate_file,data->certificate_bytes);}
        p_->artifact(*data);timing.total=elapsed(whole_start);data->timings=timing;p_->published.push_back(data);
        ++p_->status.proposals;++p_->status.cache_hits;return QualifiedProposal(std::move(data));
      }
    }
    U cache_bytes=p_->cache?p_->cache->retained_device_bytes():0;
    need(cache_bytes<p_->options.maximum_device_bytes-p_->identity->retained_device_bytes(),"cache leaves no lowering stage capacity");
    U cap=p_->options.maximum_device_bytes-p_->identity->retained_device_bytes()-cache_bytes;
    const fs::path artifact_output=p_->options.per_trial_diagnostics?output:
        p_->base/"artifacts"/("entry-"+std::to_string(p_->status.distinct_runtime_artifacts));
    need(!fs::exists(artifact_output)&&fs::create_directories(artifact_output/"runtime")&&fs::create_directories(artifact_output/"audit"),"miss artifact output creation failed");
    before=Clock::now();auto lowered=rl_category_lowering::lower_binary(p_->graph->classes,action,1048576,cap,stop,p_->device);
    need(lowered.lowered.complete,"session proposal lowering incomplete");
    if(p_->cache&&action.bits)p_->cache->audit_traces(lowered.traces);
    timing.lowering=elapsed(before);
    auto lower_receipt=apply_receipt::describe(lowered.lowered);auto traces=trace_payload(lowered);
    before=Clock::now();auto canonical=apply_receipt::canonical(p_->graph->classes,p_->graph->partitions,std::move(lowered.lowered),artifact_output.string(),cap);
    timing.canonical=elapsed(before);before=Clock::now();
    auto encoded=codec::encode(canonical.model);need(encoded.size()<=p_->options.maximum_runtime_bytes,"proposal runtime byte cap");
    auto decoded=codec::decode(encoded);same(canonical.model,decoded);
    auto runtime_sha=dpnative::sha256(encoded);dpnative::atomic_text(artifact_output/"runtime/model.bin",encoded);
    same(decoded,codec::read_model(artifact_output/"runtime/model.bin",runtime_sha,p_->options.maximum_runtime_bytes));
    dpnative::atomic_json(artifact_output/"audit/category-order.json",traces);
    timing.codec_IO=elapsed(before);need(!stop||!stop(),"cancelled_before_proposal_certificate");
    before=Clock::now();p_->audit_graph();timing.identity_after=elapsed(before);
    if(p_->cache)p_->cache->audit_order(action.bits);
    same(decoded,codec::read_model(artifact_output/"runtime/model.bin",runtime_sha,p_->options.maximum_runtime_bytes));
    need(J::parse(dpnative::read_text(artifact_output/"audit/category-order.json"))==traces,"proposal trace artifact mutated");
    J certificate{{"format","source-qualified-session-proposal-1"},{"complete",true},{"source_equivalent",true},
      {"session_binding",p_->status.binding},{"process_id",p_->pid},{"source_sha256",p_->status.source_sha256},
      {"rank_sha256",p_->status.rank_sha256},{"class_word_sha256",p_->status.class_word_sha256},
      {"baseline_runtime_sha256",p_->status.baseline_runtime_sha256},{"base_native_audit_sha256",p_->status.base_native_audit_sha256},
      {"runtime_sha256",runtime_sha},{"runtime_bytes",encoded.size()},{"runtime_nodes",decoded.nodes.size()},
      {"policy_version",action.policy_version},{"order_trace_sha256",dp_streaming::sha256_file(artifact_output/"audit/category-order.json")},
      {"universal_lowering",lower_receipt},{"collector_input_sha256",canonical.collector_input_sha256},
      {"collector_output_sha256",canonical.collector_output_sha256},{"codec_readback_words_equal",true},
      {"fresh_native_grid_audits",0},{"score_refolds",0},{"class_rebuilds",0},
      {"certificate_rule","candidate equals immutable class graph equals fully native-qualified baseline on every declared domain input"},
      {"sampling_or_reward_confers_class_authority",false},{"whole_source_conversion_complete",true},
      {"future_process_reuse_authority",false},{"construction_actions","within-group category permutation only"}};
    certificate["qualification_source"]="captured immutable same-process source graph; external publication identity rechecked at Session::finalize";
    certificate["effective_key_sha256"]=effective.key_sha256;certificate["qualified_cache_miss"]=p_->cache!=nullptr;
    if(sample){certificate["sampled_seed"]=sample->seed;certificate["sampled_episode"]=sample->episode;
      certificate["baseline_bytes_frozen_before_sample"]=sample->baseline_bytes;}
    dpnative::atomic_json(artifact_output/"certificate.json",certificate);
    auto data=std::make_shared<QualifiedProposal::Data>();data->owner=p_->owner;
    data->bytes=encoded.size();data->nodes=decoded.nodes.size();data->policy_version=action.policy_version;
    data->order_injected=lowered.order_injected;data->order=lowered.selected_order;
    if(sample){data->sample_bound=true;data->sampled_seed=sample->seed;data->sampled_episode=sample->episode;}
    data->directory=output;data->runtime_sha=runtime_sha;data->session_binding=p_->status.binding;
    data->runtime_file=artifact_output/"runtime/model.bin";data->certificate_file=artifact_output/"certificate.json";data->trace_file=artifact_output/"audit/category-order.json";
    data->effective_key_sha=effective.key_sha256;
    data->certificate_sha=dp_streaming::sha256_file(data->certificate_file);
    data->trace_sha=dp_streaming::sha256_file(data->trace_file);
    data->certificate_bytes=dpnative::read_text(data->certificate_file);data->trace_bytes=dpnative::read_text(data->trace_file);
    if(p_->cache){data->cache_id=p_->cache->admit();if(data->cache_id!=UINT64_MAX){need(data->cache_id==p_->cache_proofs.size(),"cache key/proof admission order");p_->cache_proofs.push_back(data);}
      p_->status.cache_retained_device_bytes=p_->cache->retained_device_bytes();}
    timing.total=elapsed(whole_start);data->timings=timing;p_->published.push_back(data);
    ++p_->status.proposals;++p_->status.cache_misses;++p_->status.distinct_runtime_artifacts;return QualifiedProposal(std::move(data));
  }catch(const std::exception&e){pending["error"]=e.what();fs::create_directories(output);dpnative::atomic_json(output/"incomplete-proposal.json",pending);throw;}
}

rl_category_policy::Credit Session::credit(const QualifiedProposal&proposal,const rl_category_policy::Policy&policy,U rate)const{
  std::lock_guard lock(p_->mutex);p_->audit_graph();const auto&v=*proposal.p_;
  need(v.owner==p_->owner&&v.session_binding==p_->status.binding,"proposal belongs to another source session");
  need(v.sample_bound&&v.order_injected&&v.policy_version&&policy.version()==v.policy_version,"proposal is not a current sampled policy action");
  const auto trace=policy.trace();
  need(trace[0].seed==v.sampled_seed&&trace[0].episode==v.sampled_episode,"proposal sampled identity changed before credit");
  detail::audit_policy_order(v.order,trace,v.policy_version,
      p_->options.maximum_device_bytes-p_->identity->retained_device_bytes()-(p_->cache?p_->cache->retained_device_bytes():0),p_->device);
  p_->artifact(v);if(v.cache_id!=UINT64_MAX){need(v.cache_id<p_->cache_proofs.size(),"credit cache proof index");p_->artifact(*p_->cache_proofs[v.cache_id]);}
  rl_category_policy::Credit credit{};credit.sampled_version=v.policy_version;
  credit.sampled_seed=trace[0].seed;credit.sampled_episode=trace[0].episode;
  credit.encoded_bytes=v.bytes;credit.baseline_bytes=p_->status.baseline_bytes;
  credit.learning_rate_bits=rate;credit.complete=1;credit.independently_validated=1;credit.scope_qualified=1;
  return credit;
}
QualifiedProposal Session::propose(const rl_category_policy::OrderView&sample,fs::path output,const Stop&stop){
  return propose_impl({sample.order44,sample.version},&sample,std::move(output),stop);
}
}
