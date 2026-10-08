#include "session.hpp"
#include "policy_state.hpp"
#include "class_io.hpp"
#include <chrono>
#include <csignal>
#include <iostream>
namespace fs=std::filesystem;using J=dpnative::json;using U=rl_qualified_session::U;
using Clock=std::chrono::steady_clock;
static volatile std::sig_atomic_t cancelled=0;
static void cancel_handler(int){cancelled=1;}
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
static U integer(const char*s){std::string v=s;need(!v.empty()&&v.find_first_not_of("0123456789")==std::string::npos,"unsigned metadata required");return std::stoull(v);}
int main(int argc,char**argv){fs::path out;bool owns_output=false;J result{{"passed",false},{"accepted_learned_state",false},{"whole_source_conversion_complete",false}};
try{
  need(argc>=14&&argc<=16,"usage session_resident_loop MODEL SOURCE_SHA LIB FRESH_OUT BUILD_TOKEN TRIALS SEED0 EPISODE0 RATE_BITS MAX_NATIVE_CELLS NATIVE_BATCH_ROWS CHUNK_RECORDS WARMUP_EPISODES [POLICY_STATE_JSON] [--diagnostic-trajectories]");
  std::signal(SIGTERM,cancel_handler);std::signal(SIGINT,cancel_handler);auto stop=[](){return cancelled!=0;};
  const auto process_start=Clock::now();U trials=integer(argv[6]),seed0=integer(argv[7]),episode0=integer(argv[8]),rate=integer(argv[9]),chunk=integer(argv[12]),warmups=integer(argv[13]);
  need(trials&&chunk&&chunk<=UINT_MAX&&warmups&&warmups<=1024&&trials<=UINT64_MAX-seed0&&trials<=UINT64_MAX-episode0,"resident schedule/warmup extent");
  bool diagnostic=false;fs::path input;std::string input_bytes,input_sha;session_policy_state::Imported imported;
  for(int k=14;k<argc;++k){if(std::string(argv[k])=="--diagnostic-trajectories"){need(!diagnostic,"duplicate diagnostic flag");diagnostic=true;}
    else{need(input.empty(),"duplicate policy state argument");input=fs::absolute(argv[k]);input_bytes=dpnative::read_text(input);input_sha=dpnative::sha256(input_bytes);imported=session_policy_state::parse(J::parse(input_bytes));}}
  out=fs::absolute(argv[4]);need(!fs::exists(out)&&fs::create_directories(out),"resident loop requires fresh output");owns_output=true;
  result["format"]="source-qualified-resident-Policy1-loop-1";result["source_sha256"]=argv[2];result["sampler_schema_version"]=1;
  result["schedule"]={{"trials",trials},{"seed0",seed0},{"episode0",episode0},{"seed_stride",1},{"episode_stride",1},{"learning_rate_bits",rate},{"chunk_records",chunk},{"initial_policy",input.empty()?"uniform zero logits":"GPU-validated imported exact words"},{"frozen_rng_episode_version_alias_caveat",true}};
  auto save=[&](const char*phase){result["phase"]=phase;result["whole_process_wall_seconds"]=seconds(process_start);dpnative::atomic_json(out/"result.json",result);};save("setup");
  rl_qualified_session::Options options;options.maximum_validation_cells=integer(argv[10]);options.native_batch_rows=integer(argv[11]);
  rl_qualified_session::Session session(argv[3],argv[1],argv[2],out/"base",argv[5],options,stop);auto base=session.summary();
  result["setup_wall_seconds"]=base.setup_seconds;result["base"]={{"runtime_bytes",base.baseline_bytes},{"runtime_nodes",base.baseline_nodes},{"runtime_sha256",base.baseline_runtime_sha256},{"native_cells",base.base_native_cells},{"class_builds",base.class_builds},{"native_audits",base.base_native_audits},{"session_binding",base.binding}};
  // A separate uniform warmup policy discovers/certifies costs only. It never
  // updates and never substitutes warmup gradients for the training policy.
  const auto warm_start=Clock::now();rl_category_policy::Policy warm_policy;J warm=J::array();
  for(U k=0;k<warmups;++k){need(!stop(),"cancelled during cost warmup");auto sample=warm_policy.sample44(1200+k,1+k,base.baseline_bytes);
    auto proposal=session.propose(sample,out/("warm-"+std::to_string(k)),stop);
    warm.push_back({{"warmup",k},{"seed",sample.seed},{"episode",sample.episode},{"policy_version",sample.version},{"updates",0},{"entry_runtime_sha256",proposal.runtime_sha256()},{"encoded_bytes",proposal.encoded_bytes()},{"cache_hit",proposal.cache_hit()},{"certificate_sha256",proposal.certificate_sha256()}});
  }
  result["cost_warmup_wall_seconds"]=seconds(warm_start);result["cost_warmup"]=warm;result["warmup_policy_never_updated"]=true;
  need(!stop(),"cancelled before resident sealing");auto seal_start=Clock::now();auto table=session.seal_resident_table();result["table_seal_wall_seconds"]=seconds(seal_start);
  J artifacts=J::array();for(const auto&a:table.artifacts())artifacts.push_back({{"entry",a.entry},{"runtime_bytes",a.encoded_bytes},{"runtime_nodes",a.nodes},{"runtime_sha256",a.runtime_sha256},{"certificate_sha256",a.certificate_sha256},{"runtime_path",a.runtime_path.string()}});
  result["qualified_cost_table"]={{"generation",table.generation()},{"entries",table.entries()},{"source_binding",table.source_binding()},{"semantics_binding",table.semantics_binding()},{"artifacts",artifacts},{"imported_cost_arrays_authorize_entries",false},{"unseen_key_behavior","fatal before current credit; completed prefix remains diagnostic"}};
  rl_category_policy::Policy initial;if(!input.empty())initial.upload(imported.words,imported.version);const auto initial_words=initial.logit_words();const auto initial_version=initial.version();
  auto initial_state=session_policy_state::words(initial_version,initial_words);initial_state["parent_policy_sha256"]=input_sha;dpnative::atomic_json(out/"policy-initial.json",initial_state);const auto initial_bytes=dpnative::read_text(out/"policy-initial.json");
  rl_qualified_session::ResidentOptions resident_options;resident_options.maximum_chunk_records=chunk;resident_options.retain_full_trajectory_words=diagnostic;
  auto engine=session.make_resident(table,initial,rate,{seed0,episode0,trials},resident_options);
  U done=0,chunks=0,incumbent=0,final_version=initial_version,updates=0;std::array<U,44>final_words=initial_words;J records=J::array();std::ostringstream journal;
  save("resident_chunks");const auto loop_start=Clock::now();
  while(done<trials){need(!stop(),"cancelled before resident chunk");U count=std::min(chunk,trials-done);auto t=Clock::now();auto batch=engine->run_chunk({seed0+done,episode0+done,count});double wall=seconds(t);
    need(batch.complete&&batch.completed==count&&batch.table_generation==table.generation(),"resident chunk incomplete; no accepted export");
    for(U k=0;k<count;++k){const auto&r=batch.records[k];need(r.error==0&&r.entry<table.artifacts().size()&&r.seed==seed0+done+k&&r.episode==episode0+done+k,"resident transport record mismatch");
      const auto&a=table.artifacts()[r.entry];need(a.encoded_bytes==r.encoded_bytes,"resident certified cost/record mismatch");
      J row{{"trial",done+k},{"seed",r.seed},{"episode",r.episode},{"sampled_version",r.sampled_version},{"new_version",r.committed_version},{"updates",r.updates},{"entry",r.entry},{"runtime_bytes",a.encoded_bytes},{"runtime_nodes",a.nodes},{"runtime_sha256",a.runtime_sha256},{"incumbent_entry",r.incumbent_entry},{"baseline_bytes",r.baseline_bytes},{"learning_rate_bits",r.learning_rate_bits},{"reward_word",r.reward_bits},{"order44",r.order44},{"after_logit_words",r.after_logit_words},{"table_generation",batch.table_generation},{"fresh_current_score",true},{"cached_gradients_used",false}};
      records.push_back(row);J line{{"trial",row}};if(diagnostic){need(batch.diagnostic_trajectories.size()==count,"diagnostic trajectory extent");line["full_trajectory_words"]=session_policy_state::trajectory(batch.diagnostic_trajectories[k]);}journal<<line.dump()<<'\n';
    }
    done+=count;++chunks;incumbent=batch.incumbent_entry;final_words=batch.final_logit_words;final_version=batch.policy_version;updates=batch.updates;
    result["completed_episodes"]=done;result["completed_chunks"]=chunks;result["last_chunk_wall_seconds"]=wall;
  }
  result["steady_loop_wall_seconds"]=seconds(loop_start);need(done==trials&&updates==trials&&!stop(),"resident schedule cancelled or incomplete");
  auto publish_start=Clock::now();const auto journal_bytes=journal.str();dpnative::atomic_text(out/"episodes.jsonl",journal_bytes);need(dpnative::read_text(out/"episodes.jsonl")==journal_bytes,"resident journal readback mismatch");result["checkpoint_publication_wall_seconds"]=seconds(publish_start);
  auto finalize_start=Clock::now();session.finalize_resident(table,*engine);result["finalization_wall_seconds"]=seconds(finalize_start);auto final=session.summary();
  need(final.publication_identity_current&&!stop()&&incumbent<table.artifacts().size(),"resident final publication authority absent");const auto&best=table.artifacts()[incumbent];
  const auto runtime=dpnative::read_text(best.runtime_path);need(dpnative::sha256(runtime)==best.runtime_sha256,"resident incumbent artifact changed");fs::create_directories(out/"incumbent");dpnative::atomic_text(out/"incumbent/model.bin",runtime);need(dpnative::read_text(out/"incumbent/model.bin")==runtime,"resident incumbent readback mismatch");
  need(dpnative::read_text(out/"policy-initial.json")==initial_bytes&&dpnative::read_text(out/"episodes.jsonl")==journal_bytes,"resident evidence changed before publication");if(!input.empty())need(dpnative::read_text(input)==input_bytes,"imported state changed before publication");
  J qualification{{"schema",U(1)},{"source_sha256",argv[2]},{"session_binding",base.binding},{"library_sha256",final.library_sha256},{"rank_sha256",final.rank_sha256},{"class_word_sha256",final.class_word_sha256},{"build_binding_sha256",final.build_binding_sha256},{"base_native_audit_sha256",final.base_native_audit_sha256},{"table_generation",table.generation()},{"cost_table_source_binding",table.source_binding()},{"cost_table_semantics_binding",table.semantics_binding()},{"cost_artifacts",artifacts},{"completed_requested_schedule",true},{"completed_episodes",done},{"runtime_sha256",best.runtime_sha256},{"runtime_bytes",best.encoded_bytes},{"episode_journal_sha256",dpnative::sha256(journal_bytes)},{"policy_initial_sha256",dpnative::sha256(initial_bytes)},{"fresh_sample_score_once_each",true},{"final_external_identity_checked",true},{"future_process_reuse_authority",false}};
  dpnative::atomic_json(out/"episode-qualification.json",qualification);const auto qualification_bytes=dpnative::read_text(out/"episode-qualification.json");need(J::parse(qualification_bytes)==qualification,"resident qualification readback mismatch");
  auto learned=session_policy_state::words(final_version,final_words);learned["updates_this_process"]=updates;learned["trained_source_sha256"]=argv[2];learned["source_runtime_sha256"]=best.runtime_sha256;learned["episode_qualification_sha256"]=dpnative::sha256(qualification_bytes);learned["parent_policy_sha256"]=input_sha;
  dpnative::atomic_json(out/"learned-policy.json",learned);const auto learned_bytes=dpnative::read_text(out/"learned-policy.json");need(J::parse(learned_bytes)==learned&&session_policy_state::parse(J::parse(learned_bytes)).words==final_words&&!stop(),"resident learned-state readback/cancellation failure");
  result["trials"]=records;result["incumbent"]={{"entry",incumbent},{"runtime_bytes",best.encoded_bytes},{"runtime_nodes",best.nodes},{"runtime_sha256",best.runtime_sha256},{"selection","strict byte improvement on CUDA; equal cost retains previous entry"}};
  result["qualified_session"]={{"class_builds",final.class_builds},{"base_native_audits",final.base_native_audits},{"warmup_proposals",final.proposals},{"cache_hits",final.cache_hits},{"cache_misses",final.cache_misses},{"distinct_runtime_artifacts",final.distinct_runtime_artifacts},{"final_external_identity_checked",true}};
  result["learned_policy_sha256"]=dpnative::sha256(learned_bytes);result["episode_qualification_sha256"]=dpnative::sha256(qualification_bytes);
  result["policy_update_performed"]=true;result["accepted_learned_state"]=true;result["whole_source_conversion_complete"]=true;result["passed"]=true;
  result["CPU_model_math"]=false;result["host_per_episode_launches_or_downloads"]=false;result["per_episode_files"]=false;result["diagnostic_full_trajectories"]=diagnostic;
  result["timing_scope"]="host wall: synchronized resident CUDA chunks and boundary record encoding; separate cost warmup/seal, final checkpoint IO and final external audit; no GPU-only speed claim";
  save("complete");std::cout<<result.dump(2)<<'\n';return 0;
}catch(const std::exception&e){result["error"]=e.what();result["passed"]=false;result["accepted_learned_state"]=false;result["whole_source_conversion_complete"]=false;if(owns_output)try{dpnative::atomic_json(out/"result.json",result);}catch(...){}std::cerr<<e.what()<<'\n';return 1;}}
