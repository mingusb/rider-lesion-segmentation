#include "resident_session.hpp"
#include "resident_policy1_device.cuh"
namespace resident_policy1 = rl_qualified_session::resident_policy1;
#include "cooperative_policy1_device.cuh"
#include "resident_effective_key.cuh"
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <mutex>

namespace rl_qualified_session {
using U=ResidentWord;
namespace resident_detail {
void need(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U add(U a,U b){need(b<=UINT64_MAX-a,"resident byte sum overflow");return a+b;}
U mul(U a,U b){need(!b||a<=UINT64_MAX/b,"resident byte product overflow");return a*b;}
struct Budget {U cap,used=0;void take(U n){need(n<=cap-used,"resident device byte cap");used+=n;}};
template<class T>struct Buf {
  Budget&budget;U count,bytes;T*p=nullptr;
  Buf(Budget&b,U n):budget(b),count(n),bytes(mul(n,sizeof(T))){budget.take(bytes);if(bytes){auto e=cudaMalloc(reinterpret_cast<void**>(&p),bytes);if(e!=cudaSuccess){budget.used-=bytes;cu(e);}e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){cudaFree(p);p=nullptr;budget.used-=bytes;cu(e);}}}
  ~Buf(){if(p)cudaFree(p);budget.used-=bytes;}
  Buf(const Buf&)=delete;
  void put(const T*host){if(bytes)cu(cudaMemcpy(p,host,bytes,cudaMemcpyHostToDevice));}
  std::vector<T>get(U n=UINT64_MAX)const {if(n==UINT64_MAX)n=count;need(n<=count,"resident download extent");std::vector<T>v(n);if(n)cu(cudaMemcpy(v.data(),p,mul(n,sizeof(T)),cudaMemcpyDeviceToHost));return v;}
};
using State=resident_policy1::State;using Meta=resident_policy1::Meta;using Trace=rl_category_policy::Trace;
// Header + complete ClassIdentity words + rows(id,generation,cost,nodes,key).
inline constexpr U header_words=16;
struct Control {U requested=0,completed=0,failed_episode=UINT64_MAX,version=0,updates=0,incumbent=0;int error=0;unsigned reserved=0;};
static_assert(sizeof(Control)==56);
__global__ void table_equal(const U*a,const U*b,U n,int*error){U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n&&a[i]!=b[i])atomicCAS(error,0,101);}
__global__ void table_shape(const U*table,U n,int*error){if(threadIdx.x||blockIdx.x)return;
  if(n<header_words||table[0]!=1||!table[1]||!table[2]||table[3]!=2+45*table[2]||table[4]<37+7*table[2]||!table[5]||table[6]>(U(1)<<53)||
     table[4]>n-header_words||table[5]>(n-header_words-table[4])/(4+table[3])||header_words+table[4]+table[5]*(4+table[3])!=n){*error=102;return;}
  const U*source=table+header_words;if(source[22]!=table[2]||source[0]>=table[2]){*error=103;return;}
  const U*rows=source+table[4];for(U e=0;e<table[5];++e){const U*r=rows+e*(4+table[3]);
    if(r[0]!=e||r[1]!=table[1]||r[2]>(U(1)<<53)||!r[3]||r[4]!=source[0]||r[5]!=table[2]||(e==0&&r[2]!=table[6])){*error=104;return;}
  }
}
__global__ void initialize_state(State*s,Meta*m,U*incumbent,const U*words,U version,U rate,int*error){if(blockIdx.x||threadIdx.x)return;
  if(!version){*error=105;return;}double lr=__longlong_as_double(rate);if(!isfinite(lr)||lr<=0||lr>1){*error=106;return;}
  for(unsigned j=0;j<44;++j)if(!isfinite(__longlong_as_double(words[j]))){*error=107;return;}
  *s=State{};for(unsigned j=0;j<44;++j)s->logits[j]=__longlong_as_double(words[j]);s->version=version;*m=Meta{};*incumbent=0;
}
__global__ void chunk_kernel(const U*table,U generation,State*s,Meta*m,Trace*traces,unsigned*order,
    U*key,U*incumbent,U rate,ResidentSchedule schedule,ResidentEpisodeRecord*records,Trace*diagnostics,Control*out){
  __shared__ int different;__shared__ U matched;
  if(!threadIdx.x){*out=Control{};out->requested=schedule.episodes;out->version=s->version;out->updates=s->updates;out->incumbent=*incumbent;
    if(table[1]!=generation)out->error=108;else if(*incumbent>=table[5])out->error=111;}
  __syncthreads();if(out->error)return;
  const U nn=table[2],words=table[3],entries=table[5],baseline=table[6];
  const U*source=table+header_words;const U*rows=source+table[4];
  for(U step=0;step<schedule.episodes;++step){
    resident_cooperative_policy1::cooperative_order(s,traces,order,m,schedule.seed0+step,schedule.episode0+step,baseline);
    __syncthreads();
    if(!threadIdx.x){out->error=m->error;matched=UINT64_MAX;}
    __syncthreads();if(out->error){if(!threadIdx.x)out->failed_episode=schedule.episode0+step;return;}
    for(U id=threadIdx.x;id<nn;id+=blockDim.x)effective_key_node(source,nn,order,true,key,&out->error,id);
    __syncthreads();if(out->error){if(!threadIdx.x)out->failed_episode=schedule.episode0+step;return;}
    for(U e=0;e<entries;++e){if(!threadIdx.x)different=0;__syncthreads();const U*other=rows+e*(4+words)+4;
      for(U k=threadIdx.x;k<words;k+=blockDim.x)if(key[k]!=other[k])atomicExch(&different,1);
      __syncthreads();if(!threadIdx.x&&!different){if(matched!=UINT64_MAX)out->error=109;else matched=e;}__syncthreads();
    }
    if(!threadIdx.x&&matched==UINT64_MAX)out->error=110;__syncthreads();
    if(out->error){if(!threadIdx.x)out->failed_episode=schedule.episode0+step;return;}
    if(!threadIdx.x){const U*entry=rows+matched*(4+words);rl_category_policy::Credit c{};
      c.sampled_version=m->version;c.sampled_seed=m->seed;c.sampled_episode=m->episode;c.encoded_bytes=entry[2];c.baseline_bytes=baseline;c.learning_rate_bits=rate;
      c.complete=c.independently_validated=c.scope_qualified=1;
      resident_policy1::apply_update(s,traces,m,c);out->error=m->error;
      if(!out->error){if(entry[2]<rows[*incumbent*(4+words)+2])*incumbent=matched;
        ResidentEpisodeRecord r{};r.seed=c.sampled_seed;r.episode=c.sampled_episode;r.sampled_version=c.sampled_version;r.committed_version=s->version;
        r.updates=s->updates;r.entry=matched;r.encoded_bytes=entry[2];r.incumbent_entry=*incumbent;r.baseline_bytes=baseline;r.learning_rate_bits=rate;r.reward_bits=__double_as_longlong(m->reward);
        for(unsigned j=0;j<44;++j){r.order44[j]=order[j];r.after_logit_words[j]=__double_as_longlong(s->logits[j]);}records[step]=r;
        if(diagnostics){diagnostics[2*step]=traces[0];diagnostics[2*step+1]=traces[1];}
        ++out->completed;out->version=s->version;out->updates=s->updates;out->incumbent=*incumbent;
      }else out->failed_episode=schedule.episode0+step;
    }
    __syncthreads();if(out->error)return;
  }
}
__global__ void state_words(const State*s,U*out){unsigned j=threadIdx.x;if(j<44)out[j]=__double_as_longlong(s->logits[j]);}
} // resident_detail

U resident_device_plan(U source_words,U nodes,U entries,U records,bool diagnostics){using namespace resident_detail;
  need(nodes&&entries&&records&&records<=UINT_MAX&&source_words>=add(37,mul(7,nodes)),"resident plan shape");
  U keywords=add(2,mul(45,nodes));U payload=add(add(header_words,source_words),mul(entries,add(4,keywords)));
  U table=add(mul(16,payload),4);
  U engine=add(add(add(sizeof(State),sizeof(Meta)),mul(2,sizeof(Trace))),add(176,mul(8,keywords)));
  engine=add(engine,add(add(8,352),add(sizeof(Control),4)));
  engine=add(engine,mul(records,sizeof(ResidentEpisodeRecord)));
  if(diagnostics)engine=add(engine,mul(mul(2,records),sizeof(Trace)));
  return add(table,engine);
}

struct QualifiedCostTable::Data {
  using U=ResidentWord;
  std::shared_ptr<const void>owner;U process_id,generation,baseline;int device;
  std::string source_binding,semantics_binding;std::vector<ResidentArtifact>artifacts;
  mutable std::mutex audit_mutex;
  resident_detail::Budget budget;resident_detail::Buf<U>table,reference;resident_detail::Buf<int>error;
  Data(std::shared_ptr<const void>o,U pid,int gpu,U gen,U base,std::string src,std::string semantics,
      const std::vector<U>&payload,std::vector<ResidentArtifact>a,U cap):owner(std::move(o)),process_id(pid),generation(gen),baseline(base),device(gpu),
      source_binding(std::move(src)),semantics_binding(std::move(semantics)),artifacts(std::move(a)),budget{cap},table(budget,payload.size()),reference(budget,payload.size()),error(budget,1){
    table.put(payload.data());reference.put(payload.data());resident_detail::table_shape<<<1,1>>>(table.p,table.count,error.p);
    resident_detail::cu(cudaGetLastError());resident_detail::need(error.get()[0]==0,"private resident table shape failed");
  }
  void audit()const{using namespace resident_detail;std::lock_guard lock(audit_mutex);cu(cudaSetDevice(device));cu(cudaMemset(error.p,0,4));
    need(table.count<=U(INT_MAX)*128,"resident table audit launch extent");table_equal<<<unsigned((table.count+127)/128),128>>>(table.p,reference.p,table.count,error.p);
    cu(cudaGetLastError());need(error.get()[0]==0,"immutable resident table changed");}
};
QualifiedCostTable::QualifiedCostTable(std::shared_ptr<const Data>p):p_(std::move(p)){}
QualifiedCostTable QualifiedCostTable::seal(std::shared_ptr<const void>owner,U pid,int device,U generation,U baseline,std::string source_binding,std::string semantics_binding,
    const std::vector<U>&source,const std::vector<std::vector<U>>&keys,std::vector<ResidentArtifact>artifacts,U cap){
  using namespace resident_detail;cu(cudaSetDevice(device));need(owner&&pid==U(getpid())&&generation&&source.size()>=37&&!keys.empty()&&keys.size()==artifacts.size(),"resident seal metadata");
  U nn=source[22],key_words=add(2,mul(45,nn));std::vector<U>payload(header_words);payload[0]=1;payload[1]=generation;payload[2]=nn;payload[3]=key_words;payload[4]=source.size();payload[5]=keys.size();payload[6]=baseline;payload[7]=pid;
  need(source_binding.size()==64&&semantics_binding.size()==64,"resident binding digest extent");
  for(unsigned j=0;j<4;++j){payload[8+j]=std::stoull(source_binding.substr(16*j,16),nullptr,16);payload[12+j]=std::stoull(semantics_binding.substr(16*j,16),nullptr,16);}
  payload.insert(payload.end(),source.begin(),source.end());
  for(U e=0;e<keys.size();++e){need(keys[e].size()==key_words&&artifacts[e].entry==e,"resident key/proof extent");
    payload.insert(payload.end(),{e,generation,artifacts[e].encoded_bytes,artifacts[e].nodes});payload.insert(payload.end(),keys[e].begin(),keys[e].end());}
  return QualifiedCostTable(std::make_shared<const Data>(std::move(owner),pid,device,generation,baseline,std::move(source_binding),std::move(semantics_binding),payload,std::move(artifacts),cap));
}
void QualifiedCostTable::require_owner(const std::shared_ptr<const void>&owner,U pid,const std::string&binding)const{
  resident_detail::need(p_->owner==owner&&p_->process_id==pid&&pid==U(getpid())&&p_->source_binding==binding,"resident table belongs to another private Session");p_->audit();
}
U QualifiedCostTable::entries()const{return p_->artifacts.size();}U QualifiedCostTable::generation()const{return p_->generation;}
const std::string&QualifiedCostTable::source_binding()const{return p_->source_binding;}
const std::string&QualifiedCostTable::semantics_binding()const{return p_->semantics_binding;}
const std::vector<ResidentArtifact>&QualifiedCostTable::artifacts()const{return p_->artifacts;}

struct ResidentTraining::Impl {
  using U=ResidentWord;std::shared_ptr<const QualifiedCostTable::Data>table;
  ResidentSchedule declared;ResidentOptions options;U rate,completed=0;bool faulted=false;
  mutable std::mutex mutex;
  resident_detail::Budget budget;resident_detail::Buf<resident_detail::State>state;
  resident_detail::Buf<resident_detail::Meta>meta;resident_detail::Buf<rl_category_policy::Trace>traces,diagnostics;
  resident_detail::Buf<unsigned>order;resident_detail::Buf<U>key,incumbent,words;
  resident_detail::Buf<ResidentEpisodeRecord>records;resident_detail::Buf<resident_detail::Control>control;resident_detail::Buf<int>error;
  Impl(const QualifiedCostTable&q,const std::array<U,44>&initial,U version,U lr,ResidentSchedule plan,ResidentOptions opt):table(q.p_),declared(plan),options(opt),rate(lr),
      budget{table->budget.cap-table->budget.used},state(budget,1),meta(budget,1),traces(budget,2),diagnostics(budget,opt.retain_full_trajectory_words?resident_detail::mul(2,opt.maximum_chunk_records):0),
      order(budget,44),key(budget,resident_detail::add(2,resident_detail::mul(45,table->table.get(16)[2]))),incumbent(budget,1),words(budget,44),records(budget,opt.maximum_chunk_records),control(budget,1),error(budget,1){
    using namespace resident_detail;need(opt.maximum_chunk_records&&opt.maximum_chunk_records<=UINT_MAX&&plan.episodes&&plan.seed0<=UINT64_MAX-plan.episodes&&plan.episode0<=UINT64_MAX-plan.episodes,"resident schedule/record extent");
    words.put(initial.data());initialize_state<<<1,1>>>(state.p,meta.p,incumbent.p,words.p,version,lr,error.p);cu(cudaGetLastError());need(error.get()[0]==0,"resident initial Policy1 state rejected");
  }
};
ResidentTraining::ResidentTraining(const QualifiedCostTable&q,const std::array<U,44>&words,U version,U rate,ResidentSchedule plan,ResidentOptions opt){
  resident_detail::cu(cudaSetDevice(q.p_->device));q.p_->audit();p_=std::make_unique<Impl>(q,words,version,rate,plan,opt);
}
ResidentTraining::~ResidentTraining(){if(p_){cudaSetDevice(p_->table->device);p_.reset();}}
ResidentChunk ResidentTraining::run_chunk(ResidentSchedule schedule){using namespace resident_detail;
  std::lock_guard lock(p_->mutex);
  need(p_->table->process_id==U(getpid())&&!p_->faulted,"resident engine process/fault state");
  need(schedule.episodes&&schedule.episodes<=p_->options.maximum_chunk_records&&schedule.episodes<=p_->declared.episodes-p_->completed&&
       schedule.seed0==p_->declared.seed0+p_->completed&&schedule.episode0==p_->declared.episode0+p_->completed,"resident schedule is not exact next uncommitted prefix");
  cu(cudaSetDevice(p_->table->device));ResidentChunk out;out.requested=schedule.episodes;out.table_generation=p_->table->generation;
  try{p_->table->audit();chunk_kernel<<<1,128>>>(p_->table->table.p,p_->table->generation,p_->state.p,p_->meta.p,p_->traces.p,p_->order.p,p_->key.p,p_->incumbent.p,p_->rate,schedule,p_->records.p,p_->diagnostics.p,p_->control.p);
    cu(cudaGetLastError());const auto c=p_->control.get()[0];out.CUDA_executed=true;out.completed=c.completed;out.failed_episode=c.failed_episode;out.error=c.error;
    out.policy_version=c.version;out.updates=c.updates;out.incumbent_entry=c.incumbent;p_->completed+=c.completed;
    p_->table->audit();state_words<<<1,64>>>(p_->state.p,p_->words.p);cu(cudaGetLastError());auto final=p_->words.get();std::copy(final.begin(),final.end(),out.final_logit_words.begin());
    out.records=p_->records.get(c.completed);if(p_->options.retain_full_trajectory_words){auto traces=p_->diagnostics.get(mul(2,c.completed));for(U k=0;k<c.completed;++k)out.diagnostic_trajectories.push_back({traces[2*k],traces[2*k+1]});}
    out.complete=!out.error&&out.completed==out.requested;if(!out.complete)p_->faulted=true;
  }catch(...){p_->faulted=true;throw;}return out;
}
void ResidentTraining::require_complete(const QualifiedCostTable&q)const{
  std::lock_guard lock(p_->mutex);
  resident_detail::need(q.p_==p_->table&&!p_->faulted&&p_->completed==p_->declared.episodes&&p_->table->process_id==U(getpid()),"resident requested schedule did not complete");p_->table->audit();
}
namespace testing {
namespace {
using namespace resident_detail;
__global__ void literal_source(U*source,U*keys){if(blockIdx.x||threadIdx.x)return;
  for(unsigned k=0;k<77;++k)source[k]=0;
  source[0]=3;source[21]=(U(1)<<44)-1;source[22]=4;source[23]=3;
  for(unsigned k=0;k<3;++k){source[37+7*k]=0;source[38+7*k]=12;source[39+7*k]=k;}
  source[58]=1;source[59]=11;source[62]=0;source[63]=3;
  U soil=((U(1)<<44)-1)^U(15);source[67]=soil&~((U(1)<<6)|(U(1)<<13));source[68]=0;
  source[71]=U(1)<<6;source[72]=1;source[75]=U(1)<<13;source[76]=2;
  unsigned order[44];for(unsigned k=0;k<44;++k)order[k]=k;int error=0;
  for(U id=0;id<4;++id)effective_key_node(source,4,order,true,keys,&error,id);
  order[6]=13;order[13]=6;
  for(U id=0;id<4;++id)effective_key_node(source,4,order,true,keys+182,&error,id);
}
__device__ U literal_match(const U*table,const unsigned*order,U*key,int*error){
  const U*source=table+header_words;U nn=table[2],words=table[3];
  for(U id=0;id<nn;++id)effective_key_node(source,nn,order,true,key,error,id);
  const U*rows=source+table[4];for(U e=0;e<table[5];++e){const U*other=rows+e*(4+words)+4;bool equal=true;
    for(U k=0;k<words;++k)equal=equal&&key[k]==other[k];if(equal)return e;}
  *error=110;return UINT64_MAX;
}
__global__ void reference_choose(const U*table,const unsigned*order,U*key,U*result,U*incumbent,int*error){if(blockIdx.x||threadIdx.x)return;
  const U entry=literal_match(table,order,key,error);if(*error)return;const U*rows=table+header_words+table[4];const U*row=rows+entry*(4+table[3]);
  if(row[2]<rows[*incumbent*(4+table[3])+2])*incumbent=entry;
  result[0]=entry;result[1]=row[2];result[2]=*incumbent;result[3]=__double_as_longlong(double(table[6])-double(row[2]));
}
__global__ void bytes_equal(const unsigned char*a,const unsigned char*b,U n,int*error){U id=U(blockIdx.x)*blockDim.x+threadIdx.x;if(id<n&&a[id]!=b[id])atomicCAS(error,0,1);}
__global__ void find_missing_seed(const U*table,U*out,U*key,int*error){if(blockIdx.x||threadIdx.x)return;
  State state{};Meta meta{};Trace traces[2];unsigned order[44];
  for(U seed=0;seed<1024;++seed){resident_policy1::sample_order(&state,traces,order,&meta,seed,77,table[6]);if(meta.error){*error=112;return;}
    if(literal_match(table,order,key,error)==1){*out=seed;return;}}
  *error=113;
}
}
ResidentCheckReport resident_gpu_checks(int device){using namespace resident_detail;cu(cudaSetDevice(device));ResidentCheckReport report;
  auto check=[&](bool value,const char*s){++report.assertions;need(value,s);};
  auto reject=[&](auto operation){bool failed=false;try{operation();}catch(const std::exception&){failed=true;}check(failed,"resident rejection absent");++report.rejections;};
  Budget budget{4*1024*1024};Buf<U>source(budget,77),keys(budget,364),working_key(budget,182),choice(budget,4),incumbent(budget,1),seed(budget,1);Buf<int>error(budget,1);
  literal_source<<<1,1>>>(source.p,keys.p);cu(cudaGetLastError());auto source_words=source.get(),key_words=keys.get();
  std::vector<std::vector<U>>key_rows{std::vector<U>(key_words.begin(),key_words.begin()+182),std::vector<U>(key_words.begin()+182,key_words.end())};
  std::vector<ResidentArtifact>artifacts{{0,1207,127,"literal-default","literal-proof",{}},{1,1199,126,"literal-opposite","literal-proof",{}}};
  auto owner=std::make_shared<const unsigned char>(0);
  auto table=QualifiedCostTable::seal(owner,U(getpid()),device,1,1207,std::string(64,'1'),std::string(64,'2'),source_words,key_rows,artifacts,4*1024*1024);
  check(table.entries()==2&&table.generation()==1,"literal private table shape");
  table.require_owner(owner,U(getpid()),std::string(64,'1'));
  reject([&]{table.require_owner(std::make_shared<const unsigned char>(0),U(getpid()),std::string(64,'1'));});
  reject([&]{table.require_owner(owner,U(getpid())+1,std::string(64,'1'));});
  reject([&]{table.require_owner(owner,U(getpid()),std::string(64,'3'));});
  const U rate=4560780790824889414ull;std::array<U,44>initial{};ResidentOptions diagnostic;diagnostic.maximum_chunk_records=32;diagnostic.retain_full_trajectory_words=true;
  auto compare_bytes=[&](const void*a,const void*b,U bytes){Buf<unsigned char>actual(budget,bytes),expected(budget,bytes);actual.put(static_cast<const unsigned char*>(a));expected.put(static_cast<const unsigned char*>(b));cu(cudaMemset(error.p,0,4));
    bytes_equal<<<unsigned((bytes+127)/128),128>>>(actual.p,expected.p,bytes,error.p);cu(cudaGetLastError());check(error.get()[0]==0,"resident/reference exact word mismatch");report.episode_word_checks+=bytes/8;};
  auto reference=[&](std::array<U,44>start,U version,ResidentSchedule schedule){
    rl_category_policy::Policy policy(device);policy.upload(start,version);cu(cudaMemset(incumbent.p,0,8));std::vector<ResidentEpisodeRecord>records;std::vector<std::array<Trace,2>>trajectories;
    for(U k=0;k<schedule.episodes;++k){auto view=policy.sample44(schedule.seed0+k,schedule.episode0+k,1207);auto traces=policy.trace();cu(cudaMemset(error.p,0,4));
      reference_choose<<<1,1>>>(table.p_->table.p,view.order44,working_key.p,choice.p,incumbent.p,error.p);cu(cudaGetLastError());check(error.get()[0]==0,"reference literal action lookup failed");auto selected=choice.get();
      rl_category_policy::Credit credit{};credit.sampled_version=view.version;credit.sampled_seed=view.seed;credit.sampled_episode=view.episode;credit.encoded_bytes=selected[1];credit.baseline_bytes=1207;credit.learning_rate_bits=rate;credit.complete=credit.independently_validated=credit.scope_qualified=1;
      ResidentEpisodeRecord r{};cu(cudaMemcpy(r.order44.data(),view.order44,176,cudaMemcpyDeviceToHost));
      auto update=policy.reinforce(credit);check(update.applied,"frozen Policy1 reference update rejected");r.seed=view.seed;r.episode=view.episode;r.sampled_version=view.version;r.committed_version=update.new_version;r.updates=update.updates;r.entry=selected[0];r.encoded_bytes=selected[1];r.incumbent_entry=selected[2];r.baseline_bytes=1207;r.learning_rate_bits=rate;r.reward_bits=selected[3];
      r.after_logit_words=policy.logit_words();records.push_back(r);trajectories.push_back(traces);
    }return std::pair{records,trajectories};
  };
  ResidentSchedule plan{1200,1,32};auto expected=reference(initial,1,plan);
  ResidentTraining all(table,initial,1,rate,plan,diagnostic);auto full=all.run_chunk(plan);
  check(full.complete&&full.completed==32&&full.policy_version==33&&full.updates==32,"whole resident chunk incomplete");all.require_complete(table);
  compare_bytes(full.records.data(),expected.first.data(),32*sizeof(ResidentEpisodeRecord));compare_bytes(full.diagnostic_trajectories.data(),expected.second.data(),32*sizeof(std::array<Trace,2>));
  ResidentTraining one(table,initial,1,rate,plan,diagnostic);std::vector<ResidentEpisodeRecord>one_records;std::vector<std::array<Trace,2>>one_traces;
  for(U k=0;k<32;++k){auto step=one.run_chunk({1200+k,1+k,1});check(step.complete&&step.completed==1,"chunk1 boundary failed");one_records.push_back(step.records[0]);one_traces.push_back(step.diagnostic_trajectories[0]);}one.require_complete(table);
  compare_bytes(one_records.data(),expected.first.data(),32*sizeof(ResidentEpisodeRecord));compare_bytes(one_traces.data(),expected.second.data(),32*sizeof(std::array<Trace,2>));
  ResidentSchedule continuation{2000,101,8};auto continued_expected=reference(full.final_logit_words,33,continuation);
  ResidentTraining continuation_engine(table,full.final_logit_words,33,rate,continuation,diagnostic);auto continued=continuation_engine.run_chunk(continuation);
  check(continued.complete&&continued.policy_version==41&&continued.updates==8,"imported word/version continuation failed");continuation_engine.require_complete(table);
  compare_bytes(continued.records.data(),continued_expected.first.data(),8*sizeof(ResidentEpisodeRecord));compare_bytes(continued.diagnostic_trajectories.data(),continued_expected.second.data(),8*sizeof(std::array<Trace,2>));
  reject([&]{one.run_chunk({1232,33,1});});reject([&]{ResidentTraining bad(table,initial,0,rate,plan,diagnostic);});
  reject([&]{ResidentTraining bad(table,initial,1,0,plan,diagnostic);});auto nonfinite=initial;nonfinite[0]=0x7ff0000000000000ull;reject([&]{ResidentTraining bad(table,nonfinite,1,rate,plan,diagnostic);});
  ResidentTraining prefix(table,initial,1,rate,plan,diagnostic);auto part=prefix.run_chunk({1200,1,4});check(part.complete&&part.completed==4,"diagnostic prefix did not complete");
  reject([&]{prefix.require_complete(table);});reject([&]{prefix.run_chunk({1200,1,1});});reject([&]{prefix.run_chunk({1204,5,0});});reject([&]{prefix.run_chunk({1204,5,33});});
  cu(cudaMemset(error.p,0,4));find_missing_seed<<<1,1>>>(table.p_->table.p,seed.p,working_key.p,error.p);cu(cudaGetLastError());check(error.get()[0]==0,"missing action fixture seed absent");
  auto incomplete=QualifiedCostTable::seal(owner,U(getpid()),device,2,1207,std::string(64,'1'),std::string(64,'2'),source_words,{key_rows[0]},{artifacts[0]},4*1024*1024);
  ResidentSchedule miss_plan{seed.get()[0],77,1};ResidentTraining miss(incomplete,initial,1,rate,miss_plan,diagnostic);auto failure=miss.run_chunk(miss_plan);
  check(!failure.complete&&failure.error==110&&!failure.completed&&failure.failed_episode==77&&failure.policy_version==1&&failure.updates==0&&failure.incumbent_entry==0,"cache miss credited an episode");
  compare_bytes(failure.final_logit_words.data(),initial.data(),44*sizeof(U));reject([&]{miss.require_complete(incomplete);});reject([&]{miss.run_chunk(miss_plan);});
  U corrupt=9;cu(cudaMemcpy(table.p_->table.p+1,&corrupt,8,cudaMemcpyHostToDevice));reject([&]{prefix.run_chunk({1204,5,1});});corrupt=1;cu(cudaMemcpy(table.p_->table.p+1,&corrupt,8,cudaMemcpyHostToDevice));
  reject([&]{prefix.require_complete(table);});table.p_->audit();
  report.passed=report.CUDA_executed=true;return report;
}
} // testing
} // namespace rl_qualified_session
