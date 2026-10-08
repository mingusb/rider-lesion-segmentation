#include "class_rank_xai_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
namespace rank_xai_cuda { namespace {
using Node=rank_xai_export::Node;using Gate=rank_xai_export::GateKind;
void need(bool v,const char*m){if(!v)throw std::runtime_error(m);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U product(U a,U b){need(!b||a<=UINT64_MAX/b,"xAI size overflow");return a*b;}
struct Budget {U limit,used=0,peak=0;};
template<class T>struct Buffer {
 T*p=nullptr;U n;Budget&b;
 Buffer(Budget&budget,U count):n(count),b(budget){U bytes=product(n,sizeof(T));need(bytes<=b.limit-b.used,"xAI device budget exceeded");cu(cudaMalloc(&p,bytes));b.used+=bytes;b.peak=std::max(b.peak,b.used);}
 ~Buffer(){if(p)cudaFree(p);b.used-=n*sizeof(T);}Buffer(const Buffer&)=delete;
 void set(const T*v){cu(cudaMemcpy(p,v,n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T>get(){std::vector<T>v(n);cu(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct Meta {U nodes,rows,root,allowed;unsigned offset[11],lo[10],hi[10];};
struct Output {int hard=-1,smooth=-1;U length=0;double values[7]{},derivatives[70]{};};
struct Hop {U node;int left;};
__device__ void fail(int*error,int code){atomicExch(error,code);}
__device__ bool raw_gate(Node n,const float*x){
 switch(n.gate){case Gate::constant_false:return false;case Gate::constant_true:return true;
 case Gate::numeric_less:case Gate::category_less:return x[n.feature]<__uint_as_float(n.raw_threshold_bits);
 default:return false;}
}
__global__ void evaluate(Meta m,const Node*nodes,const unsigned*cuts,const float*raw,const float*temps,
                         double*workspace,Hop*paths,Output*out,int*error){
 U row=U(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=m.rows)return;
 const float*x=raw+row*54;unsigned ranks[10]{};int a=0,b=0;U categories=0;
 for(int f=0;f<54;++f){if(!isfinite(x[f])){fail(error,1);return;}if(f>=10){if(x[f]!=0.f&&x[f]!=1.f){fail(error,2);return;}if(x[f]==1.f){categories|=1ull<<(f-10);if(f<14)++a;else ++b;}}}
 if(a!=1||b!=1||(categories&~m.allowed)){fail(error,3);return;}
 for(int f=0;f<10;++f){if(!isfinite(temps[f])||temps[f]<=0.f){fail(error,4);return;}
  for(unsigned j=m.offset[f];j<m.offset[f+1];++j)ranks[f]+=x[f]>=__uint_as_float(cuts[j]);
  if(ranks[f]<m.lo[f]||ranks[f]>m.hi[f]){fail(error,5);return;}}
 // Validate every equation and evaluate its seven soft membership scores and
 // ten continuous derivatives in topological order. No categorical derivative
 // is inferred from the discrete exactly-one input domain.
 double*w=workspace+row*m.nodes*77;
 for(U i=0;i<m.nodes;++i){Node n=nodes[i];if(n.id!=i){fail(error,6);return;}double*v=w+i*77;
  if(n.kind==2){if(n.gate!=Gate::leaf||n.label<0||n.label>=7){fail(error,7);return;}for(int k=0;k<77;++k)v[k]=0;v[n.label*11]=1;continue;}
  if(n.kind!=0||n.feature<0||n.feature>=54||n.left>=i||n.right>=i||n.gate==Gate::leaf||!isfinite(__uint_as_float(n.stored_cut_bits))){fail(error,8);return;}
  double g,dg=0;
  if(n.gate==Gate::numeric_less){if(n.feature>=10||!isfinite(__uint_as_float(n.raw_threshold_bits))){fail(error,9);return;}
   double z=(double(__uint_as_float(n.raw_threshold_bits))-double(x[n.feature]))/double(temps[n.feature]);
   double s=exp(-fabs(z));if(z>=0)g=1.0/(1.0+s);else g=s/(1.0+s);
   // Avoid cancellation when g rounds to one, and avoid premature exp underflow
   // before division by a tiny positive FP32 temperature.
   if(fabs(z)<700.)dg=(-s/((1.0+s)*(1.0+s)))/double(temps[n.feature]);
   else dg=-exp(-fabs(z)-2.0*log1p(s)-log(double(temps[n.feature])));
  }else if(n.gate==Gate::category_less){if(n.feature<10||!isfinite(__uint_as_float(n.raw_threshold_bits))){fail(error,10);return;}g=raw_gate(n,x)?1.:0.;}
  else if(n.gate==Gate::constant_false)g=0.;else if(n.gate==Gate::constant_true)g=1.;else{fail(error,11);return;}
  const double*l=w+n.left*77,*r=w+n.right*77;
  for(int c=0;c<7;++c){v[c*11]=g*l[c*11]+(1.0-g)*r[c*11];
   for(int f=0;f<10;++f){double d=g*l[c*11+1+f]+(1.0-g)*r[c*11+1+f];if(n.gate==Gate::numeric_less&&f==n.feature)d+=(l[c*11]-r[c*11])*dg;v[c*11+1+f]=d;}
   for(int f=0;f<11;++f)if(!isfinite(v[c*11+f])){fail(error,12);return;}}
 }
 Output answer;double total=0;const double*v=w+m.root*77;answer.smooth=0;
 for(int c=0;c<7;++c){answer.values[c]=v[c*11];total+=answer.values[c];if(answer.values[c]>answer.values[answer.smooth])answer.smooth=c;
  for(int f=0;f<10;++f)answer.derivatives[c*10+f]=v[c*11+1+f];}
 if(fabs(total-1.)>1e-8){fail(error,13);return;}
 U at=m.root;bool ended=false;
 for(U step=0;step<m.nodes;++step){Node n=nodes[at];if(n.kind==2){answer.hard=n.label;ended=true;break;}
  bool go_left=raw_gate(n,x);auto&hop=paths[row*m.nodes+answer.length++];hop.node=at;hop.left=int(go_left);at=go_left?n.left:n.right;}
 if(!ended){fail(error,14);return;}
 // Independent original representation: rank the raw inputs, then use each
 // original stored predicate, rather than the lowered raw equation gate.
 at=m.root;ended=false;
 for(U step=0;step<m.nodes;++step){Node n=nodes[at];if(n.kind==2){if(n.label!=answer.hard||step!=answer.length){fail(error,15);return;}ended=true;break;}
  float value=n.feature<10?float(ranks[n.feature]):x[n.feature];bool go_left=value<__uint_as_float(n.stored_cut_bits);
  if(step>=answer.length||paths[row*m.nodes+step].node!=at||paths[row*m.nodes+step].left!=int(go_left)){fail(error,17);return;}at=go_left?n.left:n.right;}
 if(!ended){fail(error,16);return;}out[row]=answer;
}
}
Result explain(const rank_xai_export::Model&model,const std::vector<std::array<float,54>>&rows,
               const std::array<float,10>&temperatures,Options options){
 need(!rows.empty()&&rows.size()<=options.maximum_rows&&!model.nodes.empty()&&model.nodes.size()<=options.maximum_nodes&&model.root<model.nodes.size(),"xAI input capacity invalid");
 need(options.maximum_device_bytes>0&&rows.size()<=U(INT_MAX)*128,"xAI options invalid");
 // Model is a public transport type: restore its stored representation and
 // require canonical lowering again before trusting caller-supplied raw gates.
 rank_regional_model_export::Model stored;stored.source_sha256=model.source_sha256;stored.rank_sha256=model.rank_sha256;stored.scope=model.scope;stored.rank_cut_bits=model.rank_cut_bits;stored.root=model.root;
 for(const auto&v:model.nodes){rank_regional_model_export::collect::dl::Node n;n.id=v.id;n.left=v.left;n.right=v.right;n.kind=v.kind;n.feature=v.feature;n.label=v.label;n.cut_bits=v.stored_cut_bits;stored.nodes.push_back(n);}
 auto validated=rank_xai_export::lower(stored);
 need(validated.nodes==model.nodes&&validated.runtime_sha256==model.runtime_sha256&&validated.runtime_bytes==model.runtime_bytes,"xAI lowered model metadata differs from bound runtime");
 Meta meta{};meta.nodes=model.nodes.size();meta.rows=rows.size();meta.root=model.root;meta.allowed=model.scope.allowed;
 std::vector<unsigned>cuts;for(int f=0;f<10;++f){need(cuts.size()+model.rank_cut_bits[f].size()<=UINT_MAX,"xAI rank table capacity");meta.offset[f]=cuts.size();cuts.insert(cuts.end(),model.rank_cut_bits[f].begin(),model.rank_cut_bits[f].end());meta.lo[f]=model.scope.lo[f];meta.hi[f]=model.scope.hi[f];}meta.offset[10]=cuts.size();if(cuts.empty())cuts.push_back(0);
 Budget budget{options.maximum_device_bytes};Buffer<Node>nodes(budget,meta.nodes);nodes.set(model.nodes.data());Buffer<unsigned>cutbuf(budget,cuts.size());cutbuf.set(cuts.data());
 Buffer<float>raw(budget,product(meta.rows,54));raw.set(rows.front().data());Buffer<float>temps(budget,10);temps.set(temperatures.data());
 Buffer<double>workspace(budget,product(product(meta.rows,meta.nodes),77));Buffer<Hop>paths(budget,product(meta.rows,meta.nodes));Buffer<Output>output(budget,meta.rows);Buffer<int>error(budget,1);cu(cudaMemset(error.p,0,sizeof(int)));
 cu(cudaMemset(paths.p,0,paths.n*sizeof(Hop)));
 evaluate<<<unsigned((meta.rows+127)/128),128>>>(meta,nodes.p,cutbuf.p,raw.p,temps.p,workspace.p,paths.p,output.p,error.p);cu(cudaGetLastError());cu(cudaDeviceSynchronize());
 int code=error.get()[0];need(code==0,("xAI CUDA validation code "+std::to_string(code)).c_str());
 auto computed=output.get();auto hops=paths.get();Result result;result.CUDA_executed=true;result.raw_and_rank_routes_equal=true;result.owned_device_peak_bytes=budget.peak;
 for(U i=0;i<meta.rows;++i){const auto&computed_row=computed[i];Answer answer;answer.hard_class=computed_row.hard;answer.smooth_class=computed_row.smooth;
  std::copy(computed_row.values,computed_row.values+7,answer.smooth_scores.begin());std::copy(computed_row.derivatives,computed_row.derivatives+70,answer.smooth_derivatives.begin());
  need(computed_row.length<=meta.nodes,"xAI trace length invalid");for(U j=0;j<computed_row.length;++j){auto hop=hops[i*meta.nodes+j];need(hop.node<meta.nodes,"xAI trace node invalid");const auto&node=model.nodes[hop.node];answer.path.push_back({hop.node,node.feature,node.raw_threshold_bits,node.stored_cut_bits,node.gate,bool(hop.left)});}result.rows.push_back(std::move(answer));
 }return result;
}
}
