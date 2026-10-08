#include "class_rank_xai_export.hpp"
#include "class_io.hpp"
#include <iostream>
#include <functional>
namespace x=rank_xai_export;namespace c=x::codec;namespace fs=std::filesystem;using J=dpnative::json;using U=std::uint64_t;
U assertions=0,rejections=0;
void check(bool v,const char*s){++assertions;if(!v)throw std::runtime_error(s);}
void reject(std::function<void()>f){++rejections;bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"expected rejection");}
std::uint32_t bits(float x){return std::bit_cast<std::uint32_t>(x);}
c::collect::dl::Node leaf(U id,int label){c::collect::dl::Node n;n.id=id;n.label=label;return n;}
c::collect::dl::Node axis(U id,U l,U r,int f,std::uint32_t cut){c::collect::dl::Node n;n.id=id;n.kind=0;n.label=-1;n.feature=f;n.left=l;n.right=r;n.cut_bits=cut;return n;}
c::Model base(){c::Model m;m.source_sha256=std::string(64,'a');m.scope.allowed=(U(1)<<44)-1;m.rank_cut_bits[0]={0xff7fffffu,0x80000001u,0u,1u,0x7f7fffffu};for(unsigned i=0;i<10;++i){m.scope.lo[i]=0;m.scope.hi[i]=m.rank_cut_bits[i].size();}m.rank_sha256=c::rank_digest(m.rank_cut_bits);m.nodes={leaf(0,0),leaf(1,6),axis(2,0,1,0,bits(1.f))};m.root=2;return m;}
void put(const fs::path&p,const std::string&s){std::ofstream f(p,std::ios::binary);f<<s;if(!f)throw std::runtime_error("fixture write");}
J get(const fs::path&p){return J::parse(dpnative::read_text(p));}
void metadata(){
 auto m=base();auto ir=x::lower(m);check(ir.nodes.size()==3&&ir.root==2,"basic shape");check(ir.runtime_sha256==dpnative::sha256(c::encode(m)),"runtime hash");check(ir.nodes[2].raw_threshold_bits==0xff7fffffu,"FLTMAX threshold word");
 const std::vector<std::pair<std::uint32_t,std::uint32_t>> cuts={{bits(1.f),0xff7fffffu},{bits(1.5f),0x80000001u},{bits(2.f),0x80000001u},{bits(2.25f),0u},{bits(3.f),0u},{bits(4.f),1u},{bits(5.f),0x7f7fffffu},{1u,0xff7fffffu}};
 for(auto [stored,raw]:cuts){m.nodes[2].cut_bits=stored;auto a=x::lower(m);check(a.nodes[2].gate==x::GateKind::numeric_less,"numeric gate");check(a.nodes[2].raw_threshold_bits==raw&&a.nodes[2].stored_cut_bits==stored,"exact threshold transport");}
 for(auto w:{0u,0x80000000u,bits(-1.f),0xff7fffffu}){m.nodes[2].cut_bits=w;check(x::lower(m).nodes[2].gate==x::GateKind::constant_false,"false rank gate");}
 for(auto w:{bits(5.01f),bits(6.f),0x7f7fffffu}){m.nodes[2].cut_bits=w;check(x::lower(m).nodes[2].gate==x::GateKind::constant_true,"true rank gate");}
 m.nodes[2].feature=9;m.nodes[2].cut_bits=1u;check(x::lower(m).nodes[2].gate==x::GateKind::constant_true,"empty rank column");
 for(int feature:{10,13,14,53})for(auto w:{1u,bits(.5f),bits(1.f)}){m.nodes[2].feature=feature;m.nodes[2].cut_bits=w;auto a=x::lower(m);check(a.nodes[2].gate==x::GateKind::category_less&&a.nodes[2].raw_threshold_bits==w,"exact onehot gate");}
 for(auto w:{0u,0x80000000u,bits(-1.f)}){m.nodes[2].cut_bits=w;check(x::lower(m).nodes[2].gate==x::GateKind::constant_false,"false category");}
 m.nodes[2].cut_bits=bits(1.01f);check(x::lower(m).nodes[2].gate==x::GateKind::constant_true,"true category");
 m=base();m.nodes={leaf(0,4)};m.root=0;check(x::lower(m).nodes[0].gate==x::GateKind::leaf,"constant model");
 auto bad=base();bad.nodes[2].kind=1;bad.nodes[2].feature=-1;bad.nodes[2].cut_bits=0;bad.nodes[2].term_count=1;bad.terms={{0,0,bits(1.f)}};reject([&]{x::lower(bad);});
 for(auto mutate:std::vector<std::function<void(c::Model&)>>{
 [](auto&m){m.nodes[2].left=2;},[](auto&m){m.nodes[0].label=7;},[](auto&m){m.nodes[2].feature=54;},
 [](auto&m){m.nodes[2].cut_bits=0x7f800000u;},[](auto&m){m.nodes[2].cut_bits=0x7fc00000u;},
 [](auto&m){m.scope.allowed=1;},[](auto&m){m.rank_cut_bits[0][2]=0x80000000u;m.rank_sha256=c::rank_digest(m.rank_cut_bits);},
 [](auto&m){m.nodes.push_back(leaf(3,3));m.root=3;},[](auto&m){m.scope.hi[0]=6;},
 [](auto&m){m.rank_sha256=std::string(64,'0');}}){auto b=base();mutate(b);reject([&]{x::lower(b);});}
}
void files(const fs::path&root){check(!fs::exists(root),"fresh fixture root");fs::create_directories(root);auto m=base();auto raw=c::encode(m);auto p=root/"model.bin";put(p,raw);auto digest=dpnative::sha256(raw);
 auto r=x::write(p,digest,root/"hard");check(r.structural_roundtrip&&r.runtime_bytes==raw.size(),"hard write");x::verify(root/"hard");++assertions;
 auto smooth=x::write(p,digest,root/"smooth",x::Options{true});check(smooth.structural_roundtrip,"smooth write");auto sj=get(root/"smooth/equations.json");check(sj["smooth"]["changes_classifier"]==true&&sj["smooth"]["source_equivalence_claim"]==false,"surrogate caveat");check(sj["scientist_capabilities"]["fuzzy_gradients"]["implemented"]==false,"gradient limitation");check(sj["shared_equations"].size()==3,"no expansion");
 reject([&]{x::write(p,digest,root/"hard");});reject([&]{x::write(p,std::string(64,'0'),root/"bad-hash");});check(!fs::exists(root/"bad-hash"),"reject before output");reject([&]{x::write(p,digest,root/"short-cap",x::Options{false,1});});
 auto eq=get(root/"hard/equations.json");auto oldeq=eq;eq["shared_equations"][2]["gate"]["raw_threshold_bits"]=0u;put(root/"hard/equations.json",eq.dump(2)+"\n");reject([&]{x::verify(root/"hard");});
 // Even recomputing the changed file hash cannot defeat structural regeneration.
 auto receipt=get(root/"hard/result.json");receipt["equations_sha256"]=dpnative::sha256(dpnative::read_text(root/"hard/equations.json"));receipt["export_payload_bytes"]=fs::file_size(root/"hard/equations.json")+fs::file_size(root/"hard/equations.txt")+fs::file_size(root/"hard/inspect.json");put(root/"hard/result.json",receipt.dump(2)+"\n");reject([&]{x::verify(root/"hard");});
 put(root/"hard/equations.json",oldeq.dump(2)+"\n");receipt["equations_sha256"]=dpnative::sha256(dpnative::read_text(root/"hard/equations.json"));receipt["export_payload_bytes"]=fs::file_size(root/"hard/equations.json")+fs::file_size(root/"hard/equations.txt")+fs::file_size(root/"hard/inspect.json");put(root/"hard/result.json",receipt.dump(2)+"\n");x::verify(root/"hard");++assertions;
 // A shared binary doubling chain has exponential unfolded size, with 128 equations.
 auto huge=base();huge.nodes={leaf(0,2)};for(U i=1;i<128;++i)huge.nodes.push_back(axis(i,i-1,i-1,0,bits(1.f)));huge.root=127;put(root/"huge.bin",c::encode(huge));auto hr=x::write(root/"huge.bin",dpnative::sha256(c::encode(huge)),root/"huge");check(hr.nodes==128,"shared chain count");auto hi=get(root/"huge/inspect.json");check(hi["unfolded_nodes_decimal"]=="340282366920938463463374607431768211455","arbitrary precision unfolded size");check(hi["depth_decisions"]==127&&hi["nodes_with_multiple_incoming_edges"]==127,"shared depth/count");
 auto regional=base();regional.scope.lo[0]=1;regional.scope.hi[0]=3;regional.scope.allowed=U(1)|(U(1)<<4);put(root/"regional.bin",c::encode(regional));x::write(root/"regional.bin",dpnative::sha256(c::encode(regional)),root/"regional");check(get(root/"regional/inspect.json")["full_forest_domain_scope"]==false,"regional scope preserved");
 auto extreme=base();extreme.scope.lo[0]=1;put(root/"extreme.bin",c::encode(extreme));x::write(root/"extreme.bin",dpnative::sha256(c::encode(extreme)),root/"extreme");check(get(root/"extreme/inspect.json")["full_forest_domain_scope"]==true,"finite domain excludes impossible rank below -FLTMAX");
 auto cp=base();cp.nodes={leaf(0,3)};cp.root=0;put(root/"constant.bin",c::encode(cp));x::write(root/"constant.bin",dpnative::sha256(c::encode(cp)),root/"constant");check(get(root/"constant/inspect.json")["depth_decisions"]==0,"constant depth");
 // Rehashed negative/overflow metadata must not wrap during reconstruction.
 for(int mode=0;mode<3;++mode){auto tampered=oldeq;if(mode==0)tampered["codec_payload"]["nodes"][2]["cut_bits"]=-1;else if(mode==1)tampered["codec_payload"]["nodes"][2]["feature"]=UINT64_MAX;else tampered["codec_payload"]["rank_cut_bits"][0][0]=U(UINT32_MAX)+1;put(root/"hard/equations.json",tampered.dump(2)+"\n");receipt["equations_sha256"]=dpnative::sha256(dpnative::read_text(root/"hard/equations.json"));receipt["export_payload_bytes"]=fs::file_size(root/"hard/equations.json")+fs::file_size(root/"hard/equations.txt")+fs::file_size(root/"hard/inspect.json");put(root/"hard/result.json",receipt.dump(2)+"\n");reject([&]{x::verify(root/"hard");});}
}
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("checks NEW_FIXTURE_DIR");metadata();files(argv[1]);std::cout<<J{{"passed",true},{"assertions",assertions},{"rejections",rejections},{"CPU_predictions_evaluated",false},{"GPU_executed",false}}.dump(2)<<'\n';return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

