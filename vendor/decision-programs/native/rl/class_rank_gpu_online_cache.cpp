#include "class_rank_gpu_online_cache.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace rank_gpu_online_cache { namespace {
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
std::uint64_t add(std::uint64_t a,std::uint64_t b){need(a<=UINT64_MAX-b,"online cache size addition overflow");return a+b;}
std::uint64_t mul(std::uint64_t a,std::uint64_t b){need(!a||b<=UINT64_MAX/a,"online cache size product overflow");return a*b;}
std::uint64_t table(std::uint64_t n){auto m=mul(n,2);std::uint64_t k=1;while(k<m){need(k<=UINT64_MAX/2,"online cache table overflow");k*=2;}return k;}
bool sha(const std::string&s){return s.size()==64&&std::all_of(s.begin(),s.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');});}
struct Hash {
 EVP_MD_CTX*p=EVP_MD_CTX_new();Hash(){need(p&&EVP_DigestInit_ex(p,EVP_sha256(),nullptr)==1,"SHA init failed");}
 ~Hash(){EVP_MD_CTX_free(p);}Hash(const Hash&)=delete;
 void bytes(const void*v,std::size_t n){need(EVP_DigestUpdate(p,v,n)==1,"SHA update failed");}
 template<class T>void u(T v){using U=std::make_unsigned_t<T>;auto q=static_cast<U>(v);unsigned char b[sizeof(T)];for(unsigned i=0;i<sizeof(T);++i)b[i]=static_cast<unsigned char>(q>>(8*i));bytes(b,sizeof b);}
 void s(const std::string&v){u<std::uint64_t>(v.size());bytes(v.data(),v.size());}
 std::string done(){unsigned char b[32];unsigned n=0;need(EVP_DigestFinal_ex(p,b,&n)==1&&n==32,"SHA final failed");static constexpr char h[]="0123456789abcdef";std::string s;for(auto c:b){s+=h[c>>4];s+=h[c&15];}return s;}
};
void prefix(Hash&h,const Prefix&p){for(auto x:p.genesis)h.u(x);h.u(p.generation);h.u(p.nodes);h.u(p.terms);h.s(p.head_sha256);}
void node(Hash&h,const dl::Node&n){h.u(n.id);h.u(n.left);h.u(n.right);h.u(n.first_term);h.u(n.term_count);h.u(n.label);h.u(n.kind);h.u(n.feature);h.u(n.cut_bits);h.u(n.threshold_bits);}
void term(Hash&h,const dl::Term&t){h.u(t.id);h.u(t.feature);h.u(t.weight_bits);}
void ref(Hash&h,const Ref&r){h.u(r.kind);h.u(r.id);}
}
void validate_options(const Options&o){need(o.maximum_nodes&&o.maximum_batch_nodes&&o.rows_per_chunk,"online cache zero node/chunk capacity");need(o.hash_bits<=64,"online cache hash bits exceed64");need(o.maximum_batch_nodes<=o.maximum_nodes,"online cache batch node limit exceeds arena");need(o.maximum_batch_terms<=o.maximum_terms,"online cache batch term limit exceeds arena");need(o.maximum_nodes<none&&o.maximum_terms<none,"online cache sentinel capacity");auto n=device_bytes(o);need(n<=o.maximum_device_bytes&&n<=SIZE_MAX,"online cache device byte limit");}
std::uint64_t device_bytes(const Options&o){
 auto n=mul(o.maximum_nodes,sizeof(dl::Node)+sizeof(Module));n=add(n,mul(o.maximum_terms,sizeof(dl::Term)));n=add(n,mul(table(o.maximum_nodes),8));
 n=add(n,mul(o.maximum_batch_nodes,sizeof(Draft)+sizeof(dl::Node)+2*sizeof(Module)+8));n=add(n,mul(o.maximum_batch_terms,2*sizeof(dl::Term)));n=add(n,mul(table(o.maximum_batch_nodes),8));return add(n,256);
}
std::string delta_digest(const std::vector<dl::Node>&ns,const std::vector<dl::Term>&ts){Hash h;h.s("online-cache-delta-1");h.u<std::uint64_t>(ns.size());h.u<std::uint64_t>(ts.size());for(auto&n:ns)node(h,n);for(auto&t:ts)term(h,t);return h.done();}
Prefix genesis(const Seed&s){need(sha(s.authority_sha256),"invalid authority SHA256");need(!s.nonce.empty()&&s.nonce.size()<=4096,"invalid unique arena nonce");Hash h;h.s("online-cache-genesis-1");h.s(s.authority_sha256);h.s(s.nonce);h.s(delta_digest(s.nodes,s.terms));auto d=h.done();Prefix p;for(unsigned j=0;j<4;++j){std::uint64_t w=0;for(unsigned k=0;k<16;++k){char c=d[j*16+k];w=(w<<4)|static_cast<unsigned>(c<='9'?c-'0':c-'a'+10);}p.genesis[j]=w;}p.nodes=s.nodes.size();p.terms=s.terms.size();Hash q;q.s("online-cache-head-1");q.s(d);q.u(p.nodes);q.u(p.terms);p.head_sha256=q.done();return p;}
std::string request_digest(const Request&r){Hash h;h.s("online-cache-request-1");prefix(h,r.expected);h.u<std::uint64_t>(r.nodes.size());h.u<std::uint64_t>(r.terms.size());for(auto&n:r.nodes){ref(h,n.left);ref(h,n.right);h.u(n.first_term);h.u(n.threshold_bits);h.u(n.term_count);h.u(n.cut_bits);h.u(n.kind);h.u(n.label);h.u(n.feature);}for(auto&t:r.terms)term(h,t);return h.done();}
std::string mapping_digest(const std::vector<Module>&m){Hash h;h.s("online-cache-mapping-1");h.u<std::uint64_t>(m.size());for(auto&x:m){h.u(x.id);h.u(x.expanded_nodes);h.u(x.expanded_terms);h.u(x.height);h.u(x.nodes_saturated);h.u(x.terms_saturated);}return h.done();}
Extension make_extension(const Prefix&b,std::uint64_t nn,std::uint64_t nt,const std::string&r,const std::string&d,const std::string&m){need(sha(b.head_sha256)&&sha(r)&&sha(d)&&sha(m),"invalid extension SHA256");need(nn>0&&b.generation<UINT64_MAX,"invalid empty/overflow extension");Extension e;e.before=b;e.after=b;e.after.generation++;e.after.nodes=add(b.nodes,nn);e.after.terms=add(b.terms,nt);e.request_sha256=r;e.delta_sha256=d;e.mapping_sha256=m;Hash h;h.s("online-cache-extension-1");prefix(h,b);h.u(e.after.generation);h.u(e.after.nodes);h.u(e.after.terms);h.s(r);h.s(d);h.s(m);e.receipt_sha256=h.done();Hash q;q.s("online-cache-head-1");q.s(b.head_sha256);q.s(e.receipt_sha256);e.after.head_sha256=q.done();return e;}
}
