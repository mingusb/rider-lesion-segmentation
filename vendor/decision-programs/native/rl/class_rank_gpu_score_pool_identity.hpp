#pragma once
#include "class_rank_gpu_score_diagram.hpp"
#include "class_io.hpp"
namespace rank_gpu_score_identity {
// Canonical byte identity only: no geometry, routing, score calculation or
// semantic equivalence. Root order and every semantic FP32 terminal word count.
// This does not claim globally minimal/cross-class-interned graph structure.
class Hash {
 EVP_MD_CTX*context_=nullptr;
 static void need(bool b){if(!b)throw std::runtime_error("score pool streaming SHA256 failed");}
public:
 Hash(){context_=EVP_MD_CTX_new();need(context_!=nullptr);if(!EVP_DigestInit_ex(context_,EVP_sha256(),nullptr)){EVP_MD_CTX_free(context_);context_=nullptr;need(false);}}
 ~Hash(){if(context_)EVP_MD_CTX_free(context_);}Hash(const Hash&)=delete;Hash&operator=(const Hash&)=delete;
 void bytes(const void*p,std::size_t n){need(EVP_DigestUpdate(context_,p,n)==1);}
 void word(std::uint64_t x,unsigned n){unsigned char b[8];for(unsigned k=0;k<n;++k)b[k]=static_cast<unsigned char>(x>>(8*k));bytes(b,n);}
 void text(const std::string&s){word(s.size(),8);bytes(s.data(),s.size());}
 std::string finish(){std::array<unsigned char,32>out{};unsigned n=0;need(EVP_DigestFinal_ex(context_,out.data(),&n)==1&&n==out.size());std::string s;static constexpr char hex[]="0123456789abcdef";s.reserve(64);for(unsigned char b:out){s+=hex[b>>4];s+=hex[b&15];}return s;}
};
inline std::string digest(const rank_gpu_score_diagram::Result&s){Hash h;h.text("seven-score-pool-canonical-words-v1");h.text(s.source_binding);h.text(s.domain_binding);for(auto x:s.domain.lo)h.word(static_cast<std::uint32_t>(x),4);for(auto x:s.domain.hi)h.word(static_cast<std::uint32_t>(x),4);h.word(s.domain.allowed,8);for(auto x:s.roots)h.word(x,8);h.word(s.nodes.size(),8);h.word(s.arcs.size(),8);for(const auto&n:s.nodes){h.word(n.kind,4);h.word(n.dimension,4);h.word(n.score_bits,4);h.word(n.reserved,4);h.word(n.first_arc,8);h.word(n.arc_count,8);}for(const auto&a:s.arcs){h.word(static_cast<std::uint32_t>(a.lo),4);h.word(static_cast<std::uint32_t>(a.hi),4);h.word(a.allowed,8);h.word(a.child,8);}return h.finish();}
}
