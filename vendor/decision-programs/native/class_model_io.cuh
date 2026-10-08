#pragma once
// Generic host byte/metadata transport and owned CUDA array utility.
#include "class_io.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <utility>
using namespace dpnative;
using u32=uint32_t; using u64=uint64_t;
static u64 owned=0,peak_owned=0;
void ck(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void done(){ck(cudaGetLastError(),"launch");ck(cudaDeviceSynchronize(),"synchronize");}
template<class T> struct Dev {
  T* p=nullptr; size_t n=0;
  Dev()=default; explicit Dev(size_t count):n(count){
    if(n>SIZE_MAX/sizeof(T))throw std::runtime_error("allocation overflow");
    if(n){ck(cudaMalloc(&p,n*sizeof(T)),"allocate");owned+=n*sizeof(T);peak_owned=std::max(peak_owned,owned);}}
  ~Dev(){if(p){cudaFree(p);owned-=n*sizeof(T);}}
  Dev(const Dev&)=delete;Dev& operator=(const Dev&)=delete;
  Dev(Dev&& a)noexcept:p(std::exchange(a.p,nullptr)),n(std::exchange(a.n,0)){}
  Dev& operator=(Dev&& a)noexcept{if(this!=&a){if(p){cudaFree(p);owned-=n*sizeof(T);}p=std::exchange(a.p,nullptr);n=std::exchange(a.n,0);}return *this;}
  void zero(){if(n)ck(cudaMemset(p,0,n*sizeof(T)),"zero");}
  void put(const std::vector<T>& v){if(v.size()!=n)throw std::runtime_error("upload shape");if(n)ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice),"upload");}
  std::vector<T> get()const{std::vector<T>v(n);if(n)ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"download");return v;}
  T at(size_t i)const{if(i>=n)throw std::runtime_error("metadata index");T v;ck(cudaMemcpy(&v,p+i,sizeof(T),cudaMemcpyDeviceToHost),"metadata");return v;}
};
int blocks(u64 n){return int(std::min<u64>((n+255)/256,65535));}
void clean_output(const fs::path&p){if(fs::exists(p))throw std::runtime_error("output already exists");fs::create_directories(p);}

using J=nlohmann::json;using U=std::uint64_t;
