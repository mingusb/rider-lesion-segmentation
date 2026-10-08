// Generic source-model CUDA class-reference adapter; no training API.
#pragma once
#include <dlfcn.h>
#include <climits>
namespace native_class_reference {
constexpr const char* library_pin="462aa6331ecb178df8d10f16612c865ae70f571c248138e50c8a9eb4bc007dd4";
// Historical qualification pin above belongs to frozen research artifacts.
// Maintained consumers bind the exact caller-supplied library bytes instead.
inline std::string library_identity(const J& plan) {
 need(plan.at("native_library_path").is_string(),"native library path must be a string");
 const auto path=plan.at("native_library_path").get<std::string>();
 need(path.find('\0')==std::string::npos&&fs::path(path).is_absolute(),"native library path must be absolute");
 need(plan.at("native_library_sha256").is_string(),"native library SHA must be a string");
 const auto pin=plan.at("native_library_sha256").get<std::string>();
 need(pin.size()==64,"native library SHA must contain 64 lowercase hexadecimal digits");
 for(char c:pin)need((c>='0'&&c<='9')||(c>='a'&&c<='f'),"invalid native library SHA");
 return pin;
}
struct Api{
 using H=void*;void*library=nullptr;H matrix=nullptr,model=nullptr;unsigned D;
 const char*(*error)()=nullptr;int(*free_booster)(H)=nullptr;int(*free_matrix)(H)=nullptr;
 template<class T>T symbol(const char*name){dlerror();auto p=dlsym(library,name);auto e=dlerror();need(p&&!e,std::string("missing native symbol ")+name);return reinterpret_cast<T>(p);}
 void check(int status,const char*operation){if(status)throw std::runtime_error(std::string(operation)+": "+(error?error():"native error"));}
 Api(const fs::path&path,unsigned dims):D(dims){need(D>0,"empty response bank");library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);need(library,std::string("native dlopen failed: ")+(library?"":dlerror()));error=symbol<const char*(*)()>("XGBGetLastError");free_booster=symbol<int(*)(H)>("XGBoosterFree");free_matrix=symbol<int(*)(H)>("XGDMatrixFree");std::array<int,3>version{};symbol<void(*)(int*,int*,int*)>("XGBoostVersion")(&version[0],&version[1],&version[2]);need(version==std::array<int,3>{3,4,1},"native version differs");const char*info=nullptr;check(symbol<int(*)(const char**)>("XGBuildInfo")(&info),"build info");need(info&&J::parse(info).at("USE_CUDA")==true,"native library has no CUDA");}
 ~Api(){if(model&&free_booster)free_booster(model);if(matrix&&free_matrix)free_matrix(matrix);if(library)dlclose(library);}
 std::string array(const float*p,U rows,bool labels=false){return J{{"data",J::array({reinterpret_cast<std::uintptr_t>(p),true})},{"shape",labels?J::array({rows}):J::array({rows,U(D)})},{"strides",nullptr},{"typestr","<f4"},{"version",3},{"stream",1}}.dump();}
 void set(const char*key,const std::string&value){check(symbol<int(*)(H,const char*,const char*)>("XGBoosterSetParam")(model,key,value.c_str()),key);}
 void reset(bool){if(model){check(free_booster(model),"free booster");model=nullptr;}check(symbol<int(*)(const H*,U,H*)>("XGBoosterCreate")(nullptr,0,&model),"create booster");}
 void load(const fs::path&path){reset(false);check(symbol<int(*)(H,const char*)>("XGBoosterLoadModel")(model,path.c_str()),"load native meta model");set("device","cuda:0");}
 const float*classes(const float*x,U rows,unsigned outputs=1){const U*shape=nullptr;U dims=0;const float*p=nullptr;auto input=array(x,rows);auto options="{\"type\":0,\"training\":false,\"iteration_begin\":0,\"iteration_end\":0,\"strict_shape\":true,\"cache_id\":0,\"missing\":NaN}";check(symbol<int(*)(H,const char*,const char*,H,const U**,U*,const float**)>("XGBoosterPredictFromCudaArray")(model,input.c_str(),options,nullptr,&shape,&dims,&p),"native GPU direct classes");need(shape&&dims==2&&shape[0]==rows&&shape[1]==outputs&&p,"native meta prediction shape differs");cudaPointerAttributes a{};cu(cudaPointerGetAttributes(&a,p));need(a.type==cudaMemoryTypeDevice&&a.device==0,"native meta prediction used CPU fallback");gpu_sync();return p;}
};
}
