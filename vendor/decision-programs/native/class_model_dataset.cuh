#pragma once
#include "class_model_mnist_adapter.cuh"
#include "class_model_dataset_contract.hpp"

struct ModelDataset {
 Dev<float> x; Dev<u32> labels;
 u64 rows=0,stride=0,fit_rows=0,valid_rows=0;
 u32 F=0,K=0;
 J binding;
};
__global__ void class_labels_from_exact_float(const float*input,u32*out,u64 rows,u32*bad){
 for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=u64(blockDim.x)*gridDim.x){float v=input[i];if(!isfinite(v)||v<0||v>4294967295.0||v!=floorf(v)){atomicOr(bad,1u);continue;}out[i]=u32(v);}
}
#include "class_model_mnist_dataset.cuh"
ModelDataset load_model_dataset(const J&p,const fs::path&out,bool fit_only=false){
 ModelDataset d;
 if(p.contains("dataset")){
  auto b=p.at("dataset");auto shape=class_model_contract::dense(b,fit_only);
  d.F=shape.features;d.K=shape.classes;d.rows=shape.rows;d.stride=shape.stride;d.fit_rows=shape.fit_rows;d.valid_rows=shape.valid_rows;
  fs::path xp=b.at("values_path").get<std::string>(),yp=b.at("labels_path").get<std::string>();Cpath(xp);Cpath(yp);
  auto xb=mn_read(xp),yb=mn_read(yp);need(sha256(xb)==b.at("values_sha256").get<std::string>()&&sha256(yb)==b.at("labels_sha256").get<std::string>(),"dense data pins differ");
  need(d.rows<=UINT64_MAX/d.stride&&d.rows*d.stride<=SIZE_MAX/4&&xb.size()==d.rows*d.stride*4&&d.rows<=SIZE_MAX/4&&yb.size()==d.rows*4,"dense byte extents");
  d.x=Dev<float>(d.rows*d.stride);d.labels=Dev<u32>(d.rows);cu(cudaMemcpy(d.x.p,xb.data(),xb.size(),cudaMemcpyHostToDevice));cu(cudaMemcpy(d.labels.p,yb.data(),yb.size(),cudaMemcpyHostToDevice));d.binding=b;d.binding["input_dtype"]="little-endian FP32";d.binding["label_dtype"]="little-endian uint32";d.binding["roles"]=fit_only?"All declared rows FIT; no VALID input": "first declared FIT_rows FIT; remaining declared VALID_rows VALID";d.binding["FIT_only"]=fit_only;return d;
 }
 need(!fit_only,"FIT-only producer requires a separate dense FIT file; IDX evaluator fallback refused");
 return load_idx_model_dataset(p,out);
}
void recheck_model_dataset(const ModelDataset&d){
 auto checkpin=[&](const char*path,const char*pin){need(sha256(mn_read(d.binding.at(path).get<std::string>()))==d.binding.at(pin).get<std::string>(),"dataset changed during evaluation");};
 if(d.binding.at("format")=="dense-fp32-u32-class-labels-1"){checkpin("values_path","values_sha256");checkpin("labels_path","labels_sha256");}
 else{checkpin("images_path","images_sha256");checkpin("labels_path","labels_sha256");}
}
