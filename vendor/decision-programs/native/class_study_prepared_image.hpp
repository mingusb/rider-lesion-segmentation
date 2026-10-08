#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Host transport and metadata only. Continuous image sampling and all twenty
// feature calculations remain in the CUDA preparation translation unit.
namespace class_study::prepared_image {
using J=nlohmann::json;
using U=std::uint64_t;
namespace fs=std::filesystem;
inline void require(bool ok,const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
inline J json_file(const fs::path& path) {
  std::ifstream input(path); require(bool(input),"cannot read "+path.string());
  J value; input>>value; return value;
}
inline fs::path relative_file(const fs::path& descriptor,const std::string& name) {
  fs::path path(name);
  return fs::absolute(path.is_absolute()?path:descriptor.parent_path()/path).lexically_normal();
}
struct Array {
  std::array<int,4> shape{};
  std::vector<float> values;
};
inline Array read_npy(std::istream& input,U file_bytes,U maximum_channels=1) {
  require(std::endian::native==std::endian::little,"prepared arrays require a little-endian host");
  std::array<unsigned char,8> prefix{};
  input.read(reinterpret_cast<char*>(prefix.data()),prefix.size());
  require(bool(input)&&prefix[0]==0x93&&std::string(reinterpret_cast<char*>(prefix.data()+1),5)=="NUMPY",
          "invalid NPY magic");
  require((prefix[6]==1||prefix[6]==2)&&prefix[7]==0,"only NPY 1.0 and 2.0 are supported");
  unsigned length_bytes=prefix[6]==1?2:4;
  std::array<unsigned char,4> length{};
  input.read(reinterpret_cast<char*>(length.data()),length_bytes);
  require(bool(input),"truncated NPY header length");
  U header_bytes=0;
  for(unsigned i=0;i<length_bytes;++i)header_bytes|=U(length[i])<<(8*i);
  require(header_bytes>0&&header_bytes<=65536&&file_bytes>=8+length_bytes+header_bytes,
          "invalid NPY header extent");
  std::string header(std::size_t(header_bytes),'\0');
  input.read(header.data(),std::streamsize(header_bytes));
  require(bool(input),"truncated NPY header");
  std::smatch match;
  require(std::regex_search(header,match,std::regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]")),
          "NPY dtype missing");
  require(match[1].str()=="<f4"||match[1].str()=="=f4","prepared image must be little-endian float32");
  require(std::regex_search(header,match,std::regex("['\"]fortran_order['\"]\\s*:\\s*(True|False)")),
          "NPY storage order missing");
  require(match[1].str()=="False","prepared image must be C-contiguous CZYX");
  require(std::regex_search(header,match,std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)")),
          "NPY shape missing");
  std::string shape=match[1].str(); std::size_t at=0; Array result;
  U count=1;
  for(unsigned axis=0;axis<4;++axis) {
    while(at<shape.size()&&std::isspace(static_cast<unsigned char>(shape[at])))++at;
    int dim=0;auto parsed=std::from_chars(shape.data()+at,shape.data()+shape.size(),dim);
    require(parsed.ec==std::errc{}&&parsed.ptr!=shape.data()+at&&dim>0,"invalid NPY CZYX extent");
    at=std::size_t(parsed.ptr-shape.data());
    while(at<shape.size()&&std::isspace(static_cast<unsigned char>(shape[at])))++at;
    if(axis<3){require(at<shape.size()&&shape[at]==',',"NPY image needs four CZYX axes");++at;}
    result.shape[axis]=dim;
    require(count<=std::numeric_limits<U>::max()/U(dim),"NPY extent overflow");count*=U(dim);
  }
  if(at<shape.size()&&shape[at]==',')++at;
  while(at<shape.size()&&std::isspace(static_cast<unsigned char>(shape[at])))++at;
  require(at==shape.size()&&maximum_channels>0&&U(result.shape[0])<=maximum_channels,"NPY image channel count exceeds its declaration or needs exactly four CZYX axes");
  require(count<=std::numeric_limits<std::size_t>::max()/sizeof(float)&&
          count<=U(std::numeric_limits<std::streamsize>::max())/sizeof(float),"NPY array too large");
  U offset=8+length_bytes+header_bytes;
  require(count<=std::numeric_limits<U>::max()/sizeof(float)&&
          file_bytes-offset==count*sizeof(float),"NPY payload length does not match CZYX float32 shape");
  result.values.resize(std::size_t(count));
  input.read(reinterpret_cast<char*>(result.values.data()),std::streamsize(count*sizeof(float)));
  require(bool(input),"truncated NPY image payload");return result;
}
inline Array read_npy(const fs::path& path,U maximum_channels=1) {
  std::ifstream input(path,std::ios::binary);require(bool(input),"cannot read "+path.string());
  return read_npy(input,fs::file_size(path),maximum_channels);
}
template<std::size_t N> inline std::array<double,N> numbers(const J& value,const std::string& name) {
  require(value.is_array()&&value.size()==N,"invalid "+name);
  std::array<double,N> result{};
  for(std::size_t i=0;i<N;++i){result[i]=value.at(i).get<double>();require(std::isfinite(result[i]),"nonfinite "+name);}
  return result;
}
inline std::array<int,3> permutation(const J& value) {
  require(value.is_array()&&value.size()==3,"transpose_forward needs three axes");
  std::array<int,3> result{};std::array<bool,3> seen{};
  for(unsigned i=0;i<3;++i){int a=value.at(i).get<int>();require(a>=0&&a<3&&!seen[a],"invalid transpose_forward");seen[a]=true;result[i]=a;}
  return result;
}
inline void orthonormal(const std::array<double,9>& direction) {
  for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
    double dot=0;for(unsigned r=0;r<3;++r)dot+=direction[r*3+i]*direction[r*3+j];
    require(std::abs(dot-(i==j?1.:0.))<1e-6,"direction matrix must be orthonormal");
  }
}
// Each row maps an output XYZ index to one saved prepared-array axis. The
// source image starts as ZYX, is transposed, cropped, then resized. The actual
// integer resize shapes carry the half-pixel center convention and rounding.
struct Geometry {
  int nz=0,ny=0,nx=0,n=0;
  double map[3][4]{};
  double cropped_map[3][4]{};
  int source_shape[3]{},cropped_shape[3]{},crop_start[3]{};
  float outside=0.f; // Filled from the declared CTNormalization air transform.
};
inline J configuration(const J& plans,const std::string& name,unsigned depth=0){
  require(depth<16,"cyclic or excessive plans configuration inheritance");
  J node=plans.at("configurations").at(name);
  if(node.contains("inherits_from")){J inherited=configuration(plans,node.at("inherits_from").get<std::string>(),depth+1);inherited.update(node);return inherited;}
  return node;
}
inline float normalized_air(const J& plans,const std::string& name){
  J setting=configuration(plans,name);
  require(setting.at("normalization_schemes")==J::array({"CTNormalization"})&&setting.at("use_mask_for_norm")==J::array({false}),
          "prepared images require declared single-channel CTNormalization without mask normalization");
  const auto& channels=plans.at("foreground_intensity_properties_per_channel");
  require(channels.is_object()&&channels.size()==1&&channels.contains("0"),"normalization needs one channel's intensity properties");
  const auto& p=channels.at("0");
  double mean=p.at("mean").get<double>(),stddev=p.at("std").get<double>(),lo=p.at("percentile_00_5").get<double>(),hi=p.at("percentile_99_5").get<double>();
  require(std::isfinite(mean)&&std::isfinite(stddev)&&std::isfinite(lo)&&std::isfinite(hi)&&stddev>0&&lo<=hi,"invalid CTNormalization intensity properties");
  // The released routine clips and normalizes an array in float32, in place.
  float result=std::clamp(-1000.f,float(lo),float(hi));result-=float(mean);result/=std::max(float(stddev),1e-8f);
  require(std::isfinite(result),"nonfinite normalized original air");return result;
}
inline Geometry geometry(const J& exported,const J& metadata,
                         const std::array<int,4>& shape,const std::array<int,3>& transpose,
                         int n,double step) {
  const auto& properties=metadata.at("properties");const auto& crop=exported.at("cropGrid");
  auto source_spacing=numbers<3>(exported.at("spacingXYZ"),"source spacing");
  auto source_origin=numbers<3>(exported.at("originLPS"),"source origin");
  auto source_direction=numbers<9>(exported.at("directionLPS"),"source direction");
  auto crop_spacing=numbers<3>(crop.at("spacingXYZ"),"crop spacing");
  auto crop_origin=numbers<3>(crop.at("originLPS"),"crop origin");
  auto crop_direction=numbers<9>(crop.at("directionLPS"),"crop direction");
  auto sizes=numbers<3>(exported.at("sizeXYZ"),"source sizes");
  auto crop_sizes=numbers<3>(crop.at("sizeXYZ"),"crop sizes");
  auto property_spacing=numbers<3>(properties.at("spacing"),"property spacing");
  auto target_spacing=numbers<3>(metadata.at("target_spacing_ZYX"),"target spacing");
  auto before=numbers<3>(properties.at("shape_before_cropping"),"shape before cropping");
  auto after=numbers<3>(properties.at("shape_after_cropping_and_before_resampling"),"cropped shape");
  orthonormal(source_direction);orthonormal(crop_direction);
  if(properties.contains("sitk_stuff")){
    const auto& sitk=properties.at("sitk_stuff");
    auto spacing=numbers<3>(sitk.at("spacing"),"metadata source spacing");
    auto origin=numbers<3>(sitk.at("origin"),"metadata source origin");
    auto direction=numbers<9>(sitk.at("direction"),"metadata source direction");
    for(unsigned i=0;i<3;++i){require(std::abs(spacing[i]-source_spacing[i])<1e-7,"metadata/export spacing mismatch");require(std::abs(origin[i]-source_origin[i])<1e-5,"metadata/export origin mismatch");}
    for(unsigned i=0;i<9;++i)require(std::abs(direction[i]-source_direction[i])<1e-7,"metadata/export direction mismatch");
  }
  require(properties.at("bbox_used_for_cropping").is_array()&&properties.at("bbox_used_for_cropping").size()==3,"invalid crop bbox");
  Geometry g{shape[1],shape[2],shape[3],n};
  for(unsigned i=0;i<3;++i) {
    require(source_spacing[i]>0&&sizes[i]>=1&&sizes[i]<=std::numeric_limits<int>::max()&&sizes[i]==std::floor(sizes[i]),"invalid source geometry");
    require(crop_sizes[i]==n&&std::abs(crop_spacing[i]-step)<1e-8,"export/reference crop grid mismatch");
    require(std::abs(property_spacing[i]-source_spacing[2-i])<1e-7,"metadata original ZYX spacing mismatch");
  }
  for(unsigned a=0;a<3;++a) {
    unsigned source_xyz=2-unsigned(transpose[a]);
    auto bbox=numbers<2>(properties.at("bbox_used_for_cropping").at(a),"crop bbox axis");
    require(before[a]==sizes[source_xyz]&&bbox[0]>=0&&bbox[1]<=before[a]&&bbox[1]>bbox[0]&&
            bbox[0]==std::floor(bbox[0])&&bbox[1]==std::floor(bbox[1])&&after[a]==bbox[1]-bbox[0],
            "transposed source/crop extent mismatch");
    require(target_spacing[a]>0&&std::abs(std::nearbyint(after[a]*source_spacing[source_xyz]/target_spacing[a])-shape[a+1])<0.01,
            "saved image shape does not match official spacing/rounded shape");
    double factor=shape[a+1]/after[a];
    double origin=0;
    for(unsigned r=0;r<3;++r)origin+=source_direction[r*3+source_xyz]*(crop_origin[r]-source_origin[r]);
    g.source_shape[a]=int(before[a]);g.cropped_shape[a]=int(after[a]);g.crop_start[a]=int(bbox[0]);
    g.cropped_map[a][3]=origin/source_spacing[source_xyz]-bbox[0];
    g.map[a][3]=(g.cropped_map[a][3]+0.5)*factor-0.5;
    for(unsigned out=0;out<3;++out) {
      double dot=0;for(unsigned r=0;r<3;++r)dot+=source_direction[r*3+source_xyz]*crop_direction[r*3+out];
      g.cropped_map[a][out]=dot*crop_spacing[out]/source_spacing[source_xyz];
      g.map[a][out]=g.cropped_map[a][out]*factor;
    }
  }
  return g;
}
struct ChannelGeometry {
  int nz=0,ny=0,nx=0,n=0,channels=0;
  double map[3][4]{}; // Output XYZ index -> declared physical volume ZYX.
};
inline ChannelGeometry channel_geometry(const J& grid,const J& output_grid,
                                        const std::array<int,4>& shape,int n,double step){
  auto origin=numbers<3>(grid.at("originLPS"),"channel origin"),spacing=numbers<3>(grid.at("spacingXYZ"),"channel spacing"),sizes=numbers<3>(grid.at("sizeXYZ"),"channel sizes");
  auto direction=numbers<9>(grid.at("directionLPS"),"channel direction");
  auto output_origin=numbers<3>(output_grid.at("originLPS"),"output origin"),output_spacing=numbers<3>(output_grid.at("spacingXYZ"),"output spacing"),output_sizes=numbers<3>(output_grid.at("sizeXYZ"),"output sizes");
  auto output_direction=numbers<9>(output_grid.at("directionLPS"),"output direction");orthonormal(direction);orthonormal(output_direction);
  ChannelGeometry g{shape[1],shape[2],shape[3],n,shape[0]};
  require(shape[0]>0&&shape[1]>0&&shape[2]>0&&shape[3]>0,"invalid channel CZYX shape");
  for(unsigned xyz=0;xyz<3;++xyz){require(spacing[xyz]>0&&sizes[xyz]==shape[3-xyz],"channel shape/physical grid mismatch");require(output_sizes[xyz]==n&&std::abs(output_spacing[xyz]-step)<1e-8,"channel output/reference physical grid mismatch");unsigned a=2-xyz;
    for(unsigned r=0;r<3;++r)g.map[a][3]+=direction[r*3+xyz]*(output_origin[r]-origin[r])/spacing[xyz];
    for(unsigned out=0;out<3;++out){double dot=0;for(unsigned r=0;r<3;++r)dot+=direction[r*3+xyz]*output_direction[r*3+out];g.map[a][out]=dot*output_spacing[out]/spacing[xyz];}
  }
  for(const auto& row:g.map)for(double value:row)require(std::isfinite(value),"nonfinite channel physical affine");
  return g;
}
} // namespace class_study::prepared_image
