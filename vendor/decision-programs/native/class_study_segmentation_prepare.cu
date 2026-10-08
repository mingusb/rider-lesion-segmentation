#include "class_study_dicom.hpp"
#include "class_study_prepared_image.hpp"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

// DICOM decoding is host I/O. Resampling, feature construction, sampling and
// reference-volume accounting run on CUDA. Reference masks never define the
// seed, crop or features. All geometry and features are fixed before training.
namespace {
using J=nlohmann::json; namespace fs=std::filesystem;
using U=std::uint64_t;
void need(bool ok,const std::string&s){if(!ok)throw std::runtime_error(s);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
  T*p=nullptr; std::size_t n;
  explicit Device(std::size_t count):n(count){ck(cudaMalloc(&p,n*sizeof(T)));}
  ~Device(){if(p)cudaFree(p);}
  Device(const Device&)=delete; Device&operator=(const Device&)=delete;
  void upload(const T*h){ck(cudaMemcpy(p,h,n*sizeof(T),cudaMemcpyHostToDevice));}
  void download(T*h){ck(cudaMemcpy(h,p,n*sizeof(T),cudaMemcpyDeviceToHost));}
};
struct Geometry {
  int nx,ny,nz,n; double sx,sy,sz,cx,cy,cz,step;
  const double* zpositions=nullptr;
  int labelNz=0; const double* labelPositions=nullptr; double labelSpacing=0;
};
__device__ U raw_index(int x,int y,int z,Geometry g){return (U(z)*g.ny+y)*g.nx+x;}
__device__ bool inside(int x,int y,int z,Geometry g){return x>=0&&x<g.nx&&y>=0&&y<g.ny&&z>=0&&z<g.nz;}
__device__ float point(const std::int16_t*raw,const float*slope,const float*intercept,int x,int y,int z,Geometry g){
  return inside(x,y,z,g)?float(raw[raw_index(x,y,z,g)])*slope[z]+intercept[z]:-1000.0f;
}
__device__ float sample(const std::int16_t*raw,const float*slope,const float*intercept,double x,double y,double z,Geometry g){
  int ix=int(floor(x)),iy=int(floor(y)),iz=int(floor(z));
  float fx=float(x-ix),fy=float(y-iy),fz=float(z-iz),v=0;
  for(int k=0;k<2;k++)for(int j=0;j<2;j++)for(int i=0;i<2;i++)
    v+=point(raw,slope,intercept,ix+i,iy+j,iz+k,g)*(i?fx:1-fx)*(j?fy:1-fy)*(k?fz:1-fz);
  return v;
}
__device__ double slice_coordinate(double distance,Geometry g){
  if(!g.zpositions)return distance/g.sz;
  const double* p=g.zpositions;
  if(distance<=p[0])return (distance-p[0])/(p[1]-p[0]);
  if(distance>=p[g.nz-1])return g.nz-1+(distance-p[g.nz-1])/(p[g.nz-1]-p[g.nz-2]);
  int lo=0,hi=g.nz-1;while(hi-lo>1){int m=(lo+hi)/2;if(p[m]<=distance)lo=m;else hi=m;}
  return lo+(distance-p[lo])/(p[hi]-p[lo]);
}
__device__ std::uint32_t label_at(const std::uint8_t*labels,double x,double y,double distance,Geometry g){
  int ix=int(floor(x+0.5)),iy=int(floor(y+0.5));
  if(ix<0||ix>=g.nx||iy<0||iy>=g.ny||g.labelNz<1)return 0;
  int iz;
  if(g.labelPositions){
    int lo=0,hi=g.labelNz;
    while(lo<hi){int m=(lo+hi)/2;if(g.labelPositions[m]<distance)lo=m+1;else hi=m;}
    iz=lo==g.labelNz?g.labelNz-1:lo;
    if(iz>0&&distance-g.labelPositions[iz-1]<g.labelPositions[iz]-distance)--iz;
    // SEG may omit empty frames: a missing plane must not inherit a distant mask.
    if(fabs(distance-g.labelPositions[iz])>0.5*g.labelSpacing+1e-6)return 0;
  }else{
    iz=int(floor(distance/g.labelSpacing+0.5));
    if(iz<0||iz>=g.labelNz)return 0;
  }
  return std::uint32_t(labels[(U(iz)*g.ny+iy)*g.nx+ix]!=0);
}
__global__ void resample(const std::int16_t*raw,const float*slope,const float*intercept,const std::uint8_t*labels,
                         float*hu,std::uint32_t*y,Geometry g){
  U count=U(g.n)*g.n*g.n;
  for(U q=U(blockIdx.x)*blockDim.x+threadIdx.x;q<count;q+=U(blockDim.x)*gridDim.x){
    int x=q%g.n,yy=(q/g.n)%g.n,z=q/(U(g.n)*g.n);
    double rx=g.cx+(x-g.n/2)*g.step/g.sx,ry=g.cy+(yy-g.n/2)*g.step/g.sy,distance=g.cz+(z-g.n/2)*g.step;
    hu[q]=sample(raw,slope,intercept,rx,ry,slice_coordinate(distance,g),g);y[q]=label_at(labels,rx,ry,distance,g);
  }
}
using PreparedGeometry=class_study::prepared_image::Geometry;
__device__ float prepared_point(const float*image,int z,int y,int x,PreparedGeometry g){
  x=max(0,min(g.nx-1,x));y=max(0,min(g.ny-1,y));z=max(0,min(g.nz-1,z));
  return image[(U(z)*g.ny+y)*g.nx+x];
}
__device__ float prepared_sample(const float*image,double z,double y,double x,PreparedGeometry g){
  // Upstream preprocessing replicates the retained image edge. Physical air
  // extrapolation is applied separately using ORIGINAL source coordinates.
  x=fmax(0.,fmin(double(g.nx-1),x));y=fmax(0.,fmin(double(g.ny-1),y));z=fmax(0.,fmin(double(g.nz-1),z));
  int ix=int(floor(x)),iy=int(floor(y)),iz=int(floor(z));
  float fx=float(x-ix),fy=float(y-iy),fz=float(z-iz),value=0;
  for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i){
    float weight=(i?fx:1-fx)*(j?fy:1-fy)*(k?fz:1-fz);
    if(weight!=0)value+=prepared_point(image,iz+k,iy+j,ix+i,g)*weight;
  }
  return value;
}
__global__ void resample_prepared(const float*image,float*intensity,PreparedGeometry g,unsigned*invalid){
  U count=U(g.n)*g.n*g.n;
  for(U q=U(blockIdx.x)*blockDim.x+threadIdx.x;q<count;q+=U(blockDim.x)*gridDim.x){
    int x=q%g.n,y=(q/g.n)%g.n,z=q/(U(g.n)*g.n);
    double full[3],support=1.;
    for(unsigned a=0;a<3;++a){
      full[a]=g.cropped_map[a][0]*x+g.cropped_map[a][1]*y+g.cropped_map[a][2]*z+g.cropped_map[a][3]+g.crop_start[a];
      support*=fmax(0.,fmin(1.,full[a]+1.))*fmax(0.,fmin(1.,g.source_shape[a]-full[a]));
    }
    if(support==0.){intensity[q]=g.outside;continue;}
    double coordinate[3];bool missing=false;
    int prepared_shape[3]{g.nz,g.ny,g.nx};
    for(unsigned a=0;a<3;++a){
      double cropped=fmax(0.,fmin(double(g.source_shape[a]-1),full[a]))-g.crop_start[a];
      if(cropped< -1e-7||cropped>g.cropped_shape[a]-1+1e-7)missing=true;
      cropped=fmax(0.,fmin(double(g.cropped_shape[a]-1),cropped));
      coordinate[a]=(cropped+0.5)*prepared_shape[a]/g.cropped_shape[a]-0.5;
    }
    // Cropped-away source interior cannot be reconstructed from a saved NPY.
    // Fail closed instead of silently inventing an intensity for that region.
    if(missing){atomicOr(invalid,2u);intensity[q]=g.outside;continue;}
    float sampled=prepared_sample(image,coordinate[0],coordinate[1],coordinate[2],g);
    float value=support==1.?sampled:g.outside+float(support)*(sampled-g.outside);
    intensity[q]=value;if(!isfinite(value))atomicOr(invalid,1u);
  }
}
using ChannelGeometry=class_study::prepared_image::ChannelGeometry;
__global__ void copy_feature_columns(const float*source,float*destination,U rows,unsigned source_width,unsigned width,unsigned*invalid){
  for(U i=U(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*source_width;i+=U(blockDim.x)*gridDim.x){float value=source[i];destination[(i/source_width)*width+i%source_width]=value;if(!isfinite(value))atomicOr(invalid,1u);}
}
__global__ void audit_channel_values(const float*image,U spatial,unsigned channels,const unsigned*kinds,unsigned*invalid){
  for(U i=U(blockIdx.x)*blockDim.x+threadIdx.x;i<spatial*channels;i+=U(blockDim.x)*gridDim.x){float value=image[i];unsigned kind=kinds[i/spatial];if(!isfinite(value)||(kind==1&&(value<0||value>1))||(kind==2&&value!=0&&value!=1))atomicOr(invalid,2u);}
}
__device__ float channel_point(const float*image,unsigned channel,int z,int y,int x,ChannelGeometry g,float outside){
  if(x<0||x>=g.nx||y<0||y>=g.ny||z<0||z>=g.nz)return outside;
  return image[((U(channel)*g.nz+z)*g.ny+y)*g.nx+x];
}
__device__ float channel_sample(const float*image,unsigned channel,double z,double y,double x,ChannelGeometry g,bool nearest,float outside){
  if(nearest){
    if(x< -0.5||x>=g.nx-0.5||y< -0.5||y>=g.ny-0.5||z< -0.5||z>=g.nz-0.5)return outside;
    return channel_point(image,channel,int(floor(z+0.5)),int(floor(y+0.5)),int(floor(x+0.5)),g,outside);
  }
  if(x<=-1.||x>=g.nx||y<=-1.||y>=g.ny||z<=-1.||z>=g.nz)return outside;
  int ix=int(floor(x)),iy=int(floor(y)),iz=int(floor(z));float fx=float(x-ix),fy=float(y-iy),fz=float(z-iz),value=0;
  for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i){float weight=(i?fx:1-fx)*(j?fy:1-fy)*(k?fz:1-fz);if(weight!=0)value+=channel_point(image,channel,iz+k,iy+j,ix+i,g,outside)*weight;}
  return value;
}
__global__ void append_sampled_channels(const float*image,float*destination,ChannelGeometry g,unsigned width,unsigned first,bool nearest,const float*outside,unsigned*invalid){
  U rows=U(g.n)*g.n*g.n;
  for(U i=U(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*unsigned(g.channels);i+=U(blockDim.x)*gridDim.x){U q=i/g.channels;unsigned c=unsigned(i%g.channels);int x=q%g.n,y=(q/g.n)%g.n,z=q/(U(g.n)*g.n);double coordinate[3];
    for(unsigned a=0;a<3;++a)coordinate[a]=g.map[a][0]*x+g.map[a][1]*y+g.map[a][2]*z+g.map[a][3];
    float value=outside[c];if(isfinite(coordinate[0])&&isfinite(coordinate[1])&&isfinite(coordinate[2]))value=channel_sample(image,c,coordinate[0],coordinate[1],coordinate[2],g,nearest,outside[c]);else atomicOr(invalid,4u);
    destination[q*width+first+c]=value;if(!isfinite(value))atomicOr(invalid,2u);
  }
}
__device__ float cube(const float*hu,int x,int y,int z,int n){
  // Fixed edge replication is a feature boundary convention, not learned padding.
  x=max(0,min(n-1,x));y=max(0,min(n-1,y));z=max(0,min(n-1,z));
  return hu[(U(z)*n+y)*n+x];
}
__global__ void features(const float*hu,const std::uint32_t*y,float*out,std::uint32_t*eligible,
                         unsigned long long*counts,Geometry g,std::uint32_t stride){
  U count=U(g.n)*g.n*g.n;
  for(U q=U(blockIdx.x)*blockDim.x+threadIdx.x;q<count;q+=U(blockDim.x)*gridDim.x){
    int x=q%g.n,yy=(q/g.n)%g.n,z=q/(U(g.n)*g.n);float*f=out+q*20;
    f[0]=hu[q];int a=1;
    for(int radius=1;radius<=2;radius++){
      double sum=0,sq=0;int m=0;
      for(int dz=-radius;dz<=radius;dz++)for(int dy=-radius;dy<=radius;dy++)for(int dx=-radius;dx<=radius;dx++){
        double v=cube(hu,x+dx,yy+dy,z+dz,g.n);sum+=v;sq+=v*v;++m;
      }
      double mean=sum/m;f[a++]=float(mean);f[a++]=float(sqrt(fmax(0.,sq/m-mean*mean)));
    }
    float gx=(cube(hu,x+1,yy,z,g.n)-cube(hu,x-1,yy,z,g.n))/float(2*g.step);
    float gy=(cube(hu,x,yy+1,z,g.n)-cube(hu,x,yy-1,z,g.n))/float(2*g.step);
    float gz=(cube(hu,x,yy,z+1,g.n)-cube(hu,x,yy,z-1,g.n))/float(2*g.step);
    f[5]=gx;f[6]=gy;f[7]=gz;f[8]=sqrtf(gx*gx+gy*gy+gz*gz);
    f[9]=cube(hu,x-2,yy,z,g.n);f[10]=cube(hu,x+2,yy,z,g.n);
    f[11]=cube(hu,x,yy-2,z,g.n);f[12]=cube(hu,x,yy+2,z,g.n);
    f[13]=cube(hu,x,yy,z-2,g.n);f[14]=cube(hu,x,yy,z+2,g.n);
    float dx=float((x-g.n/2)*g.step),dy=float((yy-g.n/2)*g.step),dz=float((z-g.n/2)*g.step);
    f[15]=dx;f[16]=dy;f[17]=dz;f[18]=sqrtf(dx*dx+dy*dy+dz*dz);
    f[19]=cube(hu,g.n/2,g.n/2,g.n/2,g.n);
    // Spatial hash, not q modulo stride (which selects entire parallel planes).
    std::uint32_t h=std::uint32_t(q)+0x9e3779b9u;h^=h>>16;h*=0x7feb352du;h^=h>>15;h*=0x846ca68bu;h^=h>>16;
    eligible[q]=h%stride==0; if(y[q])atomicAdd(counts,1ULL);
    if(eligible[q]){atomicAdd(counts+1,1ULL);if(y[q])atomicAdd(counts+2,1ULL);}
  }
}
// Count the same nearest-neighbor reference on the same physical grid outside
// the fixed crop. A clipped tumor is penalized; it never expands its own crop.
__global__ void outside_reference(const std::uint8_t*labels,Geometry g,int3 start,int3 extent,unsigned long long*out){
  U count=U(extent.x)*extent.y*extent.z;
  for(U q=U(blockIdx.x)*blockDim.x+threadIdx.x;q<count;q+=U(blockDim.x)*gridDim.x){
    int x=start.x+int(q%extent.x),y=start.y+int((q/extent.x)%extent.y),z=start.z+int(q/(U(extent.x)*extent.y));
    if(x>=0&&x<g.n&&y>=0&&y<g.n&&z>=0&&z<g.n)continue;
    if(label_at(labels,g.cx+(x-g.n/2)*g.step/g.sx,g.cy+(y-g.n/2)*g.step/g.sy,g.cz+(z-g.n/2)*g.step,g))atomicAdd(out,1ULL);
  }
}
template<class T>void append(std::ofstream&f,const std::vector<T>&v){f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*sizeof(T)));need(bool(f),"dataset write failed");}
std::ofstream output(const fs::path&p){std::ofstream f(p,std::ios::binary);need(bool(f),"cannot create "+p.string());return f;}
J names(){return J::array({"hu","mean_radius_1_grid_step","std_radius_1_grid_step","mean_radius_2_grid_steps","std_radius_2_grid_steps","gradient_x","gradient_y","gradient_z","gradient_magnitude","hu_x_minus_2_grid_steps","hu_x_plus_2_grid_steps","hu_y_minus_2_grid_steps","hu_y_plus_2_grid_steps","hu_z_minus_2_grid_steps","hu_z_plus_2_grid_steps","seed_relative_x_mm","seed_relative_y_mm","seed_relative_z_mm","seed_distance_mm","seed_hu"});}
J normalized_names(){
  J result=names();
  for(auto& item:result){std::string name=item.get<std::string>();auto at=name.find("hu");if(at!=std::string::npos)name.replace(at,2,"normalized_intensity");else if(name.find("seed_relative_")!=0&&name!="seed_distance_mm")name="normalized_"+name;item=name;}
  return result;
}
std::string npy_fixture(const std::string& descriptor="<f4",bool fortran=false,const std::string& shape="1, 2, 3, 4",std::size_t payload=24){
  std::string header="{'descr': '"+descriptor+"', 'fortran_order': "+(fortran?"True":"False")+", 'shape': ("+shape+"), }";
  while((10+header.size()+1)%64)header+=' ';header+='\n';
  std::string bytes="\x93NUMPY";bytes+=char(1);bytes+=char(0);bytes+=char(header.size()&255);bytes+=char(header.size()>>8);bytes+=header;
  for(std::size_t i=0;i<payload;++i){float value=float(i)+0.25f;bytes.append(reinterpret_cast<const char*>(&value),sizeof(value));}
  return bytes;
}
template<class F>void rejects(F&& call,const std::string& message){bool rejected=false;try{call();}catch(const std::exception&){rejected=true;}need(rejected,message);}
J prepared_fixture_metadata(){return J{{"properties",{{"spacing",J::array({4.,3.,2.})},{"shape_before_cropping",J::array({7,11,9})},
  {"bbox_used_for_cropping",J::array({J::array({1,6}),J::array({2,10}),J::array({3,9})})},{"shape_after_cropping_and_before_resampling",J::array({5,8,6})}}},
  {"target_spacing_ZYX",J::array({1.,2.,1.5})}};}
J prepared_fixture_export(){return J{{"spacingXYZ",J::array({2.,3.,4.})},{"sizeXYZ",J::array({7,9,11})},
  {"originLPS",J::array({10.,-5.,8.})},{"directionLPS",J::array({1.,0.,0.,0.,1.,0.,0.,0.,1.})},
  {"cropGrid",{{"sizeXYZ",J::array({4,4,4})},{"spacingXYZ",J::array({2.,2.,2.})},
    {"originLPS",J::array({12.5,5.5,20.})},{"directionLPS",J::array({1.,0.,0.,0.,1.,0.,0.,0.,1.})}}}};}
int prepared_host_test(){
  namespace pi=class_study::prepared_image;
  auto parse=[](const std::string& bytes){std::istringstream stream(bytes,std::ios::binary);return pi::read_npy(stream,bytes.size());};
  auto array=parse(npy_fixture());need(array.shape==std::array<int,4>{1,2,3,4}&&array.values[23]==23.25f,"NPY CZYX transport failed");
  auto multichannel=npy_fixture("<f4",false,"2, 2, 3, 4",48);std::istringstream multistream(multichannel,std::ios::binary);auto channels=pi::read_npy(multistream,multichannel.size(),2);need(channels.shape==std::array<int,4>{2,2,3,4}&&channels.values[47]==47.25f,"declared multichannel NPY transport failed");
  for(const auto& bytes:std::vector<std::string>{npy_fixture(">f4"),npy_fixture("<f8"),npy_fixture("<i4"),npy_fixture("<f4",true),
      npy_fixture("<f4",false,"2, 2, 3, 4"),npy_fixture("<f4",false,"1, 0, 3, 4"),
      npy_fixture("<f4",false,"1, -2, 3, 4"),npy_fixture("<f4",false,"1, 2, 3, 4, 1"),
      npy_fixture("<f4",false,"1, 2, 3, 4",23),npy_fixture("<f4",false,"1, 2, 3, 4",25),
      npy_fixture("<f4",false,"1, 2147483648, 3, 4"),npy_fixture("<f4",false,"1, 2147483647, 2147483647, 2147483647")})
    rejects([&]{parse(bytes);},"unsafe NPY dtype/shape/payload was accepted");
  auto metadata=prepared_fixture_metadata(),exported=prepared_fixture_export();
  auto g=pi::geometry(exported,metadata,{1,10,16,12},{2,0,1},4,2.);
  // Independently specified half-center mapping: prepared axes are source X,
  // Z,Y; the nonzero crop bbox is subtracted before rounded-shape scaling.
  need(std::abs(g.map[0][3]-1.)<1e-12&&std::abs(g.map[0][0]-2.)<1e-12&&
       std::abs(g.map[1][3]-2.5)<1e-12&&std::abs(g.map[1][2]-1.)<1e-12&&
       std::abs(g.map[2][3]-1.5)<1e-12&&std::abs(g.map[2][1]-4./3.)<1e-12,"prepared transpose/crop/anisotropic spacing geometry failed");
  auto bad=metadata;bad["properties"]["bbox_used_for_cropping"][0]=J::array({1,7});
  rejects([&]{pi::geometry(exported,bad,{1,10,16,12},{2,0,1},4,2.);},"invalid prepared bbox accepted");
  rejects([&]{pi::permutation(J::array({0,0,2}));},"duplicate transpose axis accepted");
  bad=metadata;bad["target_spacing_ZYX"][0]=0.;
  rejects([&]{pi::geometry(exported,bad,{1,10,16,12},{2,0,1},4,2.);},"invalid prepared target spacing accepted");
  rejects([&]{pi::geometry(exported,metadata,{1,11,16,12},{2,0,1},4,2.);},"prepared rounded-shape mismatch accepted");
  J plans={{"configurations",{{"base",{{"normalization_schemes",J::array({"CTNormalization"})},{"use_mask_for_norm",J::array({false})}}},{"child",{{"inherits_from","base"}}}}},
    {"foreground_intensity_properties_per_channel",{{"0",{{"mean",0.1},{"std",3.},{"percentile_00_5",-10.},{"percentile_99_5",20.}}}}}};
  need(pi::normalized_air(plans,"child")==-3.3666667938232421875f,"synthetic float32 CTNormalization air transform failed");
  auto wrong_normalization=plans;wrong_normalization["configurations"]["base"]["normalization_schemes"]=J::array({"ZScoreNormalization"});
  rejects([&]{pi::normalized_air(wrong_normalization,"child");},"unsupported normalization accepted");
  J volume_grid={{"originLPS",J::array({10.,20.,30.})},{"spacingXYZ",J::array({2.,3.,4.})},{"directionLPS",J::array({0.,-1.,0.,1.,0.,0.,0.,0.,1.})},{"sizeXYZ",J::array({5,4,3})}};
  auto channel=pi::channel_geometry(volume_grid,exported.at("cropGrid"),{2,3,4,5},4,2.);
  need(channel.channels==2&&std::abs(channel.map[0][3]+2.5)<1e-12&&std::abs(channel.map[0][2]-0.5)<1e-12&&std::abs(channel.map[1][3]+5./6.)<1e-12&&std::abs(channel.map[1][0]+2./3.)<1e-12&&std::abs(channel.map[2][3]+7.25)<1e-12&&channel.map[2][1]==1.,"channel rotated anisotropic physical affine failed");
  auto invalid_grid=volume_grid;invalid_grid["sizeXYZ"][0]=6;rejects([&]{pi::channel_geometry(invalid_grid,exported.at("cropGrid"),{2,3,4,5},4,2.);},"channel grid/shape mismatch accepted");invalid_grid=volume_grid;invalid_grid["spacingXYZ"][1]=0.;rejects([&]{pi::channel_geometry(invalid_grid,exported.at("cropGrid"),{2,3,4,5},4,2.);},"zero channel spacing accepted");
  std::cout<<"prepared image host transport/geometry checks passed\n";return 0;
}
void prepared_gpu_test(){
  auto g=class_study::prepared_image::geometry(prepared_fixture_export(),prepared_fixture_metadata(),{1,10,16,12},{2,0,1},4,2.);
  std::vector<float>source(10*16*12);
  for(int z=0;z<10;++z)for(int y=0;y<16;++y)for(int x=0;x<12;++x)source[(z*16+y)*12+x]=float(100*z+10*y+x);
  Device<float>image(source.size()),values(64);image.upload(source.data());Device<unsigned>invalid(1);ck(cudaMemset(invalid.p,0,sizeof(unsigned)));
  resample_prepared<<<1,128>>>(image.p,values.p,g,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
  std::vector<float>result(64);values.download(result.data());
  for(unsigned q=0;q<64;++q){double x=q%4,y=(q/4)%4,z=q/16;
    double expected=100*(1+2*x)+10*(2.5+z)+(1.5+4*y/3);
    need(std::abs(result[q]-expected)<1e-3,"CUDA prepared nonidentity transpose/crop/anisotropic linear interpolation failed");}
  // Original source air padding at every face, fractional blending at its
  // physical boundary, and large outside coordinates exercise safe indexing.
  PreparedGeometry boundary{2,2,2,4};boundary.outside=-3.f;
  for(unsigned a=0;a<3;++a){boundary.source_shape[a]=2;boundary.cropped_shape[a]=2;}
  boundary.cropped_map[2][0]=0.5;boundary.cropped_map[2][3]=-1.;
  std::vector<float>constant(8,8.f);Device<float>small(8);small.upload(constant.data());
  resample_prepared<<<1,128>>>(small.p,values.p,boundary,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());values.download(result.data());
  for(unsigned q=0;q<64;++q)need(result[q]==(q%4==0?-3.f:q%4==1?2.5f:8.f),"CUDA prepared original outside/fractional air padding failed");
  for(unsigned axis=0;axis<3;++axis)for(double outside:{-1.,2.,1e100}){
    auto out=boundary;for(unsigned a=0;a<3;++a)for(unsigned b=0;b<4;++b)out.cropped_map[a][b]=0.;out.cropped_map[axis][3]=outside;
    resample_prepared<<<1,128>>>(small.p,values.p,out,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());values.download(result.data());
    for(float v:result)need(v==-3.f,"CUDA prepared original outside face air padding failed");}
  // A downsampled saved array has shifted half-pixel endpoints. Its edge must
  // not shrink ORIGINAL source support or inject air at source voxel center0.
  auto downsampled=boundary;for(unsigned a=0;a<3;++a){downsampled.source_shape[a]=4;downsampled.cropped_shape[a]=4;}
  resample_prepared<<<1,128>>>(small.p,values.p,downsampled,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());values.download(result.data());
  for(unsigned q=0;q<64;++q)need(result[q]==(q%4==0?-3.f:q%4==1?2.5f:8.f),"CUDA prepared half-pixel edge shifted original source support");
  downsampled.cropped_map[2][3]=2.;
  resample_prepared<<<1,128>>>(small.p,values.p,downsampled,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());values.download(result.data());
  for(unsigned q=0;q<64;++q)need(result[q]==(q%4==3?2.5f:8.f),"CUDA prepared upper physical source air halo failed");
  constant[0]=std::numeric_limits<float>::quiet_NaN();small.upload(constant.data());
  boundary.cropped_map[2][0]=0.;boundary.cropped_map[2][3]=0.;ck(cudaMemset(invalid.p,0,sizeof(unsigned)));
  resample_prepared<<<1,128>>>(small.p,values.p,boundary,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());unsigned bad=0;invalid.download(&bad);need(bad==1,"CUDA prepared nonfinite input check failed");
  auto omitted=boundary;omitted.source_shape[2]=4;omitted.crop_start[2]=1;omitted.cropped_map[2][3]=-1.;ck(cudaMemset(invalid.p,0,sizeof(unsigned)));
  resample_prepared<<<1,128>>>(small.p,values.p,omitted,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());invalid.download(&bad);need(bad==2,"CUDA prepared cropped-away source interior guard failed");
  ChannelGeometry cg{2,2,2,4,2};cg.map[2][0]=0.5;cg.map[2][3]=-0.5;
  std::vector<float> volume(16),base_values(64*20);for(unsigned i=0;i<8;++i){volume[i]=float(100*(i/4)+10*((i/2)%2)+3*(i%2));volume[8+i]=1000+volume[i];}for(unsigned i=0;i<base_values.size();++i)base_values[i]=i%20==0?-0.f:float(i)+0.25f;
  Device<float> channel_image(16),base_columns(base_values.size()),packed(64*22),pads(2);channel_image.upload(volume.data());base_columns.upload(base_values.data());float padding[2]{-3.f,-4.f};pads.upload(padding);ck(cudaMemset(invalid.p,0,sizeof(unsigned)));
  copy_feature_columns<<<1,128>>>(base_columns.p,packed.p,64,20,22,invalid.p);append_sampled_channels<<<1,128>>>(channel_image.p,packed.p,cg,22,20,false,pads.p,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());std::vector<float> augmented(64*22);packed.download(augmented.data());
  for(unsigned q=0;q<64;++q){for(unsigned c=0;c<20;++c)need(std::bit_cast<std::uint32_t>(augmented[q*22+c])==std::bit_cast<std::uint32_t>(base_values[q*20+c]),"CUDA base feature words changed during append");float expected[4]{-1.5f,0.f,1.5f,3.f},second[4]{498.f,1000.f,1001.5f,1003.f};need(augmented[q*22+20]==expected[q%4]&&augmented[q*22+21]==second[q%4],"CUDA multichannel linear/constant outside sampling failed");}
  append_sampled_channels<<<1,128>>>(channel_image.p,packed.p,cg,22,20,true,pads.p,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());packed.download(augmented.data());for(unsigned q=0;q<64;++q)need(augmented[q*22+20]==(q%4<2?0.f:3.f),"CUDA nearest channel rounding failed");
  cg.map[2][0]=0.;cg.map[2][3]=1e100;append_sampled_channels<<<1,128>>>(channel_image.p,packed.p,cg,22,20,false,pads.p,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());packed.download(augmented.data());for(unsigned q=0;q<64;++q)need(augmented[q*22+20]==-3.f&&augmented[q*22+21]==-4.f,"CUDA large outside channel indexing failed");
  Device<unsigned> kinds(2);unsigned declared_kind[2]{2,1};kinds.upload(declared_kind);ck(cudaMemset(invalid.p,0,sizeof(unsigned)));audit_channel_values<<<1,128>>>(channel_image.p,8,2,kinds.p,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());invalid.download(&bad);need(bad==2,"CUDA invalid binary/probability channel accepted");
}
std::vector<std::uint32_t> reference_segment(const fs::path& path,U start,U count){
  need(start<=U(std::numeric_limits<std::streamoff>::max())/4&&count<=U(std::numeric_limits<std::streamsize>::max())/4,"reference extent too large");
  std::ifstream input(path,std::ios::binary);need(bool(input),"cannot read "+path.string());
  input.seekg(std::streamoff(start*4));need(bool(input),"invalid reference row offset");
  std::vector<std::uint32_t> result(count);input.read(reinterpret_cast<char*>(result.data()),std::streamsize(count*4));
  need(bool(input),"truncated reference data");return result;
}
std::vector<float> reference_features(const fs::path& path,U start,U count,unsigned width){
  need(start<=U(std::numeric_limits<std::streamoff>::max())/4/width&&count<=std::numeric_limits<std::size_t>::max()/4/width&&count<=U(std::numeric_limits<std::streamsize>::max())/4/width,"reference feature segment extent overflow");
  std::ifstream input(path,std::ios::binary);need(bool(input),"cannot read existing feature values");input.seekg(std::streamoff(start*width*4));std::vector<float> values(std::size_t(count)*width);input.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*4));need(bool(input),"truncated existing feature values");return values;
}
int prepare_channels(int argc,char**argv){
  namespace pi=class_study::prepared_image;std::map<std::string,fs::path> options;
  for(int i=1;i<argc;i+=2){std::string key=argv[i];need(key=="--additional-channels"||key=="--reference-data"||key=="--export"||key=="--out","unknown additional-channel option "+key);need(i+1<argc&&options.emplace(key,fs::absolute(argv[i+1]).lexically_normal()).second,"missing or repeated additional-channel option "+key);}
  for(const auto& key:{"--additional-channels","--reference-data","--export","--out"})need(options.contains(key),std::string("missing ")+key);
  const auto& index_path=options.at("--additional-channels");const auto& reference_path=options.at("--reference-data");const auto& export_path=options.at("--export");const auto& root=options.at("--out");need(!fs::exists(root),"output must be fresh");
  J index=pi::json_file(index_path),reference=pi::json_file(reference_path),exported=pi::json_file(export_path);
  need(index.at("format")=="segmentation-additional-channels-1"&&index.at("complete")==true&&index.at("array_axis_order")=="CZYX","complete additional-channel CZYX index required");
  need(index.at("reference_labels_used_for_features")==false&&index.at("independent_seed")==true&&index.at("frozen_network")==true,"channel input must declare frozen network, independent seed and no reference-label features");
  need(index.at("model_provenance").is_object()&&!index.at("model_provenance").empty(),"frozen network model provenance required");
  need(reference.at("format")=="grouped-segmentation-dataset-1"&&reference.at("classes")==2&&reference.at("features")==20&&reference.value("complete",false),"complete binary twenty-feature reference dataset required");
  need(reference.at("feature_names").is_array()&&reference.at("feature_names").size()==20,"twenty reference feature names required");
  need(reference.at("scans").is_array()&&!reference.at("scans").empty()&&index.at("cases").is_array()&&exported.at("cases").is_array()&&index.at("cases").size()==reference.at("scans").size()&&exported.at("cases").size()==reference.at("scans").size(),"channel/export/reference scan coverage mismatch");
  double step=reference.at("grid_spacing_mm").get<double>(),ratio=reference.at("crop_width_mm").get<double>()/step;need(std::isfinite(step)&&step>0&&std::isfinite(ratio)&&ratio>=4&&ratio<=256&&ratio==std::round(ratio),"invalid reference feature crop");int n=int(ratio);need(n%2==0,"reference feature crop must be even");U per_scan=U(n)*n*n,rows=reference.at("rows").get<U>();need(rows<=std::numeric_limits<U>::max()/80,"reference feature extent overflow");
  fs::path base_values=pi::relative_file(reference_path,reference.at("values_path").get<std::string>());need(fs::is_regular_file(base_values)&&fs::file_size(base_values)==rows*80,"reference feature file extent differs");
  std::map<std::string,fs::path> reused;for(const auto& key:{"labels_path","patient_ids_path","scan_ids_path","train_eligible_path"}){auto path=pi::relative_file(reference_path,reference.at(key).get<std::string>());need(fs::is_regular_file(path)&&fs::file_size(path)==rows*4,std::string("invalid reference array extent: ")+key);reused[key]=path;}
  if(exported.contains("source_descriptor")){auto source_path=pi::relative_file(export_path,exported.at("source_descriptor").get<std::string>());J source=pi::json_file(source_path);for(const auto& [key,path]:reused)need(pi::relative_file(source_path,source.at(key).get<std::string>())==path,"channel export/reference row-array identity differs");}
  std::map<std::uint32_t,J> inputs,physical;for(const auto& item:index.at("cases"))need(inputs.emplace(item.at("scan_id").get<std::uint32_t>(),item).second,"duplicate channel scan id");for(const auto& item:exported.at("cases"))need(physical.emplace(item.at("scan_id").get<std::uint32_t>(),item).second,"duplicate channel export scan id");
  struct Volume {J declaration;fs::path path;std::array<int,4> shape;ChannelGeometry geometry;std::vector<unsigned> kinds;std::vector<float> outside;bool nearest;unsigned first;};
  struct Case {J original,input;std::vector<Volume> volumes;};std::vector<Case> cases;J declared_schema=nullptr,feature_names=reference.at("feature_names"),operators=reference.value("feature_operator_names",reference.at("feature_names"));need(operators.is_array()&&operators.size()==20,"reference feature operator extent");std::set<std::string> unique_names;for(const auto& name:feature_names)need(unique_names.insert(name.get<std::string>()).second,"duplicate reference feature name");U next_row=0;unsigned width=20;
  for(const auto& scan:reference.at("scans")){auto id=scan.at("scan_id").get<std::uint32_t>();need(inputs.contains(id)&&physical.contains(id),"missing channel/export scan");J item=inputs.at(id),grid_case=physical.at(id);inputs.erase(id);physical.erase(id);
    need(item.at("patient_id")==scan.at("patient_id")&&grid_case.at("patient_id")==scan.at("patient_id")&&item.at("case")==grid_case.at("case"),"channel/export/reference patient identity differs");need(scan.at("row_start")==next_row&&scan.at("rows")==per_scan,"channel reference row geometry differs");need(grid_case.at("primary_prompt_uses_reference_mask")==false,"channel crop must use the independent supplied seed");
    if(grid_case.contains("preparedScan"))for(const auto& key:{"row_start","rows","truth_positive_rows","truth_positive_outside_rows"})need(grid_case.at("preparedScan").at(key)==scan.at(key),"channel export/reference accounting differs");
    auto seed=pi::numbers<3>(grid_case.at("seedLPS"),"channel independent seed"),origin=pi::numbers<3>(grid_case.at("cropGrid").at("originLPS"),"channel crop origin");auto direction=pi::numbers<9>(grid_case.at("cropGrid").at("directionLPS"),"channel crop direction");for(unsigned r=0;r<3;++r){double center=origin[r];for(unsigned a=0;a<3;++a)center+=direction[r*3+a]*(n/2)*step;need(std::abs(center-seed[r])<1e-6,"channel crop does not retain independent seed center");}
    auto source_spacing=pi::numbers<3>(grid_case.at("spacingXYZ"),"channel source spacing"),source_origin=pi::numbers<3>(grid_case.at("originLPS"),"channel source origin"),seed_grid=pi::numbers<3>(grid_case.at("seedGridXYZ"),"channel source seed");auto source_direction=pi::numbers<9>(grid_case.at("directionLPS"),"channel source direction");pi::orthonormal(source_direction);
    for(double spacing:source_spacing)need(spacing>0,"channel source spacing must be positive");double index_base=scan.at("seed_pixel_index_base").get<double>();need(std::abs(seed_grid[0]-(scan.at("seed_x_pixel").get<double>()-index_base))<1e-7&&std::abs(seed_grid[1]-(scan.at("seed_y_pixel").get<double>()-index_base))<1e-7&&std::abs(seed_grid[2]-scan.at("dicom").at("seed_projection_offset_mm").get<double>()/source_spacing[2])<1e-7,"channel export/reference supplied seed differs");
    for(unsigned r=0;r<3;++r){double expected=source_origin[r];for(unsigned a=0;a<3;++a)expected+=source_direction[r*3+a]*seed_grid[a]*source_spacing[a];need(std::abs(expected-seed[r])<1e-6,"channel export source/physical seed differs");}for(unsigned i=0;i<9;++i)need(std::abs(source_direction[i]-direction[i])<1e-7,"channel crop/source direction differs");
    need(item.at("arrays").is_array()&&!item.at("arrays").empty(),"channel case needs array volumes");Case prepared{scan,item,{}};J schema=J::array();unsigned first=20;
    for(const auto& array:item.at("arrays")){auto dimensions=pi::numbers<4>(array.at("shape_CZYX"),"channel CZYX shape");std::array<int,4> shape;for(unsigned a=0;a<4;++a){need(dimensions[a]>=1&&dimensions[a]<=std::numeric_limits<int>::max()&&dimensions[a]==std::floor(dimensions[a]),"invalid channel CZYX shape");shape[a]=int(dimensions[a]);}unsigned channels=unsigned(shape[0]);need(channels<=UINT32_MAX-first,"channel feature width overflow");
      need(array.at("channel_names").is_array()&&array.at("channel_names").size()==channels&&array.at("channel_kinds").is_array()&&array.at("channel_kinds").size()==channels&&array.at("outside_values").is_array()&&array.at("outside_values").size()==channels,"channel declaration extent differs");bool nearest=array.at("interpolation")=="nearest";need(nearest||array.at("interpolation")=="linear","channel interpolation must be nearest or linear");
      std::vector<unsigned> kinds;std::vector<float> outside;for(unsigned c=0;c<channels;++c){std::string name=array.at("channel_names").at(c).get<std::string>(),kind=array.at("channel_kinds").at(c).get<std::string>();need(!name.empty()&&kind.size()>0,"channel names/kinds required");unsigned code=kind=="learned"?0:kind=="probability"?1:kind=="binary_mask"?2:3;need(code<3&&(code!=2||nearest),"channel kind unsupported or binary mask interpolation is not nearest");float pad=array.at("outside_values").at(c).get<float>();need(std::isfinite(pad)&&(code!=1||(pad>=0&&pad<=1))&&(code!=2||pad==0||pad==1),"channel outside value differs from declared kind");kinds.push_back(code);outside.push_back(pad);}
      schema.push_back({{"channel_names",array.at("channel_names")},{"channel_kinds",array.at("channel_kinds")},{"interpolation",array.at("interpolation")},{"outside_values",array.at("outside_values")}});auto geometry=pi::channel_geometry(array.at("grid"),grid_case.at("cropGrid"),shape,n,step);prepared.volumes.push_back({array,pi::relative_file(index_path,array.at("array").get<std::string>()),shape,geometry,kinds,outside,nearest,first});first+=channels;
    }
    if(declared_schema.is_null()){declared_schema=schema;width=first;for(const auto& volume:prepared.volumes)for(const auto& name:volume.declaration.at("channel_names")){std::string label=name.get<std::string>();need(unique_names.insert(label).second,"duplicate additional feature name");feature_names.push_back(label);operators.push_back("sample_frozen_CNN_"+label);}}
    else need(schema==declared_schema&&first==width,"channel names/order/operators differ between scans");cases.push_back(std::move(prepared));next_row+=per_scan;
  }
  need(next_row==rows&&inputs.empty()&&physical.empty()&&width>20&&width<=INT32_MAX&&rows<=std::numeric_limits<U>::max()/4/width&&per_scan<=std::numeric_limits<std::size_t>::max()/4/width,"incomplete channel coverage or feature extent overflow");
  fs::create_directories(root);auto values=output(root/"values.f32");J result=reference;result["values_path"]=(root/"values.f32").string();result["features"]=width;result["feature_names"]=feature_names;result["feature_operator_names"]=operators;result["scans"]=J::array();for(const auto& [key,path]:reused)result[key]=path.string();
  result["domain_description"]="Unchanged independent physical crop and reference; existing twenty native image features plus frozen CNN-derived volume channels";
  result["feature_channels_provenance"]={{"mode","append-frozen-CNN-volume-channels"},{"additional_channels_index",index_path.string()},{"reference_dataset",reference_path.string()},{"base_values_path",base_values.string()},{"export_cases",export_path.string()},{"base_feature_count",20},{"base_feature_FP32_words_preserved",true},{"reference_row_arrays_reused",true},{"reference_labels_read_for_features",false},{"frozen_network",true},{"requires_pretrained_network_at_inference",true},{"model_provenance",index.at("model_provenance")},{"channel_schema",declared_schema},{"array_axis_order","CZYX"},{"sampling","CUDA physical LPS affine; nearest uses floor(index+0.5); linear uses trilinear interpolation with each channel's declared constant outside value"}};
  for(const auto& c:cases){U start=c.original.at("row_start").get<U>();auto source=reference_features(base_values,start,per_scan,20);Device<float> base(source.size()),combined(per_scan*width);base.upload(source.data());Device<unsigned> invalid(1);ck(cudaMemset(invalid.p,0,sizeof(unsigned)));unsigned blocks=unsigned(std::min<U>(4096,(per_scan*width+255)/256));copy_feature_columns<<<blocks,256>>>(base.p,combined.p,per_scan,20,width,invalid.p);ck(cudaGetLastError());J detail=c.original;detail["feature_channels"]=J::array();
    for(const auto& volume:c.volumes){auto array=pi::read_npy(volume.path,unsigned(volume.shape[0]));need(array.shape==volume.shape,"channel NPY/declaration shape mismatch");Device<float> image(array.values.size()),outside(volume.outside.size());Device<unsigned> kinds(volume.kinds.size());image.upload(array.values.data());outside.upload(volume.outside.data());kinds.upload(volume.kinds.data());U spatial=U(volume.shape[1])*volume.shape[2]*volume.shape[3];audit_channel_values<<<blocks,256>>>(image.p,spatial,unsigned(volume.shape[0]),kinds.p,invalid.p);append_sampled_channels<<<blocks,256>>>(image.p,combined.p,volume.geometry,width,volume.first,volume.nearest,outside.p,invalid.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());J metadata=volume.declaration;metadata["array"]=volume.path.string();detail["feature_channels"].push_back(std::move(metadata));}
    ck(cudaDeviceSynchronize());unsigned bad=0;invalid.download(&bad);need(!(bad&1),"existing base feature contains nonfinite value");need(!(bad&2),"channel contains nonfinite or invalid probability/binary value");need(!(bad&4),"channel physical affine produces nonfinite coordinates");std::vector<float> packed(per_scan*width);combined.download(packed.data());append(values,packed);result["scans"].push_back(std::move(detail));std::cout<<J{{"event","additional_channel_features"},{"scan_id",c.original.at("scan_id")},{"patient_id",c.original.at("patient_id")},{"rows",per_scan},{"features",width}}.dump()<<'\n'<<std::flush;
  }
  values.close();need(bool(values)&&fs::file_size(root/"values.f32")==rows*width*4,"additional-channel feature write extent differs");result["complete"]=true;auto descriptor=output(root/"dataset.json");descriptor<<result.dump(2)<<'\n';descriptor.close();need(bool(descriptor),"additional-channel descriptor write failed");std::cout<<J{{"event","complete"},{"dataset",(root/"dataset.json").string()},{"rows",rows},{"features",width}}.dump()<<'\n'<<std::flush;return 0;
}
int prepare_images(int argc,char**argv){
  namespace pi=class_study::prepared_image;
  std::map<std::string,fs::path> options;
  for(int i=1;i<argc;i+=2){
    std::string key=argv[i];need(key=="--prepared-images"||key=="--reference-data"||key=="--export"||key=="--preprocessing-plans"||key=="--out","unknown prepared-image option "+key);
    need(i+1<argc&&options.emplace(key,fs::absolute(argv[i+1]).lexically_normal()).second,"missing or repeated prepared-image option "+key);
  }
  for(const auto& key:{"--prepared-images","--reference-data","--export","--out"})need(options.contains(key),std::string("missing ")+key);
  const auto& index_path=options.at("--prepared-images");const auto& reference_path=options.at("--reference-data");
  const auto& export_path=options.at("--export");const auto& root=options.at("--out");need(!fs::exists(root),"output must be fresh");
  J index=pi::json_file(index_path),reference=pi::json_file(reference_path),exported=pi::json_file(export_path);
  need(index.value("complete",false)&&index.value("array_axis_order",std::string{})=="CZYX","prepared index must be complete and CZYX");
  need(reference.at("format")=="grouped-segmentation-dataset-1"&&reference.at("classes")==2&&reference.at("features")==20,"expected original binary twenty-feature reference dataset");
  need(index.at("cases").is_array()&&exported.at("cases").is_array()&&reference.at("scans").is_array()&&!reference.at("scans").empty(),"case arrays required");
  need(index.at("cases").size()==reference.at("scans").size()&&exported.at("cases").size()==reference.at("scans").size(),"prepared/export/reference case coverage mismatch");
  if(index.contains("case_count"))need(index.at("case_count")==index.at("cases").size(),"prepared index case count mismatch");
  if(exported.contains("source_descriptor"))need(pi::relative_file(export_path,exported.at("source_descriptor").get<std::string>())==reference_path,"export did not use the selected reference descriptor");
  double width=reference.at("crop_width_mm").get<double>(),step=reference.at("grid_spacing_mm").get<double>();
  need(std::isfinite(width)&&std::isfinite(step)&&width>0&&step>0,"invalid reference physical geometry");
  double ratio=width/step;need(ratio>=4&&ratio<=256&&std::abs(ratio-std::round(ratio))<1e-9,"invalid reference crop dimensions");
  int n=int(std::round(ratio));need(n%2==0,"reference crop must have even side length");U per_scan=U(n)*n*n,rows=reference.at("rows").get<U>();
  auto stride=reference.at("training_sample_stride").get<std::uint32_t>();need(stride>0,"invalid reference sampling stride");
  need(rows<=std::numeric_limits<U>::max()/4,"reference row extent overflow");
  std::map<std::string,fs::path> reused;
  for(const auto& key:{"labels_path","patient_ids_path","scan_ids_path","train_eligible_path"}){
    auto path=pi::relative_file(reference_path,reference.at(key).get<std::string>());
    need(fs::is_regular_file(path)&&fs::file_size(path)==rows*4,std::string("invalid reference array extent: ")+key);reused[key]=path;
  }
  need(options.contains("--preprocessing-plans"),"explicit --preprocessing-plans required for declared CTNormalization and transpose");
  J plans=pi::json_file(options.at("--preprocessing-plans"));auto transpose=pi::permutation(plans.at("transpose_forward"));
  if(index.contains("transpose_forward"))need(pi::permutation(index.at("transpose_forward"))==transpose,"index/plans transpose mismatch");
  std::map<std::uint32_t,J> images,geometry_cases;
  for(const auto& item:index.at("cases"))need(images.emplace(item.at("scan_id").get<std::uint32_t>(),item).second,"duplicate prepared scan id");
  for(const auto& item:exported.at("cases"))need(geometry_cases.emplace(item.at("scan_id").get<std::uint32_t>(),item).second,"duplicate export scan id");
  struct Case {J original,input,metadata;fs::path image,metadata_path;PreparedGeometry geometry;std::array<int,4> shape;};
  std::vector<Case> cases;U next_row=0;
  for(const auto& scan:reference.at("scans")){
    auto id=scan.at("scan_id").get<std::uint32_t>(),patient=scan.at("patient_id").get<std::uint32_t>();
    need(images.contains(id)&&geometry_cases.contains(id),"missing prepared/export scan");
    J item=images.at(id),physical=geometry_cases.at(id);images.erase(id);geometry_cases.erase(id);
    need(item.at("patient_id")==patient&&physical.at("patient_id")==patient&&item.at("case")==physical.at("case"),"patient/case identity mismatch");
    need(scan.at("row_start")==next_row&&scan.at("rows")==per_scan,"reference scan row geometry mismatch");
    if(physical.contains("preparedScan")){
      const auto& accounting=physical.at("preparedScan");
      for(const auto& key:{"row_start","rows","truth_positive_rows","truth_positive_outside_rows"})need(accounting.at(key)==scan.at(key),std::string("export/reference accounting mismatch: ")+key);
    }
    need(!physical.value("primary_prompt_uses_reference_mask",true),"prepared features require independent supplied seed");
    need(item.at("image_dtype")=="float32","prepared index must declare float32 images");
    auto image_path=pi::relative_file(index_path,item.at("image").get<std::string>()),metadata_path=pi::relative_file(index_path,item.at("image_metadata").get<std::string>());
    J metadata=pi::json_file(metadata_path);
    need(metadata.at("case")==item.at("case")&&!metadata.value("reference_mask_used_for_preprocessing",true),"prepared image metadata requires independent image-only preprocessing");
    need(metadata.at("shape_CZYX")==item.at("image_shape_CZYX")&&metadata.at("target_spacing_ZYX")==item.at("target_spacing_ZYX"),"prepared index/metadata geometry mismatch");
    need(pi::relative_file(metadata_path,metadata.at("source_image").get<std::string>())==pi::relative_file(export_path,physical.at("image_path").get<std::string>()),"prepared source image does not match export");
    if(metadata.contains("transpose_forward"))need(pi::permutation(metadata.at("transpose_forward"))==transpose,"metadata/plans transpose mismatch");
    auto dimensions=pi::numbers<4>(metadata.at("shape_CZYX"),"prepared CZYX shape");std::array<int,4> shape;
    for(unsigned a=0;a<4;++a){need(dimensions[a]>=1&&dimensions[a]<=std::numeric_limits<int>::max()&&dimensions[a]==std::floor(dimensions[a]),"invalid prepared CZYX shape");shape[a]=int(dimensions[a]);}
    need(shape[0]==1,"only one prepared image channel is supported");
    if(!plans.is_null()){
      std::string configuration=item.at("configuration").get<std::string>();J setting=plans.at("configurations").at(configuration);unsigned depth=0;
      while(!setting.contains("spacing")){need(setting.contains("inherits_from")&&++depth<16,"plans configuration has no resolved spacing");setting=plans.at("configurations").at(setting.at("inherits_from").get<std::string>());}
      need(setting.at("spacing")==metadata.at("target_spacing_ZYX"),"plans/prepared target spacing mismatch");
    }
    auto g=pi::geometry(physical,metadata,shape,transpose,n,step);
    g.outside=pi::normalized_air(plans,item.at("configuration").get<std::string>());
    auto source_spacing=pi::numbers<3>(physical.at("spacingXYZ"),"source spacing");
    auto seed=pi::numbers<3>(physical.at("seedGridXYZ"),"independent source seed");
    double index_base=scan.at("seed_pixel_index_base").get<double>();
    need(std::abs(seed[0]-(scan.at("seed_x_pixel").get<double>()-index_base))<1e-7&&
         std::abs(seed[1]-(scan.at("seed_y_pixel").get<double>()-index_base))<1e-7&&
         std::abs(seed[2]-scan.at("dicom").at("seed_projection_offset_mm").get<double>()/source_spacing[2])<1e-7,"export/reference independent seed mismatch");
    auto seed_lps=pi::numbers<3>(physical.at("seedLPS"),"independent physical seed");
    auto crop_origin=pi::numbers<3>(physical.at("cropGrid").at("originLPS"),"crop origin");
    auto crop_direction=pi::numbers<9>(physical.at("cropGrid").at("directionLPS"),"crop direction");
    for(unsigned r=0;r<3;++r){double center=crop_origin[r];for(unsigned a=0;a<3;++a)center+=crop_direction[r*3+a]*(n/2)*step;need(std::abs(center-seed_lps[r])<1e-6,"export crop is not centered on unchanged independent seed");}
    // The original CT interpolator extrapolates using actual first/last plane
    // intervals. This regular-export adapter requires matching edge intervals.
    const auto& offsets=scan.at("dicom").at("slice_offsets_mm");need(offsets.is_array()&&offsets.size()>=2,"source slice positions required");
    need(std::abs(offsets.at(1).get<double>()-offsets.at(0).get<double>()-source_spacing[2])<1e-6&&
         std::abs(offsets.back().get<double>()-offsets.at(offsets.size()-2).get<double>()-source_spacing[2])<1e-6&&
         std::abs(offsets.back().get<double>()-physical.at("source_last_plane_offset_mm").get<double>())<1e-6&&
         std::abs(physical.at("last_regular_plane_offset_mm").get<double>()-physical.at("source_last_plane_offset_mm").get<double>())<1e-6,
         "source/export physical edge support differs; unsupported irregular boundary intervals");
    cases.push_back({scan,item,metadata,image_path,metadata_path,g,shape});next_row+=per_scan;
  }
  need(next_row==rows&&images.empty()&&geometry_cases.empty(),"incomplete reference scan row coverage");
  fs::create_directories(root);auto values=output(root/"values.f32");
  J result=reference;result["values_path"]=(root/"values.f32").string();result["feature_names"]=normalized_names();result["feature_operator_names"]=reference.at("feature_names");result["scans"]=J::array();
  for(const auto& [key,path]:reused)result[key]=path.string();
  result["domain_description"]="Original fixed physical crop about independent supplied location; released normalized prepared image intensities; unchanged binary reference";
  result["resampling"]="Original XYZ physical crop -> original export grid -> official ZYX transpose -> crop bbox -> rounded resize shape with half-pixel centers; CUDA trilinear saved normalized image sampling with edge replication; normalized original air weighted on original physical source bounds";
  result["feature_input_provenance"]={{"mode","prepared-normalized-image-20"},{"prepared_images_index",index_path.string()},{"reference_dataset",reference_path.string()},{"export_cases",export_path.string()},
    {"array_axis_order","CZYX"},{"transpose_forward",transpose},{"image_representation",index.at("image_representation")},
    {"intensity_normalization","saved values preserved; no HU conversion or renormalization"},{"feature_kernel","same twenty-feature CUDA kernel as original preparation"},
    {"outside_original_image_HU",-1000.},{"outside_normalized_intensity",cases.front().geometry.outside},
    {"outside_convention","Original physical trilinear air-support weights retained; saved image edge replicated after half-pixel mapping; air transformed through released float32 CTNormalization clipping/mean/std"},
    {"cropped_away_source_interior","unsupported: preparation fails if requested original source support was removed by crop bbox"},
    {"normalization_intensity_properties",plans.at("foreground_intensity_properties_per_channel")},
    {"prepared_reference_labels_used",false},{"reference_labels_used_for_features_or_seed",false},{"reference_arrays_reused_without_rewriting",true},
    {"image_sha256_verification","not recomputed; recorded index hashes retained as declared provenance"}};
  if(options.contains("--preprocessing-plans"))result["feature_input_provenance"]["preprocessing_plans"]=options.at("--preprocessing-plans").string();
  if(index.contains("source_code_commit"))result["feature_input_provenance"]["preprocessing_source_code_commit"]=index.at("source_code_commit");
  U eligible_total=0;
  for(const auto& c:cases){
    auto array=pi::read_npy(c.image);need(array.shape==c.shape,"NPY/header/metadata CZYX shape mismatch");
    U start=c.original.at("row_start").get<U>();auto truth=reference_segment(reused.at("labels_path"),start,per_scan),expected_eligible=reference_segment(reused.at("train_eligible_path"),start,per_scan);
    auto patients=reference_segment(reused.at("patient_ids_path"),start,per_scan),scans=reference_segment(reused.at("scan_ids_path"),start,per_scan);
    auto patient=c.original.at("patient_id").get<std::uint32_t>(),scan=c.original.at("scan_id").get<std::uint32_t>();
    for(U q=0;q<per_scan;++q)need(truth[q]<=1&&expected_eligible[q]<=1&&patients[q]==patient&&scans[q]==scan,"invalid reference binary/group arrays");
    Device<float>image(array.values.size()),intensity(per_scan),x(per_scan*20);image.upload(array.values.data());
    Device<std::uint32_t>y(per_scan),eligible(per_scan);y.upload(truth.data());Device<unsigned long long>totals(3);ck(cudaMemset(totals.p,0,3*sizeof(unsigned long long)));
    Device<unsigned>invalid(1);ck(cudaMemset(invalid.p,0,sizeof(unsigned)));
    unsigned blocks=unsigned(std::min<U>(4096,(per_scan+255)/256));
    resample_prepared<<<blocks,256>>>(image.p,intensity.p,c.geometry,invalid.p);ck(cudaGetLastError());
    Geometry feature_grid{};feature_grid.n=n;feature_grid.step=step;
    features<<<blocks,256>>>(intensity.p,y.p,x.p,eligible.p,totals.p,feature_grid,stride);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    unsigned bad=0;invalid.download(&bad);need(!(bad&2),"requested original source interior was cropped away by prepared image bbox");need(!(bad&1),"nonfinite prepared image intensity in feature crop");
    std::vector<float>hx(per_scan*20);std::vector<std::uint32_t>he(per_scan);std::array<unsigned long long,3>stats;
    x.download(hx.data());eligible.download(he.data());totals.download(stats.data());need(he==expected_eligible,"generated/reference training eligibility mismatch");
    need(stats[0]==c.original.at("truth_positive_rows").get<U>()&&stats[1]==c.original.at("training_rows").get<U>()&&stats[2]==c.original.at("training_positive_rows").get<U>(),"original reference counts changed");
    append(values,hx);J detail=c.original;
    detail["feature_image"]={{"case",c.input.at("case")},{"image",c.image.string()},{"image_metadata",c.metadata_path.string()},{"shape_CZYX",c.shape},
      {"target_spacing_ZYX",c.metadata.at("target_spacing_ZYX")},{"transpose_forward",transpose},{"crop_bbox",c.metadata.at("properties").at("bbox_used_for_cropping")},
      {"output_XYZ_to_prepared_ZYX_affine",J::array({J::array({c.geometry.map[0][0],c.geometry.map[0][1],c.geometry.map[0][2],c.geometry.map[0][3]}),J::array({c.geometry.map[1][0],c.geometry.map[1][1],c.geometry.map[1][2],c.geometry.map[1][3]}),J::array({c.geometry.map[2][0],c.geometry.map[2][1],c.geometry.map[2][2],c.geometry.map[2][3]})})}};
    if(c.input.contains("image_sha256"))detail["feature_image"]["declared_image_sha256"]=c.input.at("image_sha256");
    result["scans"].push_back(detail);eligible_total+=stats[1];
    std::cout<<J{{"event","prepared_image_features"},{"patient_id",patient},{"scan_id",scan},{"rows",per_scan},{"positive",stats[0]},{"positive_outside",c.original.at("truth_positive_outside_rows")},{"training_positive",stats[2]}}.dump()<<std::endl;
  }
  need(eligible_total==reference.at("train_eligible_rows").get<U>(),"total reference eligibility changed");values.close();need(bool(values),"feature dataset close failed");
  result["complete"]=true;auto descriptor=output(root/"dataset.json");descriptor<<result.dump(2)<<'\n';descriptor.close();need(bool(descriptor),"descriptor write failed");
  std::cout<<J{{"event","complete"},{"dataset",(root/"dataset.json").string()},{"rows",rows},{"features",20}}.dump()<<std::endl;return 0;
}
int self_test(){
  prepared_host_test();prepared_gpu_test();
  Geometry g{8,8,8,4,1,1,1,4,4,4,1};
  g.labelNz=8;g.labelSpacing=1;
  std::vector<std::int16_t>raw(512,55);std::vector<float>slope(8,2),intercept(8,-10);
  std::vector<std::uint8_t>mask(512,0);mask[0]=1;mask[(4*8+4)*8+4]=1;
  Device<std::int16_t>dr(512);dr.upload(raw.data());Device<float>ds(8),di(8),hu(64),x(64*20);ds.upload(slope.data());di.upload(intercept.data());
  Device<std::uint8_t>dm(512);dm.upload(mask.data());Device<std::uint32_t>y(64),eligible(64);Device<unsigned long long>stats(4);ck(cudaMemset(stats.p,0,4*sizeof(unsigned long long)));
  resample<<<1,128>>>(dr.p,ds.p,di.p,dm.p,hu.p,y.p,g);features<<<1,128>>>(hu.p,y.p,x.p,eligible.p,stats.p,g,1);
  outside_reference<<<4,128>>>(dm.p,g,make_int3(-3,-3,-3),make_int3(11,11,11),stats.p+3);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
  std::vector<float>result(64*20);std::vector<std::uint32_t>truth(64);std::array<unsigned long long,4>counts;x.download(result.data());y.download(truth.data());stats.download(counts.data());
  need(counts==std::array<unsigned long long,4>{1,64,1,1},"CUDA reference/eligibility accounting failed");
  for(unsigned i=0;i<64;i++){
    const float*f=result.data()+i*20;need(f[0]==100&&f[1]==100&&f[2]==0&&f[3]==100&&f[4]==0,"CUDA HU/local-statistic calculation failed");
    need(f[5]==0&&f[6]==0&&f[7]==0&&f[8]==0&&f[19]==100,"CUDA gradient/seed calculation failed");
    need(truth[i]==(i==42?1u:0u),"CUDA seed alignment failed");
  }
  std::vector<double>positions{0,1,2,3,7,8,9,10};Device<double>zp(8);zp.upload(positions.data());
  g.zpositions=zp.p;g.cz=7;g.labelPositions=zp.p;
  for(unsigned z=0;z<8;z++)for(unsigned xy=0;xy<64;xy++)raw[z*64+xy]=std::int16_t(positions[z]*10);
  std::fill(slope.begin(),slope.end(),1);std::fill(intercept.begin(),intercept.end(),0);dr.upload(raw.data());ds.upload(slope.data());di.upload(intercept.data());
  resample<<<1,128>>>(dr.p,ds.p,di.p,dm.p,hu.p,y.p,g);ck(cudaGetLastError());ck(cudaDeviceSynchronize());std::vector<float>ramp(64);hu.download(ramp.data());y.download(truth.data());
  for(unsigned q=0;q<64;q++)need(std::abs(ramp[q]-float((5+q/16)*10))<1e-5f,"CUDA actual slice-position interpolation failed");
  need(truth[42]==1,"CUDA irregular-grid seed/reference alignment failed");
  // Independent, sparse SEG planes must not use CT indices or fill omitted frames.
  std::vector<double>labelPositions{2,4,8};Device<double>lp(3);lp.upload(labelPositions.data());
  std::vector<std::uint8_t>sparseMask(3*64,1);std::fill_n(sparseMask.begin(),64,0);
  Device<std::uint8_t>sm(sparseMask.size());sm.upload(sparseMask.data());
  g.labelNz=3;g.labelPositions=lp.p;g.labelSpacing=2;
  resample<<<1,128>>>(dr.p,ds.p,di.p,sm.p,hu.p,y.p,g);ck(cudaGetLastError());ck(cudaDeviceSynchronize());y.download(truth.data());
  for(unsigned q=0;q<64;q++)need(truth[q]==(q/16==1?0u:1u),"CUDA independent sparse SEG grid failed");
  std::cout<<"CUDA segmentation preparation checks passed\n";return 0;
}
}
int main(int argc,char**argv){try{
  if(argc==2&&std::string(argv[1])=="--self-test")return self_test();
  if(argc==2&&std::string(argv[1])=="--self-test-host")return prepared_host_test();
  if(std::any_of(argv+1,argv+argc,[](const char* value){return std::string(value)=="--additional-channels";}))return prepare_channels(argc,argv);
  if(argc>1&&std::string(argv[1]).starts_with("--"))return prepare_images(argc,argv);
  need(argc==3,"usage: class_segmentation_prepare MANIFEST_JSON FRESH_OUTPUT_DIRECTORY");
  std::ifstream input(argv[1]);need(bool(input),"cannot read manifest");J manifest;input>>manifest;
  need(manifest.at("scans").is_array()&&!manifest.at("scans").empty(),"manifest needs scans");
  fs::path root=fs::absolute(argv[2]);need(!fs::exists(root),"output must be fresh");
  double width=manifest.value("crop_width_mm",160.0),step=manifest.value("grid_spacing_mm",2.0);
  need(std::isfinite(width)&&std::isfinite(step)&&width>0&&step>0,"positive finite geometry required");
  double ratio=width/step;need(ratio>=4&&ratio<=256&&std::abs(ratio-std::round(ratio))<1e-9,"width/spacing must be an integer from4 to256");
  int n=int(std::round(ratio));need(n%2==0,"crop grid must have even side length");
  auto stride=manifest.value("training_sample_stride",32u);need(stride>0,"positive sampling stride required");
  fs::create_directories(root);
  auto xf=output(root/"values.f32"),yf=output(root/"labels.u32"),pf=output(root/"patient_ids.u32"),sf=output(root/"scan_ids.u32"),ef=output(root/"train_eligible.u32");
  J result={{"format","grouped-segmentation-dataset-1"},{"classes",2},{"features",20},{"rows",0},
    {"values_path",(root/"values.f32").string()},{"labels_path",(root/"labels.u32").string()},
    {"patient_ids_path",(root/"patient_ids.u32").string()},{"scan_ids_path",(root/"scan_ids.u32").string()},
    {"train_eligible_path",(root/"train_eligible.u32").string()},{"feature_names",names()},
    {"crop_width_mm",width},{"grid_spacing_mm",step},{"feature_geometry",{{"local_statistic_cube_radii_mm",J::array({step,2*step})},{"offset_distance_mm",2*step},{"gradient_step_mm",step}}},{"training_sample_stride",stride},
    {"domain_description","Fixed physical crop about supplied lesion location; fractional HU resampling; binary reference"},{"task","location-assisted target-lesion segmentation; supplied seed, not autonomous detection"},
    {"resampling","bilinear in-plane/linear actual-plane HU; nearest physical-plane binary reference; crop aligned to CT axes"},
    {"scans",J::array()},{"manifest",manifest}};
  U rows=0,eligible_rows=0;
  for(const auto&s:manifest.at("scans")){
    auto volume=class_study::read_segmentation_scan(s);U voxels=U(volume.nx)*volume.ny*volume.nz,count=U(n)*n*n;
    Device<double>zpositions(volume.nz);zpositions.upload(volume.sliceOffsetsMM.data());
    need(volume.labelNz>0&&volume.labels.size()==U(volume.nx)*volume.ny*volume.labelNz,"invalid SEG buffer extent");
    Device<double>labelPositions(volume.labelNz);labelPositions.upload(volume.labelSliceOffsetsMM.data());
    Geometry g{int(volume.nx),int(volume.ny),int(volume.nz),n,volume.spacingXYZ[0],volume.spacingXYZ[1],volume.spacingXYZ[2],volume.seedGridXYZ[0],volume.seedGridXYZ[1],volume.seedProjectionOffsetMM,step,zpositions.p};
    g.labelNz=int(volume.labelNz);g.labelPositions=labelPositions.p;g.labelSpacing=volume.labelSliceSpacingMM;
    Device<std::int16_t>raw(voxels);raw.upload(volume.rawPixels.data());Device<float>slopes(g.nz),intercepts(g.nz);slopes.upload(volume.perSliceSlope.data());intercepts.upload(volume.perSliceIntercept.data());
    Device<std::uint8_t>labels(U(volume.nx)*volume.ny*volume.labelNz);labels.upload(volume.labels.data());
    Device<float>hu(count),x(count*20);Device<std::uint32_t>y(count),eligible(count);Device<unsigned long long>totals(4);ck(cudaMemset(totals.p,0,4*sizeof(unsigned long long)));
    resample<<<std::min<U>(4096,(count+255)/256),256>>>(raw.p,slopes.p,intercepts.p,labels.p,hu.p,y.p,g);ck(cudaGetLastError());
    features<<<std::min<U>(4096,(count+255)/256),256>>>(hu.p,y.p,x.p,eligible.p,totals.p,g,stride);ck(cudaGetLastError());
    // Output-grid coordinates spanning every source voxel center, with a
    // one-grid-point halo to include nearest-neighbor boundary occupancy.
    auto bounds=[&](double c,double spacing,int dim){int lo=int(std::floor(n/2+(-0.5-c)*spacing/step))-1,hi=int(std::ceil(n/2+(dim-0.5-c)*spacing/step))+1;return std::pair(lo,hi-lo+1);};
    auto bx=bounds(g.cx,g.sx,g.nx),by=bounds(g.cy,g.sy,g.ny);
    int zlo=int(std::floor(n/2+(volume.labelSliceOffsetsMM.front()-0.5*g.labelSpacing-g.cz)/step))-1,zhi=int(std::ceil(n/2+(volume.labelSliceOffsetsMM.back()+0.5*g.labelSpacing-g.cz)/step))+1;
    auto bz=std::pair(zlo,zhi-zlo+1);
    outside_reference<<<4096,256>>>(labels.p,g,make_int3(bx.first,by.first,bz.first),make_int3(bx.second,by.second,bz.second),totals.p+3);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<float>hx(count*20);std::vector<std::uint32_t>hy(count),he(count),group(count,s.at("patient_id").get<std::uint32_t>()),scan(count,s.at("scan_id").get<std::uint32_t>());std::array<unsigned long long,4>stats;
    x.download(hx.data());y.download(hy.data());eligible.download(he.data());totals.download(stats.data());
    append(xf,hx);append(yf,hy);append(pf,group);append(sf,scan);append(ef,he);
    J detail=s;detail["row_start"]=rows;detail["rows"]=count;detail["truth_positive_rows"]=stats[0];detail["training_rows"]=stats[1];detail["training_positive_rows"]=stats[2];detail["truth_positive_outside_rows"]=stats[3];detail["dicom"]=volume.metadata;
    result["scans"].push_back(detail);rows+=count;eligible_rows+=stats[1];
    std::cout<<J{{"event","scan_prepared"},{"patient_id",s.at("patient_id")},{"scan_id",s.at("scan_id")},{"rows",count},{"positive",stats[0]},{"positive_outside",stats[3]},{"training_positive",stats[2]}}.dump()<<std::endl;
  }
  result["rows"]=rows;result["train_eligible_rows"]=eligible_rows;xf.close();yf.close();pf.close();sf.close();ef.close();
  need(bool(xf)&&bool(yf)&&bool(pf)&&bool(sf)&&bool(ef),"dataset close failed");
  auto descriptor=output(root/"dataset.json");descriptor<<result.dump(2)<<'\n';descriptor.close();need(bool(descriptor),"descriptor write failed");
  std::cout<<J{{"event","complete"},{"dataset",(root/"dataset.json").string()},{"rows",rows}}.dump()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<"segmentation preparation failed: "<<e.what()<<'\n';return 1;}}
