#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
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
#include <string>
#include <vector>

// Host code transports metadata and exact binary image payloads. CUDA validates
// mask values, resamples by physical nearest neighbor, and computes all metrics.
// The matched endpoint reuses every prepared reference row and outside-crop FN.
namespace {
using J=nlohmann::json; namespace fs=std::filesystem;
using U=std::uint64_t; using L=unsigned long long;
void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void sync(){ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
unsigned blocks(U n){return unsigned(std::min<U>(4096,(n+255)/256));}
template<class T>struct Device {
  T* p=nullptr;std::size_t n;
  explicit Device(std::size_t count):n(count){need(n>0&&n<=SIZE_MAX/sizeof(T),"invalid device extent");ck(cudaMalloc(&p,n*sizeof(T)));}
  ~Device(){if(p)cudaFree(p);}
  Device(const Device&)=delete;Device&operator=(const Device&)=delete;
  void set(const T* h){ck(cudaMemcpy(p,h,n*sizeof(T),cudaMemcpyHostToDevice));}
  void set(const std::vector<T>& h){need(h.size()==n,"upload extent differs");set(h.data());}
  void zero(){ck(cudaMemset(p,0,n*sizeof(T)));}
  std::vector<T> get(){std::vector<T> h(n);ck(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
};
struct Grid {
  int size[3]{};double spacing[3]{},origin[3]{},direction[9]{};
  U count()const{return U(size[0])*size[1]*size[2];}
};
std::string trim(std::string s){auto lo=s.find_first_not_of(" \t\r\n"),hi=s.find_last_not_of(" \t\r\n");return lo==std::string::npos?std::string{}:s.substr(lo,hi-lo+1);}
J json_file(const fs::path& path){std::ifstream f(path);need(bool(f),"cannot read JSON: "+path.string());J j;f>>j;return j;}
U integer(const J& j,const char* key){const auto& v=j.at(key);need(v.is_number_unsigned()||(v.is_number_integer()&&v.get<std::int64_t>()>=0),std::string("nonnegative integer required: ")+key);return v.get<U>();}
void validate_grid(const Grid& g){
  U count=1;for(int a=0;a<3;++a){need(g.size[a]>0&&g.size[a]<=1048576,"invalid image dimension");need(count<=UINT32_MAX/U(g.size[a]),"image voxel count exceeds supported extent");count*=g.size[a];need(std::isfinite(g.spacing[a])&&g.spacing[a]>0&&std::isfinite(g.origin[a]),"invalid physical image geometry");}
  for(double d:g.direction)need(std::isfinite(d),"nonfinite direction matrix");
  // Inverse equals transpose only for an orthonormal direction matrix. Either
  // handedness is legal for an external image; no identity direction is assumed.
  for(int a=0;a<3;++a)for(int b=0;b<3;++b){double v=0;for(int r=0;r<3;++r)v+=g.direction[r*3+a]*g.direction[r*3+b];need(std::abs(v-(a==b?1.:0.))<=1e-5,"nonorthonormal image direction");}
}
Grid grid_json(const J& j){
  Grid g;for(const auto& key:{"sizeXYZ","spacingXYZ","originLPS","directionLPS"})need(j.at(key).is_array(),std::string("geometry array required: ")+key);
  need(j.at("sizeXYZ").size()==3&&j.at("spacingXYZ").size()==3&&j.at("originLPS").size()==3&&j.at("directionLPS").size()==9,"image geometry array length differs");
  for(int a=0;a<3;++a){const auto& n=j.at("sizeXYZ")[a];need(n.is_number_integer()&&n.get<std::int64_t>()>0&&n.get<std::int64_t>()<=1048576,"invalid geometry dimension");g.size[a]=n.get<int>();g.spacing[a]=j.at("spacingXYZ")[a].get<double>();g.origin[a]=j.at("originLPS")[a].get<double>();}
  for(int a=0;a<9;++a)g.direction[a]=j.at("directionLPS")[a].get<double>();validate_grid(g);return g;
}
J grid_report(const Grid& g){return{{"sizeXYZ",J::array({g.size[0],g.size[1],g.size[2]})},{"spacingXYZ",J::array({g.spacing[0],g.spacing[1],g.spacing[2]})},{"originLPS",J::array({g.origin[0],g.origin[1],g.origin[2]})},{"directionLPS",std::vector<double>(g.direction,g.direction+9)}};}
void same_grid(const Grid& a,const Grid& b,const std::string& label){for(int i=0;i<3;++i)need(a.size[i]==b.size[i]&&std::abs(a.spacing[i]-b.spacing[i])<=1e-5&&std::abs(a.origin[i]-b.origin[i])<=1e-5,label+": prediction physical geometry differs from exported input");for(int i=0;i<9;++i)need(std::abs(a.direction[i]-b.direction[i])<=1e-5,label+": prediction direction differs from exported input");}
template<class T>std::vector<T> numbers(const std::string& text,std::size_t n){std::istringstream s(text);std::vector<T> v(n);for(auto& x:v)need(bool(s>>x),"invalid MetaImage numeric field");std::string extra;need(!(s>>extra),"extra MetaImage numeric field");return v;}
bool boolean(const std::string& s){need(s=="True"||s=="False"||s=="true"||s=="false"||s=="1"||s=="0","invalid MetaImage Boolean");return s=="True"||s=="true"||s=="1";}
struct Image {Grid grid;std::string type;std::vector<std::uint8_t> bytes;};
Image read_mha(const fs::path& path){
  need(path.extension()==".mha","prediction must be a .mha image");std::ifstream f(path,std::ios::binary);need(bool(f),"missing/unreadable prediction: "+path.string());
  std::map<std::string,std::string> h;std::string line;bool payload=false;
  for(unsigned n=0;n<1024&&std::getline(f,line);++n){need(line.size()<=16384,"MetaImage header line too long");line=trim(line);if(line.empty()||line[0]=='#')continue;auto e=line.find('=');need(e!=std::string::npos,"invalid MetaImage header line");auto key=trim(line.substr(0,e)),value=trim(line.substr(e+1));need(!key.empty()&&h.emplace(key,value).second,"duplicate MetaImage header field");if(key=="ElementDataFile"){need(value=="LOCAL","only self-contained LOCAL MetaImage predictions are supported");payload=true;break;}}
  need(payload,"MetaImage has no LOCAL binary payload");auto field=[&](const char* key)->const std::string&{auto it=h.find(key);need(it!=h.end(),std::string("missing MetaImage field: ")+key);return it->second;};
  need(numbers<int>(field("NDims"),1)[0]==3,"prediction must be three-dimensional");need(boolean(field("BinaryData")),"text MetaImage predictions are unsupported");
  if(h.contains("ObjectType"))need(h.at("ObjectType")=="Image","MetaImage ObjectType must be Image");
  if(h.contains("CompressedData"))need(!boolean(h.at("CompressedData")),"compressed MetaImage predictions must be converted to uncompressed I/O artifacts");
  if(h.contains("ElementNumberOfChannels"))need(numbers<int>(h.at("ElementNumberOfChannels"),1)[0]==1,"prediction must have one channel");
  if(h.contains("HeaderSize"))need(numbers<int>(h.at("HeaderSize"),1)[0]==0,"unsupported MetaImage HeaderSize");
  bool msb=false;if(h.contains("BinaryDataByteOrderMSB"))msb=boolean(h.at("BinaryDataByteOrderMSB"));if(h.contains("ElementByteOrderMSB")){bool alternate=boolean(h.at("ElementByteOrderMSB"));need(!h.contains("BinaryDataByteOrderMSB")||alternate==msb,"conflicting MetaImage byte order");msb=alternate;}need(!msb&&std::endian::native==std::endian::little,"only native little-endian mask transport is supported");
  Image image;auto size=numbers<int>(field("DimSize"),3);auto spacing=numbers<double>(field("ElementSpacing"),3);auto direction=numbers<double>(field("TransformMatrix"),9);
  unsigned origins=0;std::string origin;for(const auto& key:{"Offset","Origin","Position"})if(h.contains(key)){++origins;origin=h.at(key);}need(origins==1,"exactly one MetaImage physical origin is required");auto pos=numbers<double>(origin,3);
  for(int a=0;a<3;++a){image.grid.size[a]=size[a];image.grid.spacing[a]=spacing[a];image.grid.origin[a]=pos[a];}// MetaImage lists each image-axis direction consecutively (column major).
  // Internal JSON/CUDA affine matrices remain physical row-major matrices.
  for(int r=0;r<3;++r)for(int a=0;a<3;++a)image.grid.direction[r*3+a]=direction[a*3+r];validate_grid(image.grid);
  image.type=field("ElementType");static const std::map<std::string,U> widths{{"MET_CHAR",1},{"MET_UCHAR",1},{"MET_SHORT",2},{"MET_USHORT",2},{"MET_INT",4},{"MET_UINT",4},{"MET_FLOAT",4},{"MET_DOUBLE",8}};need(widths.contains(image.type),"unsupported MetaImage label element type");
  U bytes=image.grid.count()*widths.at(image.type);need(bytes<=SIZE_MAX&&bytes<=U(std::numeric_limits<std::streamsize>::max()),"mask payload too large");auto start=f.tellg();need(start>=0,"invalid MetaImage payload offset");f.seekg(0,std::ios::end);auto end=f.tellg();need(end>=start&&U(end-start)==bytes,"MetaImage payload size differs (empty, truncated or trailing data)");f.seekg(start);image.bytes.resize(std::size_t(bytes));f.read(reinterpret_cast<char*>(image.bytes.data()),std::streamsize(bytes));need(bool(f),"cannot read mask payload");return image;
}
template<class T>__global__ void decode(const T* input,std::uint8_t* mask,U n,L* positive,unsigned* bad){for(U q=U(blockIdx.x)*blockDim.x+threadIdx.x;q<n;q+=U(blockDim.x)*gridDim.x){T v=input[q];if(v!=T(0)&&v!=T(1)){atomicExch(bad,1u);mask[q]=0;}else{mask[q]=std::uint8_t(v);if(v==T(1))atomicAdd(positive,1ULL);}}}
void launch_decode(const Image& image,const Device<std::uint8_t>& bytes,Device<std::uint8_t>& mask,Device<L>& positive,Device<unsigned>& bad){
#define MASK_TYPE(name,T) if(image.type==name){decode<<<blocks(mask.n),256>>>(reinterpret_cast<const T*>(bytes.p),mask.p,mask.n,positive.p,bad.p);return;}
  MASK_TYPE("MET_CHAR",std::int8_t) MASK_TYPE("MET_UCHAR",std::uint8_t) MASK_TYPE("MET_SHORT",std::int16_t) MASK_TYPE("MET_USHORT",std::uint16_t) MASK_TYPE("MET_INT",std::int32_t) MASK_TYPE("MET_UINT",std::uint32_t) MASK_TYPE("MET_FLOAT",float) MASK_TYPE("MET_DOUBLE",double)
#undef MASK_TYPE
  need(false,"unreachable mask element type");
}
__device__ unsigned sample(const std::uint8_t* mask,Grid source,Grid crop,int x,int y,int z){
  double physical[3];int c[3]{x,y,z};for(int r=0;r<3;++r){physical[r]=crop.origin[r];for(int a=0;a<3;++a)physical[r]+=crop.direction[r*3+a]*crop.spacing[a]*c[a];}
  int index[3];for(int a=0;a<3;++a){double d=0;for(int r=0;r<3;++r)d+=source.direction[r*3+a]*(physical[r]-source.origin[r]);d/=source.spacing[a];if(d<-.5||d>=source.size[a]-.5)return 0;index[a]=int(floor(d+.5));if(index[a]<0||index[a]>=source.size[a])return 0;}
  return mask[(U(index[2])*source.size[1]+index[1])*source.size[0]+index[0]];
}
__global__ void confusion(const std::uint8_t* prediction,Grid source,Grid crop,const std::uint32_t* truth,L* counts,unsigned* bad){
  U n=U(crop.size[0])*crop.size[1]*crop.size[2];for(U q=U(blockIdx.x)*blockDim.x+threadIdx.x;q<n;q+=U(blockDim.x)*gridDim.x){if(truth[q]>1){atomicExch(bad,1u);continue;}int x=q%crop.size[0],y=(q/crop.size[0])%crop.size[1],z=q/(U(crop.size[0])*crop.size[1]);unsigned p=sample(prediction,source,crop,x,y,z),t=truth[q];atomicAdd(counts+(p?(t?0:1):(t?2:3)),1ULL);}
}
__global__ void summarize(const L* counts,const L* outside,const std::uint32_t* patient,const L* native_positive,unsigned scans,unsigned patients,L* pooled,double* patient_dice,double* scan_dice,L* summary,double* macro){
  if(blockIdx.x||threadIdx.x)return;
  for(unsigned p=0;p<patients;++p){L* a=pooled+U(p)*4;for(unsigned s=0;s<scans;++s)if(patient[s]==p){const L* c=counts+U(s)*4;for(unsigned k=0;k<4;++k)a[k]+=c[k];a[2]+=outside[s];}L denominator=2*a[0]+a[1]+a[2];patient_dice[p]=denominator?2*double(a[0])/double(denominator):1.;*macro+=patient_dice[p]/patients;}
  for(unsigned s=0;s<scans;++s){const L* c=counts+U(s)*4;for(unsigned k=0;k<4;++k)summary[k]+=c[k];summary[4]+=outside[s];summary[5]+=native_positive[s]==0;summary[6]+=c[0]+c[1]==0;L d=2*c[0]+c[1]+c[2]+outside[s];scan_dice[s]=d?2*double(c[0])/double(d):1.;}
}
std::vector<std::uint32_t> read_truth(std::ifstream& f,U start,U count){need(start<=U(std::numeric_limits<std::streamoff>::max())/4&&count<=SIZE_MAX/4,"truth extent exceeds supported transport");f.seekg(std::streamoff(start*4));need(bool(f),"cannot seek prepared reference labels");std::vector<std::uint32_t> y(count);f.read(reinterpret_cast<char*>(y.data()),std::streamsize(count*4));need(bool(f),"cannot read prepared reference labels");return y;}
void write_json(const fs::path& path,const J& j){std::ofstream f(path);need(bool(f),"cannot create scoring report");f<<j.dump(2)<<'\n';f.close();need(bool(f),"scoring report write failed");}
J prompt_provenance(const J& c){J value=c.value("primary_prompt_uses_reference_mask",J(nullptr));need(value.is_null()||value.is_boolean(),"primary_prompt_uses_reference_mask must be Boolean or null");return value;}
void compare_prepared(const J& c,const J& s){
  (void)prompt_provenance(c);
  need(integer(c,"scan_id")==integer(s,"scan_id")&&integer(c,"patient_id")==integer(s,"patient_id"),"export and prepared scan/patient mapping differs");
  need(c.at("sourceScan").is_object()&&!c.at("sourceScan").empty(),"export source scan descriptor is required");need(integer(c.at("sourceScan"),"scan_id")==integer(s,"scan_id")&&integer(c.at("sourceScan"),"patient_id")==integer(s,"patient_id"),"export source descriptor identity differs");
  for(const auto& [key,value]:c.at("sourceScan").items())need(s.contains(key)&&s.at(key)==value,"export/source scan differs: "+key);
  for(const auto& key:{"row_start","rows","truth_positive_rows","truth_positive_outside_rows"})need(integer(c.at("preparedScan"),key)==integer(s,key),std::string("export/prepared metadata differs: ")+key);
}
void validate_crop(const J& c,const Grid& full,const Grid& crop,int side,double step){
  need(c.at("seedLPS").is_array()&&c.at("seedLPS").size()==3&&c.at("seedGridXYZ").is_array()&&c.at("seedGridXYZ").size()==3,"export physical and grid seed coordinates are required");
  for(int a=0;a<3;++a)need(crop.size[a]==side&&std::abs(crop.spacing[a]-step)<=1e-9,"scoring crop differs from prepared reference dimensions/spacing");
  for(int i=0;i<9;++i)need(std::abs(crop.direction[i]-full.direction[i])<=1e-5,"crop axes differ from exported source image axes");
  for(int r=0;r<3;++r){double seed=c.at("seedLPS")[r].get<double>();need(std::isfinite(seed),"nonfinite exported physical seed");double physical=full.origin[r],center=crop.origin[r];for(int a=0;a<3;++a){double coordinate=c.at("seedGridXYZ")[a].get<double>();need(std::isfinite(coordinate),"nonfinite exported grid seed");physical+=full.direction[r*3+a]*full.spacing[a]*coordinate;center+=crop.direction[r*3+a]*(side/2)*step;}need(std::abs(physical-seed)<=1e-5,"export physical seed differs from full image affine/grid seed");need(std::abs(center-seed)<=1e-5,"crop center differs from exported supplied seed");}
}
void normalize_source_paths(J& d,const fs::path& base){
  auto normalize=[&](J& scans){need(scans.is_array(),"source scan array is required");for(auto& scan:scans)for(const char* key:{"ct_dir","seg_path"})if(scan.contains(key)){fs::path path=scan.at(key).get<std::string>();if(path.is_relative())scan[key]=(base/path).lexically_normal().string();}};
  normalize(d.at("scans"));normalize(d.at("manifest").at("scans"));
}
int score(const fs::path& dataset_path,const fs::path& export_path,const fs::path& predictions,J& report){
  report["dataset_path"]=dataset_path.string();report["export_manifest_path"]=export_path.string();report["predictions_directory"]=predictions.string();auto d=json_file(dataset_path),e=json_file(export_path);normalize_source_paths(d,dataset_path.parent_path());need(d.at("format")=="grouped-segmentation-dataset-1"&&d.at("classes")==2,"unsupported prepared segmentation reference");double width=d.at("crop_width_mm").get<double>(),step=d.at("grid_spacing_mm").get<double>();need(std::isfinite(width)&&std::isfinite(step)&&width>0&&step>0,"invalid prepared crop geometry");double ratio=width/step;need(ratio>=4&&ratio<=256&&std::abs(ratio-std::round(ratio))<1e-9,"prepared crop width/spacing must yield an integer side from4 to256");int side=int(std::round(ratio));need(side%2==0,"prepared crop side must be even");U rows_per_scan=U(side)*side*side;need(e.at("complete")==true,"incomplete exported input manifest");need(e.at("source_manifest")==d.at("manifest"),"export source manifest differs from prepared reference manifest");need(d.at("scans").is_array()&&e.at("cases").is_array(),"scan/case arrays required");need(!d.at("scans").empty()&&d.at("scans").size()<=UINT32_MAX&&e.at("cases").size()==d.at("scans").size(),"matched endpoint requires every declared scan exactly once");report["endpoint"]="matched prepared physical crop; all truth outside crop remains false negative";report["crop_width_mm"]=width;report["grid_spacing_mm"]=step;report["crop_side"]=side;
  std::map<U,J> scan_map,case_map;std::set<U> patient_ids;std::set<std::string> names;U next=0;
  for(const auto& s:d.at("scans")){U sid=integer(s,"scan_id"),pid=integer(s,"patient_id");need(sid<=UINT32_MAX&&pid<=UINT32_MAX&&scan_map.emplace(sid,s).second,"duplicate/invalid prepared scan ID");patient_ids.insert(pid);need(integer(s,"row_start")==next&&integer(s,"rows")==rows_per_scan,"prepared scan rows must remain contiguous declared crop blocks");next+=integer(s,"rows");need(integer(s,"truth_positive_rows")+integer(s,"truth_positive_outside_rows")>0,"prepared scan has no foreground reference");}
  need(!patient_ids.empty()&&integer(d,"rows")==next,"prepared patient/row extent differs");
  need(fs::is_directory(predictions),"prediction directory is missing");
  for(const auto& c:e.at("cases")){U sid=integer(c,"scan_id");need(scan_map.contains(sid)&&case_map.emplace(sid,c).second,"duplicate/unknown exported scan ID");compare_prepared(c,scan_map.at(sid));std::string name=c.at("case").get<std::string>();need(!name.empty()&&name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos&&names.insert(name).second,"invalid/duplicate prediction case basename");auto full=grid_json(c),crop=grid_json(c.at("cropGrid"));validate_crop(c,full,crop,side,step);auto file=predictions/(name+".mha");need(fs::is_regular_file(file)&&fs::file_size(file)>0,"missing/empty prediction file: "+file.string());}
  fs::path labels_path=d.at("labels_path").get<std::string>();if(labels_path.is_relative())labels_path=dataset_path.parent_path()/labels_path;report["resolved_labels_path"]=labels_path.string();std::ifstream reference(labels_path,std::ios::binary);need(bool(reference),"prepared reference file missing");reference.seekg(0,std::ios::end);need(reference.tellg()>=0&&U(reference.tellg())==next*4,"prepared labels file size differs");reference.seekg(0);
  U mask_independent=0,mask_derived=0,unknown_prompt=0;for(const auto& [sid,c]:case_map){J provenance=prompt_provenance(c);if(provenance.is_null())++unknown_prompt;else if(provenance.get<bool>())++mask_derived;else ++mask_independent;}
  std::string prompt_class=mask_independent==case_map.size()?"declared_reference_mask_independent":mask_derived==case_map.size()?"declared_reference_mask_derived":unknown_prompt==case_map.size()?"unknown":"mixed_or_unknown";
  report["exported_prompt_provenance"]={{"classification",prompt_class},{"declared_reference_mask_independent_scans",mask_independent},{"declared_reference_mask_derived_scans",mask_derived},{"unknown_scans",unknown_prompt},{"all_scans_declare_reference_mask_independent_prompt",mask_independent==case_map.size()},{"scope","Exported primary prompt declaration; actual inference prompt selection requires the model-run receipt"}};
  std::vector<U> patient_names(patient_ids.begin(),patient_ids.end());std::vector<std::uint32_t> scan_patient;std::vector<L> outside,native_positive;std::vector<J> scan_meta;
  Device<L> all_counts(scan_map.size()*4);all_counts.zero();unsigned position=0;
  report["dataset_path"]=dataset_path.string();report["export_manifest_path"]=export_path.string();report["predictions_directory"]=predictions.string();report["patients_expected"]=patient_ids.size();report["scans_expected"]=scan_map.size();report["scans"]=J::array();
  for(const auto& [sid,s]:scan_map){const auto& c=case_map.at(sid);std::string name=c.at("case").get<std::string>();report["pending_case"]=name;auto path=predictions/(name+".mha");Image image=read_mha(path);auto expected=grid_json(c),crop=grid_json(c.at("cropGrid"));same_grid(image.grid,expected,name);
    Device<std::uint8_t> bytes(image.bytes.size()),mask(image.grid.count());bytes.set(image.bytes);Device<L> positives(1);positives.zero();Device<unsigned> bad(1);bad.zero();launch_decode(image,bytes,mask,positives,bad);sync();need(!bad.get()[0],name+": prediction values must be exact finite binary labels0/1");auto positive=positives.get()[0];
    auto y=read_truth(reference,integer(s,"row_start"),integer(s,"rows"));Device<std::uint32_t> truth(y.size());truth.set(y);confusion<<<blocks(y.size()),256>>>(mask.p,image.grid,crop,truth.p,all_counts.p+U(position)*4,bad.p);sync();need(!bad.get()[0],name+": prepared reference values are nonbinary");std::array<L,4> counts;ck(cudaMemcpy(counts.data(),all_counts.p+U(position)*4,4*sizeof(L),cudaMemcpyDeviceToHost));need(counts[0]+counts[1]+counts[2]+counts[3]==rows_per_scan&&counts[0]+counts[2]==integer(s,"truth_positive_rows"),name+": prepared reference count differs from declared metadata");
    U pid=integer(s,"patient_id");scan_patient.push_back(std::uint32_t(std::lower_bound(patient_names.begin(),patient_names.end(),pid)-patient_names.begin()));outside.push_back(integer(s,"truth_positive_outside_rows"));native_positive.push_back(positive);
    J row={{"scan_id",sid},{"patient_id",pid},{"patient",s.value("patient",std::string{})},{"case",name},{"prediction_path",path.string()},{"primary_prompt_uses_reference_mask",prompt_provenance(c)},{"prediction_geometry",grid_report(image.grid)},{"TP",counts[0]},{"FP",counts[1]},{"FN_in_ROI",counts[2]},{"FN_outside_ROI",outside.back()},{"TN_in_ROI",counts[3]},{"predicted_positive_voxels",counts[0]+counts[1]},{"reference_positive_voxels",counts[0]+counts[2]+outside.back()},{"full_prediction_positive_native_voxels",positive},{"empty_prediction",positive==0},{"empty_prediction_in_ROI",counts[0]+counts[1]==0}};scan_meta.push_back(row);report["scans"].push_back(row);++position;std::cout<<J{{"event","external_mask_scored"},{"case",name},{"scans_complete",position},{"empty_prediction",positive==0}}.dump()<<std::endl;
  }
  Device<L> dout(outside.size()),dpositive(native_positive.size()),pooled(patient_names.size()*4),summary(7);dout.set(outside);dpositive.set(native_positive);pooled.zero();summary.zero();Device<std::uint32_t> mapping(scan_patient.size());mapping.set(scan_patient);Device<double> pd(patient_names.size()),sd(scan_map.size()),macro(1);macro.zero();summarize<<<1,1>>>(all_counts.p,dout.p,mapping.p,dpositive.p,scan_map.size(),patient_names.size(),pooled.p,pd.p,sd.p,summary.p,macro.p);sync();auto pcounts=pooled.get();auto pmetrics=pd.get(),smetrics=sd.get();auto totals=summary.get();J patients=J::array();
  for(unsigned p=0;p<patient_names.size();++p){J ids=J::array();for(const auto& [sid,s]:scan_map)if(integer(s,"patient_id")==patient_names[p])ids.push_back(sid);const L* count=pcounts.data()+U(p)*4;patients.push_back({{"patient_id",patient_names[p]},{"scan_ids",ids},{"Dice",pmetrics[p]},{"TP",count[0]},{"FP",count[1]},{"FN",count[2]},{"TN_in_ROI",count[3]}});}
  for(unsigned s=0;s<scan_meta.size();++s)scan_meta[s]["Dice"]=smetrics[s];report["scans"]=scan_meta;report["patients"]=patients;report["macro_patient_Dice"]=macro.get()[0];report["summary"]={{"patients",patient_names.size()},{"scans",scan_map.size()},{"TP",totals[0]},{"FP",totals[1]},{"FN_in_ROI",totals[2]},{"TN_in_ROI",totals[3]},{"FN_outside_ROI",totals[4]},{"empty_prediction_scans",totals[5]},{"empty_prediction_in_ROI_scans",totals[6]}};report.erase("pending_case");report["complete"]=true;return 0;
}
int self_test(){
  unsigned checks=0;auto check=[&](bool ok,const std::string& message){need(ok,message);++checks;};auto reject=[&](auto action,const std::string& message){bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}check(rejected,message);};
  Grid full{{8,6,4},{1,1,1},{5,-7,10},{0,-1,0,1,0,0,0,0,1}};
  Grid crop{{4,3,2},{2,2,2},{4,-6,11},{0,-1,0,1,0,0,0,0,1}};validate_grid(full);validate_grid(crop);check(full.count()==192&&crop.count()==24,"geometry extents failed");
  Grid cube{{4,4,4},{2,2,2},{6,-8,7},{0,-1,0,1,0,0,0,0,1}};J case_geometry={{"seedLPS",J::array({2.,-4.,11.})},{"seedGridXYZ",J::array({3.,3.,1.})}};validate_crop(case_geometry,full,cube,4,2.);check(true,"dataset-derived crop validation failed");auto badcube=cube;badcube.spacing[0]=1.;reject([&]{validate_crop(case_geometry,full,badcube,4,2.);},"crop spacing mismatch accepted");badcube=cube;badcube.origin[0]+=1.;reject([&]{validate_crop(case_geometry,full,badcube,4,2.);},"crop center mismatch accepted");auto badseed=case_geometry;badseed["seedGridXYZ"][0]=4.;reject([&]{validate_crop(badseed,full,cube,4,2.);},"physical/grid seed mismatch accepted");
  Grid invalid=full;invalid.direction[0]=1;reject([&]{validate_grid(invalid);},"nonorthonormal geometry accepted");invalid=full;invalid.spacing[2]=0;reject([&]{validate_grid(invalid);},"zero spacing accepted");invalid=full;invalid.origin[2]=std::numeric_limits<double>::quiet_NaN();reject([&]{validate_grid(invalid);},"nonfinite origin accepted");invalid=full;invalid.size[0]=0;reject([&]{validate_grid(invalid);},"zero extent accepted");invalid=full;invalid.origin[0]+=.001;reject([&]{same_grid(invalid,full,"fixture");},"mismatched prediction origin accepted");
  std::vector<std::uint8_t> raw(full.count(),0);raw[(1*6+1)*8+3]=1;raw[(1*6+1)*8+5]=1;
  Device<std::uint8_t> source(raw.size()),mask(raw.size());source.set(raw);Device<L> positives(1),counts(4);Device<unsigned> bad(1);positives.zero();bad.zero();decode<<<1,256>>>(source.p,mask.p,raw.size(),positives.p,bad.p);sync();check(!bad.get()[0]&&positives.get()[0]==2,"binary UCHAR transport failed");
  std::vector<std::uint32_t> y(crop.count(),0);y[1]=y[3]=1;Device<std::uint32_t> truth(y.size());truth.set(y);counts.zero();confusion<<<1,256>>>(mask.p,full,crop,truth.p,counts.p,bad.p);sync();check(counts.get()==std::vector<L>{1,1,1,21},"rotated physical crop resampling/confusion failed");
  std::fill(raw.begin(),raw.end(),0);source.set(raw);positives.zero();decode<<<1,256>>>(source.p,mask.p,raw.size(),positives.p,bad.p);counts.zero();confusion<<<1,256>>>(mask.p,full,crop,truth.p,counts.p,bad.p);sync();check(positives.get()[0]==0&&counts.get()==std::vector<L>{0,0,2,22},"valid empty prediction was not scored as explicit false negatives");
  raw[0]=2;source.set(raw);bad.zero();decode<<<1,256>>>(source.p,mask.p,raw.size(),positives.p,bad.p);sync();check(bad.get()[0]==1,"nonbinary UCHAR mask accepted");
  std::vector<float> floats{0,1,.5f,std::numeric_limits<float>::quiet_NaN()};Device<float> df(floats.size());df.set(floats);Device<std::uint8_t> fm(floats.size());bad.zero();positives.zero();decode<<<1,32>>>(df.p,fm.p,floats.size(),positives.p,bad.p);sync();check(bad.get()[0]==1&&positives.get()[0]==1,"fractional/NaN labels accepted");
  Grid tieSource{{2,1,1},{1,1,1},{0,0,0},{1,0,0,0,1,0,0,0,1}},tieCrop{{3,1,1},{1,1,1},{-.5,0,0},{1,0,0,0,1,0,0,0,1}};
  Device<std::uint8_t> tieMask(2);tieMask.set(std::vector<std::uint8_t>{0,1});Device<std::uint32_t> tieTruth(3);tieTruth.set(std::vector<std::uint32_t>{0,1,0});counts.zero();bad.zero();confusion<<<1,32>>>(tieMask.p,tieSource,tieCrop,tieTruth.p,counts.p,bad.p);sync();check(counts.get()==std::vector<L>{1,0,0,2},"nearest-neighbor tie/outside support failed");tieTruth.set(std::vector<std::uint32_t>{0,2,0});bad.zero();confusion<<<1,32>>>(tieMask.p,tieSource,tieCrop,tieTruth.p,counts.p,bad.p);sync();check(bad.get()[0]==1,"corrupt prepared reference accepted");
  Device<L> sc(12),out(3),np(3),pooled(8),summary(7);Device<std::uint32_t> patients(3);Device<double> pd(2),sd(3),macro(1);
  sc.set(std::vector<L>{4,0,0,10,0,0,4,10,1,0,0,10});out.set(std::vector<L>{0,0,2});np.set(std::vector<L>{4,0,1});patients.set(std::vector<std::uint32_t>{0,0,1});pooled.zero();summary.zero();macro.zero();summarize<<<1,1>>>(sc.p,out.p,patients.p,np.p,3,2,pooled.p,pd.p,sd.p,summary.p,macro.p);sync();check(pooled.get()==std::vector<L>{4,0,4,20,1,0,2,10},"patient repeat pooling/outside FN failed");auto dice=pd.get();check(std::abs(dice[0]-2./3.)<1e-14&&dice[1]==.5,"patient Dice differs from pooled-count endpoint");check(std::abs(macro.get()[0]-7./12.)<1e-14,"equal patient macro weighting failed");check(sd.get()==std::vector<double>{1,0,.5},"scan Dice outside FN accounting failed");check(summary.get()==std::vector<L>{5,0,4,30,2,1,1},"summary/empty-prediction accounting failed");
  // Exercise real binary file transport and corrupt headers without numerical
  // Python/third-party processing or a model dependency.
  fs::path temp=fs::temp_directory_path()/("segmentation-score-check-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directory(temp);
  try{
    const std::string header="ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nTransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = 0 0 0\nElementSpacing = 1 1 1\nDimSize = 2 1 1\nElementType = MET_FLOAT\nElementDataFile = LOCAL\n";
    auto write=[&](const std::string& h,std::size_t n){std::ofstream f(temp/"test.mha",std::ios::binary);f<<h;const float payload[2]{0,1};f.write(reinterpret_cast<const char*>(payload),n);};
    write(header,8);auto image=read_mha(temp/"test.mha");check(image.type=="MET_FLOAT"&&image.bytes.size()==8&&image.grid.count()==2,"MetaImage LOCAL transport failed");
    Device<std::uint8_t> ib(image.bytes.size()),im(image.grid.count());ib.set(image.bytes);positives.zero();bad.zero();launch_decode(image,ib,im,positives,bad);sync();check(positives.get()[0]==1&&!bad.get()[0],"typed MetaImage float decoding failed");
    // File bytes use MetaImage's column-major axis list. Checking the physical
    // crop after decoding catches an accidental transpose even when a direct
    // CUDA fixture and exporter/scorer share the same internal convention.
    const std::string rotatedHeader="ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nTransformMatrix = 0 1 0 -1 0 0 0 0 1\nOffset = 5 -7 10\nElementSpacing = 1 1 1\nDimSize = 8 6 4\nElementType = MET_UCHAR\nElementDataFile = LOCAL\n";
    std::vector<std::uint8_t> rotatedPixels(full.count(),0);rotatedPixels[(1*6+1)*8+3]=1;rotatedPixels[(1*6+1)*8+5]=1;
    {std::ofstream f(temp/"rotated.mha",std::ios::binary);f<<rotatedHeader;f.write(reinterpret_cast<const char*>(rotatedPixels.data()),std::streamsize(rotatedPixels.size()));f.close();need(bool(f),"rotated serialized fixture write failed");}
    auto rotated=read_mha(temp/"rotated.mha");same_grid(rotated.grid,full,"serialized rotated fixture");check(std::equal(rotated.grid.direction,rotated.grid.direction+9,full.direction),"MetaImage axis serialization was not transposed into physical row-major geometry");
    Device<std::uint8_t> rotatedBytes(rotated.bytes.size()),rotatedMask(rotated.grid.count());rotatedBytes.set(rotated.bytes);positives.zero();bad.zero();launch_decode(rotated,rotatedBytes,rotatedMask,positives,bad);counts.zero();confusion<<<1,256>>>(rotatedMask.p,rotated.grid,crop,truth.p,counts.p,bad.p);sync();check(!bad.get()[0]&&positives.get()[0]==2&&counts.get()==std::vector<L>{1,1,1,21},"serialized nonidentity MetaImage physical crop/confusion failed");
    write(header,4);reject([&]{read_mha(temp/"test.mha");},"truncated prediction accepted");write(header,0);reject([&]{read_mha(temp/"test.mha");},"zero-byte mask payload accepted");
    auto compressed=header;compressed.replace(compressed.find("CompressedData = False"),22,"CompressedData = True");write(compressed,8);reject([&]{read_mha(temp/"test.mha");},"compressed mask transport accepted");
    auto multi=header;multi.insert(multi.find("ElementType"),"ElementNumberOfChannels = 2\n");write(multi,8);reject([&]{read_mha(temp/"test.mha");},"multichannel mask accepted");
    auto duplicate=header;duplicate.insert(duplicate.find("ElementType"),"NDims = 3\n");write(duplicate,8);reject([&]{read_mha(temp/"test.mha");},"duplicate image header accepted");
    auto external=header;external.replace(external.find("LOCAL"),5,"missing.raw");write(external,8);reject([&]{read_mha(temp/"test.mha");},"external raw image transport accepted");
    auto big=header;big.replace(big.find("ByteOrderMSB = False"),20,"ByteOrderMSB = True");write(big,8);reject([&]{read_mha(temp/"test.mha");},"big-endian mask accepted");
    reject([&]{read_mha(temp/"missing.mha");},"missing prediction accepted");
  }catch(...){fs::remove_all(temp);throw;}fs::remove_all(temp);
  std::cout<<J{{"event","segmentation_score_self_test"},{"passed",true},{"checks",checks},{"CUDA_numerical_metrics",true},{"model_inference_performed",false}}.dump()<<std::endl;return 0;
}
}
int main(int argc,char** argv){
  fs::path output;J report={{"format","external-mask-segmentation-result-1"},{"complete",false},{"failure",nullptr},{"endpoint","matched prepared physical crop; truth outside crop remains false negative"},{"patient_Dice_definition","pool all verified scans per patient; include all truth-positive voxels outside ROI as false negatives; equally weight every declared patient"},{"prediction_resampling","nearest neighbor in physical LPS space; input/output full CT geometry must match"},{"reference","unchanged prepared binary labels and precomputed outside-crop truth"},{"outside_crop_predictions","ignored for primary matched prediction domain"},{"missing_prediction_policy","fail closed; no macro endpoint on missing, unreadable, malformed, mismatched or nonbinary masks"},{"empty_prediction_policy","a present validated all-zero binary mask is explicitly flagged and scores its truth as false negatives"},{"numerical_backend","C++23 CUDA"}};
  try{
    if(argc==2&&std::string(argv[1])=="--self-test")return self_test();
    need(argc==5,"usage: class_segmentation_score PREPARED_DATASET_JSON EXPORTED_CASES_JSON PREDICTIONS_DIRECTORY FRESH_RESULT_JSON");output=fs::path(argv[4]);need(output.is_absolute()&&!fs::exists(output),"result must be a fresh absolute path");need(fs::is_directory(output.parent_path()),"result parent directory is missing");
    int result=score(fs::absolute(argv[1]),fs::absolute(argv[2]),fs::absolute(argv[3]),report);write_json(output,report);std::cout<<J{{"event","external_mask_scoring_complete"},{"result",output.string()},{"macro_patient_Dice",report.at("macro_patient_Dice")}}.dump()<<std::endl;return result;
  }catch(const std::exception& e){report["complete"]=false;report.erase("macro_patient_Dice");report["failure"]={{"message",e.what()},{"scoring_complete",false}};if(!output.empty()&&!fs::exists(output)&&fs::is_directory(output.parent_path()))try{write_json(output,report);}catch(const std::exception& write){std::cerr<<"cannot retain failure report: "<<write.what()<<'\n';}std::cerr<<"external mask scoring failed: "<<e.what()<<'\n';return 1;}
}
