#include "class_study_dicom.hpp"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Export validated source CT and independent physical SEG planes onto a common
// regular full-image grid. DICOM transport and MetaImage/JSON I/O are host work;
// attenuation resampling, binary reference sampling and mask statistics are CUDA.
// The primary prompt remains the external spreadsheet location. A separately
// named reference-derived prompt is optional and explicitly marked in metadata.
namespace {
using J = nlohmann::json;
using U = std::uint64_t;
using V3 = std::array<double, 3>;
namespace fs = std::filesystem;
void need(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void ck(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Device {
  T* p = nullptr;
  std::size_t n;
  explicit Device(std::size_t count) : n(count) {
    need(count && count <= std::numeric_limits<std::size_t>::max()/sizeof(T), "invalid CUDA allocation extent");
    ck(cudaMalloc(&p, count*sizeof(T)));
  }
  ~Device() { if (p) cudaFree(p); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void upload(const T* source) { ck(cudaMemcpy(p,source,n*sizeof(T),cudaMemcpyHostToDevice)); }
  void download(T* target) { ck(cudaMemcpy(target,p,n*sizeof(T),cudaMemcpyDeviceToHost)); }
};
struct Geometry {
  int nx, ny, sourceNz, outputNz, labelNz;
  double dz, labelSpacing;
  const double* sourcePositions;
  const double* labelPositions;
};
__device__ U index(int x, int y, int z, Geometry g) {
  return (U(z)*g.ny+y)*g.nx+x;
}
__device__ float attenuation(const std::int16_t* pixels, const float* slopes,
                             const float* intercepts, int x, int y, int z, Geometry g) {
  if (z < 0 || z >= g.sourceNz) return -1000.0f;
  return float(pixels[index(x,y,z,g)])*slopes[z]+intercepts[z];
}
__device__ double source_coordinate(double position, Geometry g) {
  const double* p = g.sourcePositions;
  if (position <= p[0]) return (position-p[0])/(p[1]-p[0]);
  if (position >= p[g.sourceNz-1])
    return g.sourceNz-1+(position-p[g.sourceNz-1])/(p[g.sourceNz-1]-p[g.sourceNz-2]);
  int low=0, high=g.sourceNz-1;
  while (high-low > 1) {
    int middle=(low+high)/2;
    if (p[middle] <= position) low=middle; else high=middle;
  }
  return low+(position-p[low])/(p[high]-p[low]);
}
__device__ std::uint8_t reference(const std::uint8_t* labels, int x, int y,
                                  double position, Geometry g) {
  int low=0, high=g.labelNz;
  while (low < high) {
    int middle=(low+high)/2;
    if (g.labelPositions[middle] < position) low=middle+1; else high=middle;
  }
  int z=low==g.labelNz ? g.labelNz-1 : low;
  if (z > 0 && position-g.labelPositions[z-1] < g.labelPositions[z]-position) --z;
  // An omitted empty SEG frame must not acquire its nearest stored mask.
  if (fabs(position-g.labelPositions[z]) > 0.5*g.labelSpacing+1e-6) return 0;
  return std::uint8_t(labels[index(x,y,z,g)] != 0);
}
__global__ void export_grid(const std::int16_t* pixels, const float* slopes,
                            const float* intercepts, const std::uint8_t* labels,
                            float* hu, std::uint8_t* mask, unsigned long long* statistics,
                            Geometry g) {
  U count=U(g.nx)*g.ny*g.outputNz;
  for (U q=U(blockIdx.x)*blockDim.x+threadIdx.x; q<count; q+=U(blockDim.x)*gridDim.x) {
    int x=int(q%g.nx), y=int((q/g.nx)%g.ny), z=int(q/(U(g.nx)*g.ny));
    double position=z*g.dz, coordinate=source_coordinate(position,g);
    int lower=int(floor(coordinate));
    float fraction=float(coordinate-lower);
    hu[q]=attenuation(pixels,slopes,intercepts,x,y,lower,g)*(1.0f-fraction)
          +attenuation(pixels,slopes,intercepts,x,y,lower+1,g)*fraction;
    mask[q]=reference(labels,x,y,position,g);
    if (mask[q]) {
      atomicAdd(statistics,1ULL);
      atomicAdd(statistics+1,static_cast<unsigned long long>(x));
      atomicAdd(statistics+2,static_cast<unsigned long long>(y));
      atomicAdd(statistics+3,static_cast<unsigned long long>(z));
    }
  }
}
V3 normal(const class_study::ScanVolume& volume) {
  const auto& r=volume.iopRow;
  const auto& c=volume.iopCol;
  V3 n{r[1]*c[2]-r[2]*c[1], r[2]*c[0]-r[0]*c[2], r[0]*c[1]-r[1]*c[0]};
  double length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
  need(std::isfinite(length) && length > 0, "invalid image normal");
  for (double& value:n) value/=length;
  return n;
}
std::array<double,9> direction(const class_study::ScanVolume& volume, const V3& n) {
  return {volume.iopRow[0],volume.iopCol[0],n[0],
          volume.iopRow[1],volume.iopCol[1],n[1],
          volume.iopRow[2],volume.iopCol[2],n[2]};
}
V3 physical(const class_study::ScanVolume& volume, const V3& n, double x, double y, double zMM) {
  V3 point=volume.imageOriginLPS;
  for (unsigned axis=0;axis<3;++axis)
    point[axis]+=volume.iopRow[axis]*x*volume.spacingXYZ[0]
                 +volume.iopCol[axis]*y*volume.spacingXYZ[1]+n[axis]*zMM;
  return point;
}
std::ofstream output(const fs::path& path) {
  std::ofstream stream(path,std::ios::binary);
  need(bool(stream),"cannot create "+path.string());
  return stream;
}
void json_file(const fs::path& path, const J& json) {
  auto stream=output(path); stream<<json.dump(2)<<'\n'; stream.close();
  need(bool(stream),"JSON write failed: "+path.string());
}
template<class T> void mha(const fs::path& path, const std::vector<T>& values,
                           const std::array<int,3>& size, const V3& spacing,
                           const V3& origin, const std::array<double,9>& orientation,
                           const char* elementType) {
  static_assert(std::endian::native==std::endian::little,"MetaImage writer requires little-endian host");
  need(values.size()==U(size[0])*size[1]*size[2],"MetaImage voxel extent mismatch");
  auto stream=output(path);
  stream<<std::setprecision(17)
        <<"ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nTransformMatrix =";
  // MetaImage serializes the physical direction matrix by columns, while the
  // public geometry records use row-major directionLPS (SimpleITK convention).
  for (unsigned column=0;column<3;++column)
    for (unsigned row=0;row<3;++row) stream<<' '<<orientation[row*3+column];
  stream<<"\nOffset = "<<origin[0]<<' '<<origin[1]<<' '<<origin[2]
        <<"\nElementSpacing = "<<spacing[0]<<' '<<spacing[1]<<' '<<spacing[2]
        <<"\nDimSize = "<<size[0]<<' '<<size[1]<<' '<<size[2]
        <<"\nElementType = "<<elementType<<"\nElementDataFile = LOCAL\n";
  stream.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(T)));
  stream.close(); need(bool(stream),"MetaImage write failed: "+path.string());
}
// Host-only I/O fixture: no device allocation, resampling or numerical model
// execution. Retain this tiny image so an independent SimpleITK read can check
// the MetaImage column-major header against the row-major physical direction.
int header_self_test(const fs::path& fixture) {
  need(fixture.is_absolute() && !fs::exists(fixture),"header fixture must be a fresh absolute path");
  need(fs::is_directory(fixture.parent_path()),"header fixture parent directory is missing");
  const std::array<double,9> orientation{0,-1,0,1,0,0,0,0,1};
  const V3 spacing{0.7,0.8,1.25}, origin{10,20,30};
  mha(fixture,std::vector<float>{3,7},std::array<int,3>{2,1,1},spacing,origin,orientation,"MET_FLOAT");
  std::ifstream stream(fixture,std::ios::binary); std::string line; bool checked=false;
  while (std::getline(stream,line)) {
    if (line.starts_with("TransformMatrix =")) {
      need(line=="TransformMatrix = 0 1 0 -1 0 0 0 0 1","MetaImage column-major direction serialization failed");
      checked=true; break;
    }
  }
  need(checked,"MetaImage fixture has no direction matrix");
  std::cout<<J{{"event","segmentation_export_header_self_test"},{"passed",true},
    {"fixture",fixture.string()},{"directionLPS_row_major",orientation},{"CUDA_numerical_execution",false}}.dump()<<std::endl;
  return 0;
}
struct Exported {
  std::vector<float> hu;
  std::vector<std::uint8_t> labels;
  std::array<unsigned long long,4> stats{};
  int nz=0;
};
Exported resample(const class_study::ScanVolume& volume) {
  need(volume.nz>=2 && volume.labelNz && volume.nx && volume.ny,"invalid source extent");
  need(volume.nz<=INT_MAX && volume.nx<=INT_MAX && volume.ny<=INT_MAX && volume.labelNz<=INT_MAX,"source extent unsupported");
  need(volume.sliceOffsetsMM.size()==volume.nz && volume.labelSliceOffsetsMM.size()==volume.labelNz,"plane coordinate extent mismatch");
  need(volume.spacingXYZ[2]>0 && volume.labelSliceSpacingMM>0,"invalid physical spacing");
  double intervals=volume.sliceOffsetsMM.back()/volume.spacingXYZ[2];
  need(std::isfinite(intervals) && intervals>=1 && intervals<INT_MAX-1,"output z extent unsupported");
  Exported result;
  // Include every original voxel-center position. A fractional last interval is
  // covered by one documented final regular output plane, without pretending
  // that any newly resampled plane existed in the original acquisition.
  result.nz=int(std::ceil(intervals-1e-7))+1;
  U sourceCount=U(volume.nx)*volume.ny*volume.nz;
  U labelCount=U(volume.nx)*volume.ny*volume.labelNz;
  U count=U(volume.nx)*volume.ny*result.nz;
  need(volume.rawPixels.size()==sourceCount && volume.labels.size()==labelCount,"source buffer extent mismatch");
  need(volume.perSliceSlope.size()==volume.nz && volume.perSliceIntercept.size()==volume.nz,"rescale extent mismatch");
  Device<std::int16_t> pixels(sourceCount); pixels.upload(volume.rawPixels.data());
  Device<float> slopes(volume.nz), intercepts(volume.nz); slopes.upload(volume.perSliceSlope.data()); intercepts.upload(volume.perSliceIntercept.data());
  Device<std::uint8_t> labels(labelCount); labels.upload(volume.labels.data());
  Device<double> sourcePositions(volume.nz), labelPositions(volume.labelNz);
  sourcePositions.upload(volume.sliceOffsetsMM.data()); labelPositions.upload(volume.labelSliceOffsetsMM.data());
  Device<float> hu(count); Device<std::uint8_t> mask(count); Device<unsigned long long> stats(4);
  ck(cudaMemset(stats.p,0,4*sizeof(unsigned long long)));
  Geometry geometry{int(volume.nx),int(volume.ny),int(volume.nz),result.nz,int(volume.labelNz),volume.spacingXYZ[2],volume.labelSliceSpacingMM,sourcePositions.p,labelPositions.p};
  export_grid<<<std::min<U>(4096,(count+255)/256),256>>>(pixels.p,slopes.p,intercepts.p,labels.p,hu.p,mask.p,stats.p,geometry);
  ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  result.hu.resize(count); result.labels.resize(count);
  hu.download(result.hu.data()); mask.download(result.labels.data()); stats.download(result.stats.data());
  return result;
}
// Seed origin belongs to the supplied descriptor. The exporter never derives
// its primary point from a reference; an unspecified origin remains unknown.
J declared_reference_use(const J& scan) {
  const auto provenance=scan.find("seed_coordinate_provenance");
  if (provenance==scan.end() || !provenance->is_object()) return nullptr;
  const auto use=provenance->find("uses_seg_pixels");
  return use!=provenance->end() && use->is_boolean() ? *use : J(nullptr);
}
J prompt(const V3& gridXYZ) {
  return J{{"1",{{"point",J::array({gridXYZ[2],gridXYZ[1],gridXYZ[0]})}}}};
}
int self_test() {
  class_study::ScanVolume v;
  v.nx=3; v.ny=2; v.nz=4; v.labelNz=3;
  v.spacingXYZ={0.7,0.8,1}; v.sliceOffsetsMM={0,1,4,5};
  v.labelSliceOffsetsMM={1,3,5}; v.labelSliceSpacingMM=1;
  v.iopRow={0,1,0}; v.iopCol={-1,0,0}; v.imageOriginLPS={100,200,300};
  v.rawPixels.resize(24); v.labels.assign(18,0);
  v.perSliceSlope={2,2,2,2}; v.perSliceIntercept={7,7,7,7};
  for (unsigned z=0;z<4;++z) for (unsigned xy=0;xy<6;++xy) v.rawPixels[z*6+xy]=std::int16_t(v.sliceOffsetsMM[z]*5);
  for (unsigned z=0;z<3;++z) v.labels[z*6+4]=1;
  auto result=resample(v);
  need(result.nz==6 && result.stats==std::array<unsigned long long,4>{3,3,3,9},"full-grid reference CUDA statistics failed");
  for (U q=0;q<36;++q) {
    unsigned z=unsigned(q/6);
    need(std::abs(result.hu[q]-(10*z+7))<1e-5,"HU actual-plane interpolation/rescale failed");
    need(result.labels[q]==((q%6==4 && z%2==1)?1:0),"independent sparse SEG grid failed");
  }
  const auto n=normal(v); const auto d=direction(v,n);
  need(d==std::array<double,9>{0,-1,0,1,0,0,0,0,1},"MetaImage direction matrix failed");
  need(physical(v,n,1,1,3)==V3{99.2,200.7,303},"LPS physical-coordinate construction failed");
  need(prompt({1.5,2.5,3.5}).at("1").at("point")==J::array({3.5,2.5,1.5}),"fractional ZYX prompt encoding failed");
  need(declared_reference_use(J::object()).is_null(),"unspecified prompt origin must remain unknown");
  need(declared_reference_use(J{{"seed_coordinate_provenance",{{"uses_seg_pixels",false}}}})==J(false),"independent declared prompt provenance failed");
  need(declared_reference_use(J{{"seed_coordinate_provenance",{{"uses_seg_pixels",true}}}})==J(true),"reference-derived declared prompt provenance failed");
  std::cout<<"CUDA full-grid export checks passed\n";
  return 0;
}
}
int main(int argc, char** argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--self-test") return self_test();
    if (argc==3 && std::string(argv[1])=="--header-self-test") return header_self_test(fs::path(argv[2]));
    need(argc>=3,"usage: class_segmentation_export MANIFEST_OR_PREPARED_DATASET_JSON FRESH_OUTPUT_DIRECTORY [--scan-id ID] [--reference-center-prompts]");
    std::optional<unsigned> selected;
    bool referencePrompts=false;
    for (int arg=3;arg<argc;++arg) {
      std::string option=argv[arg];
      if (option=="--reference-center-prompts") referencePrompts=true;
      else if (option=="--scan-id") {
        need(arg+1<argc && !selected,"--scan-id requires one integer");
        std::string value=argv[++arg]; std::size_t parsed=0;
        unsigned long id=std::stoul(value,&parsed);
        need(parsed==value.size() && id<=UINT_MAX,"invalid scan id"); selected=unsigned(id);
      } else throw std::runtime_error("unsupported export option "+option);
    }
    std::ifstream input(argv[1]); need(bool(input),"cannot read manifest/dataset");
    J source; input>>source;
    bool prepared=source.value("format",std::string{})=="grouped-segmentation-dataset-1";
    const J& manifest=prepared?source.at("manifest"):source;
    need(manifest.at("scans").is_array() && !manifest.at("scans").empty(),"manifest needs scans");
    std::map<unsigned,J> preparedScans;
    if (prepared) for (const auto& scan:source.at("scans")) preparedScans.emplace(scan.at("scan_id").get<unsigned>(),scan);
    double cropWidth=prepared?source.at("crop_width_mm").get<double>():manifest.value("crop_width_mm",160.0);
    double cropStep=prepared?source.at("grid_spacing_mm").get<double>():manifest.value("grid_spacing_mm",2.0);
    need(cropWidth>0 && cropStep>0 && std::isfinite(cropWidth) && std::isfinite(cropStep),"invalid study crop geometry");
    double ratio=cropWidth/cropStep;
    need(ratio>=4 && ratio<=256 && std::abs(ratio-std::round(ratio))<1e-9,"invalid crop dimension");
    int cropN=int(std::round(ratio)); need(cropN%2==0,"crop side must be even");
    fs::path root=fs::absolute(argv[2]); need(!fs::exists(root),"output must be fresh");
    fs::create_directories(root/"images"); fs::create_directories(root/"labels"); fs::create_directories(root/"prompts");
    if (referencePrompts) fs::create_directories(root/"prompts-reference-center");
    J result={{"format","located-segmentation-inputs-1"},{"source_descriptor",fs::absolute(argv[1]).string()},
      {"source_manifest",manifest},{"image_format","MetaImage binary uncompressed, float HU"},
      {"storage_order","x fastest, then y, then z; prompt list is zero-based Z,Y,X"},
      {"grid_policy","Original CT in-plane grid; regular Z at minimum actual CT interval from first original plane through last original plane; actual-plane HU interpolation; independent physical-plane nearest SEG with omitted-plane zero support"},
      {"primary_prompt_source","Supplied scan descriptor coordinates; origin provenance retained per case"},
      {"primary_prompt_reference_use_semantics","Per-case primary_prompt_uses_reference_mask retains seed_coordinate_provenance.uses_seg_pixels; null when unspecified"},
      {"reference_center_prompts_enabled",referencePrompts},{"cases",J::array()}};
    std::set<unsigned> ids, patients;
    std::size_t scanIndex=0;
    for (const auto& scan:manifest.at("scans")) {
      unsigned id=scan.at("scan_id").get<unsigned>(), patientId=scan.at("patient_id").get<unsigned>();
      need(ids.insert(id).second,"duplicate scan id in manifest");
      if (selected && id!=*selected) { ++scanIndex; continue; }
      auto v=class_study::read_segmentation_scan(scan);
      auto data=resample(v); need(data.stats[0]>0,"full-grid resampling removed every reference positive");
      V3 n=normal(v), seed=physical(v,n,v.seedGridXYZ[0],v.seedGridXYZ[1],v.seedProjectionOffsetMM);
      auto orientation=direction(v,n);
      V3 seedGrid{v.seedGridXYZ[0],v.seedGridXYZ[1],v.seedProjectionOffsetMM/v.spacingXYZ[2]};
      std::array<int,3> size{int(v.nx),int(v.ny),data.nz};
      std::ostringstream basename; basename<<"scan_"<<std::setfill('0')<<std::setw(4)<<id;
      std::string name=basename.str();
      fs::path image=root/"images"/(name+".mha"), reference=root/"labels"/(name+".mha"), point=root/"prompts"/(name+".json");
      mha(image,data.hu,size,v.spacingXYZ,v.imageOriginLPS,orientation,"MET_FLOAT");
      mha(reference,data.labels,size,v.spacingXYZ,v.imageOriginLPS,orientation,"MET_UCHAR");
      json_file(point,prompt(seedGrid));
      V3 cropOrigin=seed;
      for (unsigned axis=0;axis<3;++axis) cropOrigin[axis]-=(cropN/2)*cropStep*(v.iopRow[axis]+v.iopCol[axis]+n[axis]);
      J detail={{"case",name},{"scan_id",id},{"patient_id",patientId},{"scanIndex",scanIndex},{"scanId",id},{"patientId",patientId},
        {"patient",scan.value("patient",v.metadata.at("patient_id").get<std::string>())},
        {"image_path",image.string()},{"reference_path",reference.string()},{"prompt_path",point.string()},
        {"prediction_filename",name+".mha"},{"sizeXYZ",size},{"spacingXYZ",v.spacingXYZ},{"originLPS",v.imageOriginLPS},{"directionLPS",orientation},
        {"seedLPS",seed},{"seedGridXYZ",seedGrid},{"seedSourceCTGridXYZ",v.seedGridXYZ},
        {"seed_provenance",scan.value("seed_coordinate_provenance",J::object())},
        {"cropGrid",{{"sizeXYZ",J::array({cropN,cropN,cropN})},{"spacingXYZ",J::array({cropStep,cropStep,cropStep})},{"originLPS",cropOrigin},{"directionLPS",orientation}}},
        {"full_reference_positive_voxels",data.stats[0]},{"full_reference_volume_mm3",data.stats[0]*v.spacingXYZ[0]*v.spacingXYZ[1]*v.spacingXYZ[2]},
        {"sourceScan",scan},{"dicom",v.metadata},{"regular_output_planes",data.nz},
        {"resampled_additional_planes",data.nz-int(v.nz)},
        {"last_regular_plane_offset_mm",(data.nz-1)*v.spacingXYZ[2]},
        {"source_last_plane_offset_mm",v.sliceOffsetsMM.back()},
        {"primary_prompt_uses_reference_mask",declared_reference_use(scan)},
        {"primary_prompt_derived_from_reference_by_exporter",false}};
      if (scan.contains("repeat_index")) detail["repeat_index"]=scan.at("repeat_index");
      if (prepared) {
        const auto& p=preparedScans.at(id);
        detail["preparedScan"]={{"row_start",p.at("row_start")},{"rows",p.at("rows")},
          {"truth_positive_rows",p.at("truth_positive_rows")},{"truth_positive_outside_rows",p.at("truth_positive_outside_rows")}};
      }
      if (referencePrompts) {
        V3 center{double(data.stats[1])/data.stats[0],double(data.stats[2])/data.stats[0],double(data.stats[3])/data.stats[0]};
        fs::path centerPath=root/"prompts-reference-center"/(name+".json"); json_file(centerPath,prompt(center));
        detail["secondary_reference_center_prompt"]={{"path",centerPath.string()},{"gridXYZ",center},
          {"definition","Center of mass of full exported binary reference; CUDA moments; literature-comparator prompt only"},{"uses_reference_mask",true}};
      }
      result["cases"].push_back(detail); patients.insert(patientId); ++scanIndex;
      // A recoverable metadata checkpoint describes only fully written cases.
      result["complete"]=false; result["scan_count"]=result["cases"].size(); result["patient_count"]=patients.size();
      json_file(root/"cases.partial.json",result);
      std::cout<<J{{"event","scan_exported"},{"case",name},{"sizeXYZ",size},{"reference_positive",data.stats[0]},
        {"irregular_source_intervals",v.metadata.at("irregular_slice_intervals")}}.dump()<<std::endl;
    }
    need(!result["cases"].empty(),"selected scan id absent from manifest");
    result["complete"]=true; result["scan_count"]=result["cases"].size(); result["patient_count"]=patients.size();
    result["complete_cohort"]=!selected;
    json_file(root/"cases.json",result); fs::remove(root/"cases.partial.json");
    std::cout<<J{{"event","complete"},{"manifest",(root/"cases.json").string()},{"scans",result["scan_count"]},{"patients",result["patient_count"]}}.dump()<<std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<"segmentation export failed: "<<error.what()<<'\n'; return 1;
  }
}
