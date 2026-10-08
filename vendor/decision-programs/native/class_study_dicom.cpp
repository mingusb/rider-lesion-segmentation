#include "class_study_dicom.hpp"
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcxfer.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace class_study {
namespace {
using J = nlohmann::json;
using V3 = std::array<double, 3>;
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("DICOM: " + message); }
void need(bool condition, const std::string& message) { if (!condition) fail(message); }
std::string str(DcmItem& d, const DcmTagKey& key, bool required = true) {
  OFString value;
  const bool good = d.findAndGetOFStringArray(key, value).good() && !value.empty();
  if (required) need(good, "missing required string tag " + std::to_string(key.getGroup()) + ":" + std::to_string(key.getElement()));
  return good ? std::string(value.c_str()) : std::string{};
}
double number(DcmItem& d, const DcmTagKey& key, unsigned long index = 0) {
  Float64 value = 0;
  need(d.findAndGetFloat64(key, value, index).good() && std::isfinite(value), "missing/nonfinite geometry or rescale value");
  return value;
}
std::uint16_t u16(DcmItem& d, const DcmTagKey& key) {
  Uint16 value = 0;
  need(d.findAndGetUint16(key, value).good(), "missing required unsigned-short tag");
  return value;
}
DcmSequenceOfItems* sequence(DcmItem& d, const DcmTagKey& tag, bool required = true) {
  DcmSequenceOfItems* value = nullptr;
  const bool good = d.findAndGetSequence(tag, value).good() && value;
  if (required) need(good, "missing required sequence");
  return good ? value : nullptr;
}
DcmItem* single(DcmItem& d, const DcmTagKey& tag, bool required = true) {
  auto* value = sequence(d, tag, required);
  if (!value) return nullptr;
  need(value->card() == 1, "expected one functional-group sequence item");
  return value->getItem(0);
}
DcmItem& group(DcmItem& frame, DcmItem* shared, const DcmTagKey& tag) {
  if (auto* item = single(frame, tag, false)) return *item;
  if (shared) if (auto* item = single(*shared, tag, false)) return *item;
  fail("required SEG functional-group geometry missing");
}
V3 triple(DcmItem& d, const DcmTagKey& tag, unsigned start = 0) {
  return {number(d, tag, start), number(d, tag, start + 1), number(d, tag, start + 2)};
}
double dot(const V3& a, const V3& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
V3 sub(const V3& a, const V3& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
V3 cross(const V3& a, const V3& b) { return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
bool close(double a, double b, double tolerance = 1e-4) { return std::abs(a-b) <= tolerance; }
bool close3(const V3& a, const V3& b, double tolerance = 1e-4) {
  return close(a[0],b[0],tolerance) && close(a[1],b[1],tolerance) && close(a[2],b[2],tolerance);
}
void orientation(const V3& r, const V3& c) {
  need(close(dot(r,r),1,1e-5) && close(dot(c,c),1,1e-5) && std::abs(dot(r,c)) < 1e-5,
       "nonorthonormal image orientation");
}
void load(DcmFileFormat& file, const std::filesystem::path& path) {
  auto result = file.loadFile(path.string().c_str());
  need(result.good(), "cannot read " + path.string() + ": " + result.text());
  need(DcmXfer(file.getDataset()->getOriginalXfer()).usesNativeFormat(), "compressed/encapsulated pixel data unsupported");
}
struct Slice {
  std::filesystem::path path;
  std::string sop;
  V3 position;
  double projection;
  float slope, intercept;
};
std::string source_uid(DcmItem& frame) {
  std::set<std::string> values;
  auto append = [&](DcmItem& d) {
    if (auto* sources = sequence(d, DCM_SourceImageSequence, false)) {
      for (unsigned long n = 0; n < sources->card(); ++n)
        values.insert(str(*sources->getItem(n), DCM_ReferencedSOPInstanceUID));
    }
  };
  append(frame);
  if (auto* derivations = sequence(frame, DCM_DerivationImageSequence, false))
    for (unsigned long n = 0; n < derivations->card(); ++n) append(*derivations->getItem(n));
  need(values.size() == 1, "each SEG frame must reference exactly one source CT SOP instance");
  return *values.begin();
}
}

ScanVolume read_segmentation_scan(const J& scan) {
  need(scan.is_object(), "scan descriptor must be an object");
  for (const char* key : {"ct_dir", "seg_path", "seed_x_pixel", "seed_y_pixel", "seed_z_imageposition_mm", "seed_pixel_index_base"})
    need(scan.contains(key), std::string("missing descriptor field ") + key);
  need(scan.at("seed_pixel_index_base").is_number_integer(), "seed_pixel_index_base must be integer 0 or 1");
  const int index_base = scan.at("seed_pixel_index_base").get<int>();
  need(index_base == 0 || index_base == 1, "seed_pixel_index_base must be 0 or 1");
  const auto ct_dir = std::filesystem::path(scan.at("ct_dir").get<std::string>());
  need(std::filesystem::is_directory(ct_dir), "CT directory does not exist");
  ScanVolume out;
  std::vector<Slice> slices;
  V3 normal{};
  std::string ct_series, ct_patient, ct_frame;
  std::set<std::string> ct_sops;
  for (const auto& entry : std::filesystem::directory_iterator(ct_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".dcm") continue;
    DcmFileFormat file; load(file, entry.path()); auto& d = *file.getDataset();
    need(str(d, DCM_Modality) == "CT", "non-CT file in CT directory");
    const auto frames = str(d, DCM_NumberOfFrames, false);
    need(frames.empty() || frames == "1", "enhanced/multiframe CT is unsupported");
    need(u16(d, DCM_SamplesPerPixel) == 1 && str(d, DCM_PhotometricInterpretation) == "MONOCHROME2", "CT must be single-channel MONOCHROME2");
    need(u16(d, DCM_BitsAllocated) == 16 && u16(d, DCM_PixelRepresentation) == 1,
         "CT must contain signed 16-bit allocated samples");
    const auto stored = u16(d, DCM_BitsStored);
    need(stored > 0 && stored <= 16 && u16(d, DCM_HighBit) == stored - 1, "unsupported CT stored-bit layout");
    const auto nx = u16(d, DCM_Columns), ny = u16(d, DCM_Rows);
    need(nx && ny, "empty CT dimensions");
    const V3 r = triple(d, DCM_ImageOrientationPatient), c = triple(d, DCM_ImageOrientationPatient, 3);
    orientation(r,c);
    const double sx = number(d, DCM_PixelSpacing, 1), sy = number(d, DCM_PixelSpacing, 0);
    need(sx > 0 && sy > 0, "nonpositive CT pixel spacing");
    const auto series = str(d, DCM_SeriesInstanceUID), patient = str(d, DCM_PatientID), frame = str(d, DCM_FrameOfReferenceUID);
    if (slices.empty()) {
      out.nx = nx; out.ny = ny; out.iopRow = r; out.iopCol = c; normal = cross(r,c);
      const double norm = std::sqrt(dot(normal,normal)); for (double& component : normal) component /= norm;
      out.spacingXYZ[0] = sx; out.spacingXYZ[1] = sy;
      ct_series = series; ct_patient = patient; ct_frame = frame;
    } else {
      need(nx == out.nx && ny == out.ny && close3(r,out.iopRow,1e-5) && close3(c,out.iopCol,1e-5)
           && close(sx,out.spacingXYZ[0]) && close(sy,out.spacingXYZ[1]), "inconsistent CT dimensions/orientation/pixel spacing");
      need(series == ct_series && patient == ct_patient && frame == ct_frame, "mixed CT series/patients/frame of reference");
    }
    Slice s; s.path = entry.path(); s.sop = str(d, DCM_SOPInstanceUID); s.position = triple(d, DCM_ImagePositionPatient);
    need(ct_sops.insert(s.sop).second, "duplicate CT SOP instance");
    s.projection = dot(s.position, normal);
    const double slope = number(d, DCM_RescaleSlope), intercept = number(d, DCM_RescaleIntercept);
    need(slope != 0 && std::abs(slope) <= std::numeric_limits<float>::max() && std::abs(intercept) <= std::numeric_limits<float>::max(), "invalid CT rescale metadata");
    s.slope = static_cast<float>(slope); s.intercept = static_cast<float>(intercept); slices.push_back(std::move(s));
  }
  need(slices.size() >= 2 && slices.size() <= UINT32_MAX, "need at least two CT planes");
  std::sort(slices.begin(), slices.end(), [](const Slice& a, const Slice& b) { return a.projection < b.projection; });
  out.nz = static_cast<std::uint32_t>(slices.size()); out.imageOriginLPS = slices.front().position;
  out.sliceOffsetsMM.resize(slices.size());
  double min_interval = std::numeric_limits<double>::infinity(), max_interval = 0;
  for (std::size_t z = 0; z < slices.size(); ++z) {
    out.sliceOffsetsMM[z] = slices[z].projection - slices[0].projection;
    if (z) {
      const double interval = out.sliceOffsetsMM[z] - out.sliceOffsetsMM[z-1];
      need(interval > 1e-5, "duplicate or nonincreasing CT plane positions");
      min_interval = std::min(min_interval,interval); max_interval = std::max(max_interval,interval);
    }
  }
  out.spacingXYZ[2] = min_interval;
  const double geometry_tolerance = std::max(1e-3, min_interval*1e-4);
  std::uint32_t irregular_intervals = 0;
  for (std::size_t z = 0; z < slices.size(); ++z) {
    const V3 delta = sub(slices[z].position, out.imageOriginLPS);
    const double offset = out.sliceOffsetsMM[z];
    const V3 wanted{normal[0]*offset, normal[1]*offset, normal[2]*offset};
    need(close3(delta, wanted, geometry_tolerance), "CT plane positions are not collinear with their normal; gantry shear unsupported");
    if (z && !close(out.sliceOffsetsMM[z]-out.sliceOffsetsMM[z-1],min_interval,geometry_tolerance)) ++irregular_intervals;
  }
  for (const char* key : {"ct_series_uid", "expected_ct_series_uid", "expectedCTUID"})
    if (scan.contains(key)) need(scan.at(key).get<std::string>() == ct_series, "CT series differs from descriptor");
  const std::size_t plane = std::size_t(out.nx) * out.ny;
  need(plane <= std::numeric_limits<std::size_t>::max() / out.nz, "volume extent overflow");
  const std::size_t voxels = plane * out.nz;
  out.rawPixels.resize(voxels); out.perSliceSlope.resize(out.nz); out.perSliceIntercept.resize(out.nz);
  std::map<std::string,std::size_t> index;
  for (std::size_t z = 0; z < slices.size(); ++z) {
    DcmFileFormat file; load(file, slices[z].path); auto& d = *file.getDataset();
    const Uint16* pixels = nullptr; unsigned long count = 0;
    need(d.findAndGetUint16Array(DCM_PixelData, pixels, &count).good() && pixels && count == plane, "CT pixel extent mismatch");
    const unsigned stored = u16(d, DCM_BitsStored);
    const std::uint32_t mask = (std::uint32_t(1) << stored) - 1, sign = std::uint32_t(1) << (stored - 1);
    for (std::size_t n = 0; n < plane; ++n) {
      const std::uint32_t bits = pixels[n] & mask;
      const std::int32_t value = (bits & sign) ? std::int32_t(bits) - std::int32_t(std::uint32_t(1) << stored) : std::int32_t(bits);
      out.rawPixels[z*plane+n] = static_cast<std::int16_t>(value);
    }
    out.perSliceSlope[z] = slices[z].slope; out.perSliceIntercept[z] = slices[z].intercept; index.emplace(slices[z].sop,z);
  }
  DcmFileFormat seg_file; load(seg_file, scan.at("seg_path").get<std::string>()); auto& seg = *seg_file.getDataset();
  need(str(seg, DCM_Modality) == "SEG" && str(seg, DCM_SegmentationType) == "BINARY", "only BINARY DICOM SEG is supported");
    need(str(seg, DCM_PatientID) == ct_patient, "SEG patient differs from CT");
  const auto seg_frame_uid = str(seg, DCM_FrameOfReferenceUID);
  const bool frame_uid_differs = seg_frame_uid != ct_frame;
  if (frame_uid_differs) {
    need(scan.contains("frame_of_reference_uid_mapping") && scan.at("frame_of_reference_uid_mapping").is_object(),
         "SEG frame of reference differs from CT; an explicit exact-UID mapping is required");
    const auto& mapping = scan.at("frame_of_reference_uid_mapping");
    need(mapping.value("ct", std::string{}) == ct_frame && mapping.value("seg", std::string{}) == seg_frame_uid
         && !mapping.value("reason", std::string{}).empty(), "frame-of-reference mapping must pin both actual UIDs and explain the exception");
  }
  need(u16(seg, DCM_Rows) == out.ny && u16(seg, DCM_Columns) == out.nx, "SEG dimensions differ from CT");
  need(u16(seg, DCM_BitsAllocated) == 1 && u16(seg, DCM_BitsStored) == 1 && u16(seg, DCM_HighBit) == 0
       && u16(seg, DCM_PixelRepresentation) == 0 && u16(seg, DCM_SamplesPerPixel) == 1, "unsupported SEG bit layout");
  auto* segments = sequence(seg, DCM_SegmentSequence);
  need(segments->card() == 1, "exactly one target segment is required");
  auto& segment = *segments->getItem(0); const auto segment_number = u16(segment, DCM_SegmentNumber);
  need(segment_number > 0, "invalid SEG target number");
  auto* referenced = sequence(seg, DCM_ReferencedSeriesSequence);
  need(referenced->card() == 1 && str(*referenced->getItem(0), DCM_SeriesInstanceUID) == ct_series, "SEG references a different or ambiguous CT series");
  auto* frames = sequence(seg, DCM_PerFrameFunctionalGroupsSequence);
  const auto frame_text = str(seg, DCM_NumberOfFrames);
  std::size_t parsed = 0; const auto frame_count = std::stoull(frame_text, &parsed);
  need(parsed == frame_text.size() && frame_count > 0 && frame_count == frames->card() && frame_count <= UINT32_MAX,
       "SEG frame count is invalid");
  auto* shared = single(seg, DCM_SharedFunctionalGroupsSequence, false);
  const Uint8* packed = nullptr; unsigned long bytes = 0;
  need(seg.findAndGetUint8Array(DCM_PixelData, packed, &bytes).good() && packed, "SEG binary pixels unavailable");
  need(plane <= (std::numeric_limits<std::size_t>::max()-7) / frame_count, "SEG packed extent overflow");
  const std::size_t required = (plane*frame_count+7)/8;
  need(bytes == required || (required%2 && bytes == required+1), "SEG packed pixel length mismatch");
  struct LabelFrame {std::size_t packed_frame; double offset;};
  std::vector<LabelFrame> label_frames; label_frames.reserve(frame_count);
  std::set<std::size_t> referenced_planes;
  double maximum_source_plane_distance = 0;
  for (std::size_t f = 0; f < frame_count; ++f) {
    auto& frame = *frames->getItem(static_cast<unsigned long>(f));
    auto& identification = group(frame,shared,DCM_SegmentIdentificationSequence);
    need(u16(identification,DCM_ReferencedSegmentNumber) == segment_number, "SEG frame uses an unexpected segment");
    const auto source = source_uid(frame); const auto found = index.find(source);
    need(found != index.end(), "SEG frame references CT instance absent from supplied series");
    const std::size_t z = found->second; referenced_planes.insert(z);
    auto& position = group(frame,shared,DCM_PlanePositionSequence);
    auto& orient = group(frame,shared,DCM_PlaneOrientationSequence);
    auto& measures = group(frame,shared,DCM_PixelMeasuresSequence);
    const V3 seg_position = triple(position,DCM_ImagePositionPatient);
    need(close3(triple(orient,DCM_ImageOrientationPatient),out.iopRow,1e-5)
         && close3(triple(orient,DCM_ImageOrientationPatient,3),out.iopCol,1e-5), "SEG orientation differs from CT");
    need(close(number(measures,DCM_PixelSpacing,1),out.spacingXYZ[0])
         && close(number(measures,DCM_PixelSpacing,0),out.spacingXYZ[1]), "SEG pixel spacing differs from CT");
    // A resampled SEG may own a different Z grid. Its referenced CT image is
    // provenance, not authority to overwrite the SEG's physical position.
    const double offset = dot(sub(seg_position,out.imageOriginLPS),normal);
    const V3 delta = sub(seg_position,slices[z].position);
    const double normal_delta = dot(delta,normal);
    need(close3(delta,V3{normal[0]*normal_delta,normal[1]*normal_delta,normal[2]*normal_delta},geometry_tolerance),
         "SEG in-plane origin differs from referenced CT geometry");
    const auto next = std::lower_bound(out.sliceOffsetsMM.begin(),out.sliceOffsetsMM.end(),offset);
    double nearest = std::numeric_limits<double>::infinity();
    if (next != out.sliceOffsetsMM.end()) nearest = std::abs(*next-offset);
    if (next != out.sliceOffsetsMM.begin()) nearest = std::min(nearest,std::abs(*std::prev(next)-offset));
    const double source_distance = std::abs(out.sliceOffsetsMM[z]-offset);
    need(source_distance <= nearest+geometry_tolerance,
         "SEG source SOP is not a physically nearest CT plane");
    const double low_bound = -0.5*(out.sliceOffsetsMM[1]-out.sliceOffsetsMM[0]);
    const double high_bound = out.sliceOffsetsMM.back()+0.5*(out.sliceOffsetsMM.back()-out.sliceOffsetsMM[out.nz-2]);
    need(offset >= low_bound-geometry_tolerance && offset <= high_bound+geometry_tolerance,
         "SEG plane lies outside source CT physical extent");
    maximum_source_plane_distance = std::max(maximum_source_plane_distance,source_distance);
    Float64 label_spacing = 0;
    if (measures.findAndGetFloat64(DCM_SpacingBetweenSlices,label_spacing).bad())
      label_spacing = number(measures,DCM_SliceThickness);
    need(std::isfinite(label_spacing) && label_spacing > 0, "SEG requires positive declared slice spacing");
    if (!f) out.labelSliceSpacingMM = label_spacing;
    else need(close(label_spacing,out.labelSliceSpacingMM,geometry_tolerance), "SEG declared spacing varies across frames");
    label_frames.push_back({f,offset});
  }
  std::sort(label_frames.begin(),label_frames.end(),[](const auto& a,const auto& b){return a.offset < b.offset;});
  out.labelNz = static_cast<std::uint32_t>(frame_count); out.labelSliceOffsetsMM.resize(frame_count);
  need(plane <= std::numeric_limits<std::size_t>::max()/out.labelNz, "SEG unpacked extent overflow");
  out.labels.resize(plane*out.labelNz);
  std::uint64_t positives = 0, omitted_label_planes = 0;
  for (std::size_t z = 0; z < label_frames.size(); ++z) {
    out.labelSliceOffsetsMM[z] = label_frames[z].offset;
    if (z) {
      const double interval = label_frames[z].offset-label_frames[z-1].offset;
      const double steps = std::round(interval/out.labelSliceSpacingMM);
      need(steps >= 1 && std::abs(interval-steps*out.labelSliceSpacingMM) <= geometry_tolerance,
           "SEG planes are duplicated or inconsistent with the declared physical grid");
      need(steps <= double(UINT32_MAX), "SEG omitted-plane extent exceeds supported range");
      omitted_label_planes += static_cast<std::uint64_t>(steps)-1;
    }
    for (std::size_t n = 0; n < plane; ++n) {
      const std::size_t bit = label_frames[z].packed_frame*plane+n;
      const auto value = std::uint8_t((packed[bit/8] >> (bit%8)) & 1u);
      out.labels[z*plane+n] = value; positives += value;
    }
  }
  need(positives > 0, "reference segment is empty");
  out.seedGridXYZ[0] = scan.at("seed_x_pixel").get<double>() - index_base;
  out.seedGridXYZ[1] = scan.at("seed_y_pixel").get<double>() - index_base;
  const double seed_z = scan.at("seed_z_imageposition_mm").get<double>();
  need(std::abs(normal[2]) > 1e-5, "image-position z cannot uniquely identify a plane for this orientation");
    out.seedProjectionOffsetMM = (seed_z-out.imageOriginLPS[2])/normal[2];
  need(std::isfinite(out.seedProjectionOffsetMM) && out.seedProjectionOffsetMM >= 0
       && out.seedProjectionOffsetMM <= out.sliceOffsetsMM.back(), "supplied seed plane lies outside the CT plane extent");
  auto upper = std::upper_bound(out.sliceOffsetsMM.begin(),out.sliceOffsetsMM.end(),out.seedProjectionOffsetMM);
  if (upper == out.sliceOffsetsMM.end()) out.seedGridXYZ[2] = double(out.nz-1);
  else {
    const std::size_t high = std::size_t(upper-out.sliceOffsetsMM.begin()), low = high-1;
    out.seedGridXYZ[2] = double(low) + (out.seedProjectionOffsetMM-out.sliceOffsetsMM[low])
        / (out.sliceOffsetsMM[high]-out.sliceOffsetsMM[low]);
  }
  for (unsigned c = 0; c < 3; ++c) {
    const double extent = c == 0 ? out.nx : c == 1 ? out.ny : out.nz;
    need(std::isfinite(out.seedGridXYZ[c]) && out.seedGridXYZ[c] >= 0 && out.seedGridXYZ[c] <= extent-1,
         "supplied lesion seed lies outside the CT image grid");
  }
  out.metadata = {{"format","dicom-segmentation-scan-1"},{"patient_id",ct_patient},{"ct_series_uid",ct_series},
        {"seg_series_uid",str(seg,DCM_SeriesInstanceUID)},{"frame_of_reference_uid",ct_frame},
    {"seg_frame_of_reference_uid",seg_frame_uid},{"frame_of_reference_uid_exception_used",frame_uid_differs},
    {"frame_of_reference_uid_exception_provenance",frame_uid_differs ? "explicit per-scan exact UID mapping; same patient and referenced series; all per-frame source SOP identities and geometry validated; no spatial transform" : "none"},
    {"frame_of_reference_uid_mapping",frame_uid_differs ? scan.at("frame_of_reference_uid_mapping") : J(nullptr)},
    {"segment_number",segment_number},{"segment_label",str(segment,DCM_SegmentLabel,false)},
    {"segment_algorithm_type",str(segment,DCM_SegmentAlgorithmType,false)},
        {"CT_slices",out.nz},{"SEG_frames",frame_count},{"reference_positive_voxels",positives},
    {"minimum_slice_interval_mm",min_interval},{"maximum_slice_interval_mm",max_interval},
    {"irregular_slice_intervals",irregular_intervals},{"uniform_slice_spacing",irregular_intervals==0},
    {"slice_offsets_mm",out.sliceOffsetsMM},{"seed_projection_offset_mm",out.seedProjectionOffsetMM},
        {"synthetic_planes_inserted",false},{"seed_grid_z_is_fractional_bracket_index",true},
    {"label_planes",out.labelNz},{"label_slice_offsets_mm",out.labelSliceOffsetsMM},
    {"label_slice_spacing_mm",out.labelSliceSpacingMM},{"omitted_empty_label_planes",omitted_label_planes},
    {"maximum_SEG_to_referenced_CT_plane_distance_mm",maximum_source_plane_distance},
    {"SEG_grid_policy","preserve physical SEG planes; source SOP must be a nearest CT plane; no snapping to CT planes"},
    {"seed_source","supplied descriptor coordinates; no mask-derived location"},{"seed_pixel_index_base",index_base},
    {"seed_z_semantics","LPS z coordinate of the requested plane image position"},
    {"storage_order","x fastest, then y, then z sorted along IOP row cross column"},
    {"label_outside_support","zero beyond half declared SEG spacing from every stored frame"},{"HU_conversion_performed",false}};
  return out;
}
}  // namespace class_study