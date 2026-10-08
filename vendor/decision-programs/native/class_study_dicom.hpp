#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>

namespace class_study {
// DICOM transport only. HU conversion, resampling and feature calculations are
// performed by the CUDA consumer. Storage is x-fastest, then y, then z.
struct ScanVolume {
  std::uint32_t nx = 0, ny = 0, nz = 0;
  std::vector<std::int16_t> rawPixels;
  std::vector<float> perSliceSlope, perSliceIntercept;
  std::vector<std::uint8_t> labels; // x/y shared with CT, z follows SEG planes.
  std::uint32_t labelNz = 0;
  std::vector<double> labelSliceOffsetsMM; // Relative to CT origin along normal.
  double labelSliceSpacingMM = 0; // Declared SEG spacing, including omitted empty frames.
  std::array<double, 3> spacingXYZ{}; // Z is nominal minimum interval only.
  // Actual sorted plane locations are authoritative; no planes are synthesized.
  std::vector<double> sliceOffsetsMM;
  double seedProjectionOffsetMM = 0;
  std::array<double, 3> imageOriginLPS{};
  // DICOM IOP first and second triples: increasing column x and row y.
  std::array<double, 3> iopRow{}, iopCol{};
  std::array<double, 3> seedGridXYZ{};
  nlohmann::json metadata;
};
// Required: ct_dir, seg_path, seed_x_pixel, seed_y_pixel,
// seed_z_imageposition_mm, seed_pixel_index_base (0 or 1).
// Supports parallel-plane single-frame signed-16 CT and one-target binary SEG.
// Nonuniform slice spacing is explicit in sliceOffsetsMM; consumers must use it.
// A differing CT/SEG FrameOfReferenceUID requires an explicit
// frame_of_reference_uid_mapping object {ct: exactUID, seg: exactUID, reason: text};
// source identities and in-plane geometry must match; each source CT plane
// must be physically nearest to its referenced SEG plane. SEG owns its Z grid.
// Fails closed on unsupported transfer syntax or mismatched geometry.
ScanVolume read_segmentation_scan(const nlohmann::json& scan);
}  // namespace class_study