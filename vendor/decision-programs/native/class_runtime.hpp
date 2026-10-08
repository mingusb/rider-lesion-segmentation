#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace class_runtime {

// Persisted metadata is CLSGDAG1 v1. Numerical work occurs only in CUDA.
// Dense input rows contain declared model-input FP32 features; NaN follows each split's
// stored missing direction. Finite inputs plus NaN are the declared domain.
struct Metadata {
  std::uint32_t features, classes, root, nodes, predicates;
  // Compact dictionary/field metadata is zero when no compact layout was built.
  std::uint32_t child_bits, predicate_bits;
  std::uint64_t canonical_bytes, compact_bytes, device_bytes;
  std::string source_sha256, canonical_sha256, compact_sha256;
  bool canonical_resident, compact_resident, cross_layout_validated;
};

enum class Layout { canonical16, compact8 };
enum class Residency { dual, canonical_only, compact_only };
enum class Traversal { checked, validated };
struct DenseBatch {
  const float* device_values;
  std::uint64_t elements, first_row, rows, row_stride;
};
struct ClassOutput {
  std::uint32_t* device_classes;
  std::uint64_t elements;
};
// Exact visited node IDs, including the terminal. A row exceeding its declared
// capacity is rejected; partial paths are never reported as complete.
struct PathOutput {
  std::uint32_t* device_nodes;
  std::uint64_t node_elements, capacity_per_row;
  std::uint32_t* device_lengths;
  std::uint64_t length_elements;
};

// Owner is created only after whole canonical graph validation on CUDA.
// Compact residency additionally requires full cross-layout equality on CUDA.
// Device allocations remain private and immutable. The
// public API rejects invalid metadata/borrowed extents/aliasing before launch.
// Default stream only; calls synchronize before returning. Callers must retain
// exclusive ownership of borrowed buffers until the synchronous call returns.
class Runtime final {
 public:
  static std::unique_ptr<Runtime> load(
      const std::string& canonical_bytes, const std::string& canonical_sha256,
      const std::string& source_sha256,
      const std::string& compact_bytes = {}, const std::string& compact_sha256 = {},
      Residency = Residency::dual);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;
  const Metadata& metadata() const noexcept;
  // Release an unused validated layout without reloading or copying the model.
  // This operation cannot recreate a previously released layout.
  void retain(Residency);
  void predict(DenseBatch, ClassOutput, Layout, Traversal,
               std::uint32_t block_size = 256) const;
  void trace(DenseBatch, ClassOutput, PathOutput, Layout,
             std::uint32_t block_size = 256) const;
  // Performs device-side checks against supplied device labels for all resident
  // layout/traversal combinations. Throws on any mismatch or invalid route.
  void verify(DenseBatch, const std::uint32_t* device_expected,
              std::uint64_t expected_elements,
              std::uint32_t block_size = 256) const;
  // Byte transport only; packing is performed on CUDA. This format retains
  // threshold bit patterns (including +/-infinity), signed missing direction,
  // children and class IDs. Infinite model cuts do not expand the input domain.
  std::string compact_bytes() const;
 private:
  struct Impl;
  explicit Runtime(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

}  // namespace class_runtime
