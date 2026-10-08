#pragma once
// Display-only decoding of validated model words. These functions neither
// evaluate a predicate nor alter model/trace data.
#include <nlohmann/json.hpp>
#include <bit>
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace decision_programs_display {
inline std::string threshold_text(std::uint32_t bits) {
  const float value=std::bit_cast<float>(bits);char buffer[64];
  auto result=std::to_chars(buffer,buffer+sizeof(buffer),value,
                            std::chars_format::general,std::numeric_limits<float>::max_digits10);
  if(result.ec!=std::errc{})throw std::runtime_error("cannot format FP32 threshold");
  return {buffer,result.ptr};
}
inline std::uint32_t feature_index(std::int32_t feature) {
  return feature>=0?std::uint32_t(feature):std::uint32_t(-std::int64_t(feature)-2);
}
inline std::string condition(std::int32_t feature,std::uint32_t bits) {
  auto input="x["+std::to_string(feature_index(feature))+"]";
  return "(isnan("+input+") ? "+(feature<0?"true":"false")+" : "+input+" < "+threshold_text(bits)+")";
}
inline nlohmann::json threshold_fields(std::int32_t feature,std::uint32_t bits) {
  auto input="x["+std::to_string(feature_index(feature))+"]";
  return {{"threshold",std::bit_cast<float>(bits)},{"threshold_text",threshold_text(bits)},
    {"decision_rule","NaN -> "+std::string(feature<0?"left":"right")+"; "+input+" < "+threshold_text(bits)+" -> left; otherwise -> right"}};
}
} // namespace decision_programs_display
