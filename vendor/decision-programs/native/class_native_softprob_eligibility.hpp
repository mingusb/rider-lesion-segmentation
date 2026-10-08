#pragma once
// Shape/identity eligibility for an existing reviewed transform contract.
// This metadata helper does not authenticate a loaded kernel or grant class
// authority. The caller must supply the active model's parsed shape/identity.
// This inactive research helper is deliberately broader than RuntimeGate's
// qualified-source authorization. Its result must never enable pruning alone.
#include <cstdint>
#include <string>

namespace native_softprob_gap {
inline constexpr std::uint32_t reviewed_classes=7;
inline constexpr const char* reviewed_contract="native-softprob-sm80-seven-class-gap-v1";
inline bool valid_source_sha256(const std::string& source) noexcept {
  if(source.size()!=64)return false;
  for(const char c:source)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
  return true;
}
struct Eligibility {
  bool reviewed_shape_available=false;
  std::string reason;
};
inline Eligibility source_eligibility(const std::string& source,
    std::uint32_t classes,const std::string& objective) {
  if(!valid_source_sha256(source))return{false,"active source identity must be a lowercase SHA256"};
  if(objective!="multi:softprob")return{false,"this reviewed contract covers multi:softprob only; a native hard-argmax contract is separate"};
  if(classes!=reviewed_classes)return{false,"class count has no reviewed native transform control-flow/normal-division contract; the current capture review covers seven classes"};
  return{true,"reviewed seven-class transform shape only; generic native-margin correspondence is unqualified and RuntimeGate retains its qualified-source restriction"};
}
} // namespace native_softprob_gap
