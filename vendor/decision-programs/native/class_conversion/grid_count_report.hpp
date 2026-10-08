#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Host reporting metadata only. The coverage interval itself is computed on CUDA.
// Convert its exact binary64 endpoints to integer cell bounds without rounding
// the total grid through binary64 or imposing a 64-bit count limit.
namespace class_grid_report {
class Integer {
  static constexpr std::uint64_t base = 1000000000;
  std::vector<std::uint32_t> digits_{0};
  void trim() { while (digits_.size() > 1 && !digits_.back()) digits_.pop_back(); }
public:
  explicit Integer(std::uint32_t value = 0) : digits_{std::uint32_t(value % base)} {
    if (value >= base) digits_.push_back(std::uint32_t(value / base));
  }
  void multiply(std::uint64_t value) {
    if (value > (std::uint64_t{1} << 32) + 1)
      throw std::invalid_argument("grid reporting factor extent");
    std::uint64_t carry = 0;
    for (auto& digit : digits_) {
      const auto product = digit * value + carry;
      digit = std::uint32_t(product % base); carry = product / base;
    }
    while (carry) { digits_.push_back(std::uint32_t(carry % base)); carry /= base; }
    trim();
  }
  void add(const Integer& other) {
    if (digits_.size() < other.digits_.size()) digits_.resize(other.digits_.size());
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < digits_.size(); ++i) {
      const auto sum = std::uint64_t(digits_[i]) + carry +
          (i < other.digits_.size() ? other.digits_[i] : 0);
      digits_[i] = std::uint32_t(sum % base); carry = sum / base;
    }
    if (carry) digits_.push_back(std::uint32_t(carry));
  }
  bool less(const Integer& other) const {
    if (digits_.size() != other.digits_.size()) return digits_.size() < other.digits_.size();
    for (std::size_t i = digits_.size(); i--;) {
      if (digits_[i] != other.digits_[i]) return digits_[i] < other.digits_[i];
    }
    return false;
  }
  Integer subtract(const Integer& other) const {
    if (less(other)) throw std::invalid_argument("negative grid reporting count");
    auto result = *this;
    std::int64_t borrow = 0;
    for (std::size_t i = 0; i < digits_.size(); ++i) {
      auto value = std::int64_t(digits_[i]) - borrow -
          (i < other.digits_.size() ? other.digits_[i] : 0);
      borrow = value < 0;
      if (borrow) value += std::int64_t(base);
      result.digits_[i] = std::uint32_t(value);
    }
    result.trim(); return result;
  }
  bool halve() {
    std::uint32_t remainder = 0;
    for (std::size_t i = digits_.size(); i--;) {
      const auto value = std::uint64_t(remainder) * base + digits_[i];
      digits_[i] = std::uint32_t(value / 2); remainder = std::uint32_t(value % 2);
    }
    trim(); return remainder != 0;
  }
  std::string decimal() const {
    auto text = std::to_string(digits_.back());
    for (std::size_t i = digits_.size() - 1; i;) {
      auto part = std::to_string(digits_[--i]); text.append(9 - part.size(), '0'); text += part;
    }
    return text;
  }
};
inline Integer product(const std::vector<std::uint64_t>& factors) {
  Integer result(1);
  for (auto value : factors) {
    if (!value) throw std::invalid_argument("empty grid reporting axis");
    result.multiply(value);
  }
  return result;
}
inline Integer scaled(const Integer& total, double fraction, bool round_up) {
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
  if (!std::isfinite(fraction) || fraction < 0 || fraction > 1)
    throw std::invalid_argument("invalid grid reporting fraction");
  const auto bits = std::bit_cast<std::uint64_t>(fraction);
  const auto exponent = unsigned((bits >> 52) & 0x7ff);
  const auto mantissa = (bits & ((std::uint64_t{1} << 52) - 1)) |
      (exponent ? std::uint64_t{1} << 52 : 0);
  auto numerator = total, low = total;
  numerator.multiply(mantissa >> 32);
  numerator.multiply(std::uint64_t{1} << 32);
  low.multiply(std::uint32_t(mantissa)); numerator.add(low);
  const auto denominator_bits = exponent ? 1075 - exponent : 1074;
  bool remainder = false;
  for (unsigned i = 0; i < denominator_bits; ++i) remainder = numerator.halve() || remainder;
  if (round_up && remainder) numerator.add(Integer(1));
  return numerator;
}
struct Bounds {
  std::string settled_lower, settled_upper, remaining_lower, remaining_upper;
};
inline Bounds bounds(const Integer& total, double lower, double upper) {
  if (lower > upper) throw std::invalid_argument("reversed grid coverage interval");
  const auto lo = scaled(total, lower, true), hi = scaled(total, upper, false);
  if (hi.less(lo)) throw std::invalid_argument("grid coverage interval contains no integer cell count");
  return {lo.decimal(), hi.decimal(), total.subtract(hi).decimal(), total.subtract(lo).decimal()};
}
} // namespace class_grid_report
