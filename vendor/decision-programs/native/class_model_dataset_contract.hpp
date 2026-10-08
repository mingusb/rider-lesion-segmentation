#pragma once
// Pure metadata preflight. This does not read rows, compute predictions or
// perform numerical preprocessing; validated integers bind array extents.
#include <nlohmann/json.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace class_model_contract {
using json=nlohmann::json;
inline void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
inline std::uint64_t unsigned_integer(const json&value,const char*message){
 require(value.is_number_integer(),message);
 if(value.is_number_unsigned())return value.get<std::uint64_t>();
 auto number=value.get<std::int64_t>();require(number>=0,message);return std::uint64_t(number);
}
inline bool sha256(const json&value){if(!value.is_string())return false;const auto&s=value.get_ref<const std::string&>();if(s.size()!=64)return false;for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;return true;}
struct Dense {
 std::uint32_t features,classes;
 std::uint64_t rows,stride,fit_rows,valid_rows;
};
inline Dense dense(const json&b,bool fit_only=false){
 require(b.at("format").is_string()&&b.at("format")=="dense-fp32-u32-class-labels-1","dense format");
 require(b.at("TEST_read").is_boolean()&&!b.at("TEST_read").get<bool>(),"dense TEST role");
 const auto F=unsigned_integer(b.at("features"),"dense features must be unsigned integer"),K=unsigned_integer(b.at("classes"),"dense classes must be unsigned integer");
 require(F>0&&F<=std::uint64_t(INT32_MAX)&&K>=2&&K<=UINT32_MAX,"dense feature/class range");
 Dense out{std::uint32_t(F),std::uint32_t(K),unsigned_integer(b.at("rows"),"dense rows must be unsigned integer"),unsigned_integer(b.at("row_stride"),"dense stride must be unsigned integer"),unsigned_integer(b.at("FIT_rows"),"dense FIT must be unsigned integer"),unsigned_integer(b.at("VALID_rows"),"dense VALID must be unsigned integer")};
 require(out.rows>0&&out.stride>=F&&out.fit_rows>0&&out.fit_rows<=out.rows&&out.valid_rows==out.rows-out.fit_rows,"dense row/split range");
 require(fit_only ? out.fit_rows==out.rows&&out.valid_rows==0 : out.valid_rows>0,"dense FIT-only or evaluation boundary");
 require(out.rows<=UINT64_MAX/out.stride&&out.rows*out.stride<=SIZE_MAX/sizeof(float)&&out.rows<=SIZE_MAX/sizeof(std::uint32_t),"dense storage extent overflow");
 require(b.at("preprocessing").is_string()&&!b.at("preprocessing").get_ref<const std::string&>().empty(),"dense preprocessing identity");
 for(const auto*field:{"values_path","labels_path"})require(b.at(field).is_string()&&!b.at(field).get_ref<const std::string&>().empty(),"dense file path");
 require(sha256(b.at("values_sha256"))&&sha256(b.at("labels_sha256")),"dense byte pins must be lowercase SHA256");
 return out;
}
} // namespace class_model_contract