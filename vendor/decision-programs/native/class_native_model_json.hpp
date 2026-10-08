#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// XGBoost's native text format extends JSON with bare Infinity, -Infinity and
// NaN numeric tokens. Keep IEEE values in the structural view; untouched model
// bytes remain the authority for loading and identity. This codec is not used
// for plans/receipts, and never substitutes a finite threshold for infinity.
namespace dp_native_json {
using J = nlohmann::json;
namespace detail {
inline std::size_t string_end(std::string_view s, std::size_t begin) {
    for (auto i = begin + 1; i < s.size(); ++i) {
        if (s[i] == '\\') { if (++i == s.size()) break; }
        else if (s[i] == '"') return i + 1;
    }
    throw std::invalid_argument("unterminated native JSON string");
}
inline bool space(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
struct Token { std::size_t begin, end; unsigned kind; };
inline std::array<std::string,3> markers(const std::set<std::string>& strings) {
    std::string prefix = "\x1f" "dp-native-IEEE:";
    for (;;) {
        std::array<std::string,3> result{prefix+"+",prefix+"-",prefix+"N"};
        if (!strings.count(result[0]) && !strings.count(result[1]) && !strings.count(result[2])) return result;
        prefix += ':';
    }
}
inline double number(unsigned kind) {
    return kind == 0 ? std::numeric_limits<double>::infinity() :
           kind == 1 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
}
inline void restore(J& value, const std::array<std::string,3>& names) {
    if (value.is_string()) {
        for (unsigned k=0;k<3;++k) if (value.get_ref<const std::string&>() == names[k]) { value=number(k);return; }
    } else if (value.is_structured()) for (auto& child:value) restore(child,names);
}
inline void collect(const J& value,std::set<std::string>& strings) {
    if(value.is_string())strings.insert(value.get<std::string>());
    else if(value.is_object())for(auto it=value.begin();it!=value.end();++it){strings.insert(it.key());collect(it.value(),strings);}
    else if(value.is_array())for(const auto& child:value)collect(child,strings);
}
inline void replace(J& value,const std::array<std::string,3>& names) {
    if(value.is_number_float()){
        const double x=value.get<double>();if(!std::isfinite(x))value=names[std::isnan(x)?2:std::signbit(x)?1:0];
    }else if(value.is_structured())for(auto& child:value)replace(child,names);
}
} // namespace detail
inline J parse(std::string_view text) {
    if(text.find("Infinity")==std::string_view::npos&&text.find("NaN")==std::string_view::npos)return J::parse(text);
    std::set<std::string> strings;std::vector<detail::Token> tokens;
    constexpr std::array<std::string_view,4> words{"Infinity","-Infinity","NaN","+Infinity"};
    for(std::size_t i=0;i<text.size();){
        if(text[i]=='"'){auto end=detail::string_end(text,i);strings.insert(J::parse(text.substr(i,end-i)).get<std::string>());i=end;continue;}
        bool found=false;
        for(unsigned k=0;k<4;++k){const auto word=words[k];
            if(text.substr(i,word.size())!=word)continue;
            const auto end=i+word.size();
            const bool before=i==0||detail::space(text[i-1])||text[i-1]=='['||text[i-1]==','||text[i-1]==':';
            const bool after=end==text.size()||detail::space(text[end])||text[end]==']'||text[end]=='}'||text[end]==',';
            if(before&&after){auto following=end;while(following<text.size()&&detail::space(text[following]))++following;if(following<text.size()&&text[following]==':')throw std::invalid_argument("native IEEE token cannot be an object key");tokens.push_back({i,end,k==3?0u:k});i=end;found=true;break;}
        }
        if(!found)++i;
    }
    if(tokens.empty())return J::parse(text);
    const auto names=detail::markers(strings);std::string sanitized;sanitized.reserve(text.size());std::size_t next=0;
    for(const auto& token:tokens){sanitized.append(text.substr(next,token.begin-next));sanitized+=J(names[token.kind]).dump();next=token.end;}
    sanitized.append(text.substr(next));auto value=J::parse(sanitized);detail::restore(value,names);return value;
}
// Ordinary json::dump turns nonfinite numbers into null. Use this only when
// deliberately rewriting native model metadata (for example feature remapping).
inline std::string dump(const J& document) {
    std::set<std::string> strings;detail::collect(document,strings);const auto names=detail::markers(strings);
    J encoded=document;detail::replace(encoded,names);const auto text=encoded.dump();std::string result;result.reserve(text.size());
    constexpr std::array<std::string_view,3> words{"Infinity","-Infinity","NaN"};
    const std::array<std::string,3> quoted{J(names[0]).dump(),J(names[1]).dump(),J(names[2]).dump()};
    for(std::size_t i=0;i<text.size();){
        if(text[i]!='"'){result+=text[i++];continue;}
        const auto end=detail::string_end(text,i);const std::string_view token(text.data()+i,end-i);bool special=false;
        for(unsigned k=0;k<3;++k)if(token==quoted[k]){result+=words[k];special=true;break;}
        if(!special)result+=token;i=end;
    }
    return result;
}
} // namespace dp_native_json
