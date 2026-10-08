#pragma once
// Input-format transport only. Model numerics remain in CUDA backends.
#include "class_io.hpp"
#include <cctype>
#include <cstring>
#include <map>

namespace decision_programs_csv {
using J=nlohmann::json;
struct Table {std::vector<std::vector<std::string>> records;};
inline Table read(const std::string&bytes){
  Table out;std::vector<std::string>row;std::string field;bool quoted=false,closed=false;bool began=false;
  auto end_field=[&]{row.push_back(std::move(field));field.clear();closed=false;began=false;};
  auto end_row=[&]{end_field();if(row.size()!=1||!row[0].empty())out.records.push_back(std::move(row));row.clear();};
  for(std::size_t i=0;i<bytes.size();++i){char c=bytes[i];
    if(quoted){if(c=='"'){if(i+1<bytes.size()&&bytes[i+1]=='"'){field+='"';++i;}else{quoted=false;closed=true;}}else field+=c;continue;}
    if(c==','||c=='\n'||c=='\r'){if(c==',')end_field();else{if(c=='\r'&&i+1<bytes.size()&&bytes[i+1]=='\n')++i;end_row();}continue;}
    if(closed){if(c==' '||c=='\t')continue;throw std::invalid_argument("CSV characters after closing quote");}
    if(c=='"'){if(began||!field.empty())throw std::invalid_argument("CSV quote inside unquoted field");quoted=true;began=true;}else{field+=c;began=true;}
  }
  if(quoted)throw std::invalid_argument("CSV unterminated quoted field");if(began||closed||!field.empty()||!row.empty())end_row();
  if(out.records.empty())throw std::invalid_argument("CSV contains no records");auto width=out.records.front().size();
  for(const auto&record:out.records)if(record.size()!=width)throw std::invalid_argument("CSV rows have different column counts");return out;
}
inline std::string trim(std::string v){auto space=[](unsigned char c){return std::isspace(c);};while(!v.empty()&&space(v.front()))v.erase(v.begin());while(!v.empty()&&space(v.back()))v.pop_back();return v;}
inline bool feature(const std::string&text,float&value){
  auto token=trim(text);std::string lower=token;for(auto&c:lower)c=char(std::tolower(static_cast<unsigned char>(c)));
  if(lower.empty()||lower=="nan"||lower=="na"||lower=="null"){value=std::numeric_limits<float>::quiet_NaN();return true;}
  if(token.front()=='+')token.erase(token.begin());if(token.empty())return false;
  auto result=std::from_chars(token.data(),token.data()+token.size(),value,std::chars_format::general);
  return result.ec==std::errc{}&&result.ptr==token.data()+token.size()&&std::isfinite(value);
}
struct Dense {
  std::string values,labels;std::uint64_t rows=0;std::uint32_t features=0,classes=0;J names=J::array(),label_mapping=nullptr;bool header=false;
};
inline Dense decode(const std::string&bytes,const std::string&target={},const std::string&header_mode="auto"){
  auto table=read(bytes);Dense out;const auto columns=table.records[0].size();std::size_t target_index=columns;
  if(!target.empty()){
    auto found=std::find(table.records[0].begin(),table.records[0].end(),target);
    if(found!=table.records[0].end()){target_index=std::size_t(found-table.records[0].begin());out.header=true;}
    else if(target=="last")target_index=columns-1;
    else {auto n=std::from_chars(target.data(),target.data()+target.size(),target_index);if(n.ec!=std::errc{}||n.ptr!=target.data()+target.size()||target_index>=columns)throw std::invalid_argument("--target must name a CSV header or a zero-based column index (or last)");}
  }
  if(header_mode=="yes")out.header=true;else if(header_mode=="no"){if(out.header)throw std::invalid_argument("named --target requires a CSV header");out.header=false;}else if(header_mode=="auto"){
    if(!out.header)for(std::size_t c=0;c<columns;++c)if(c!=target_index){float value;if(!feature(table.records[0][c],value)){out.header=true;break;}}
  }else throw std::invalid_argument("--header must be auto, yes or no");
  if(columns<=std::size_t(!target.empty())||columns>UINT32_MAX)throw std::invalid_argument("CSV feature column count invalid");out.features=std::uint32_t(columns-std::size_t(!target.empty()));
  const std::size_t first=out.header?1:0;if(table.records.size()<=first)throw std::invalid_argument("CSV contains no data rows");out.rows=table.records.size()-first;
  if(out.rows>SIZE_MAX/(std::uint64_t(out.features)*4))throw std::invalid_argument("CSV input extent overflow");
  for(std::size_t c=0;c<columns;++c)if(c!=target_index)out.names.push_back(out.header?table.records[0][c]:"feature_"+std::to_string(c));
  std::vector<float>values;std::vector<std::uint32_t>labels;values.reserve(out.rows*out.features);labels.reserve(out.rows);std::map<std::string,std::uint32_t>mapping;bool numeric_labels=!target.empty();
  if(!target.empty())for(std::size_t r=first;r<table.records.size();++r){auto token=trim(table.records[r][target_index]);std::uint32_t label;auto n=std::from_chars(token.data(),token.data()+token.size(),label);if(n.ec!=std::errc{}||n.ptr!=token.data()+token.size()||token.empty()){numeric_labels=false;break;}}
  for(std::size_t r=first;r<table.records.size();++r){
    for(std::size_t c=0;c<columns;++c)if(c!=target_index){float value;if(!feature(table.records[r][c],value))throw std::invalid_argument("invalid finite-FP32 CSV feature at record "+std::to_string(r+1)+", column "+std::to_string(c));values.push_back(value);}
    if(!target.empty()){
      auto token=trim(table.records[r][target_index]);if(token.empty())throw std::invalid_argument("empty CSV target at record "+std::to_string(r+1));std::uint32_t label;
      if(numeric_labels){(void)std::from_chars(token.data(),token.data()+token.size(),label);if(label==UINT32_MAX)throw std::invalid_argument("CSV target exceeds supported class ID");}
      else {auto [it,inserted]=mapping.try_emplace(token,std::uint32_t(mapping.size()));label=it->second;}
      labels.push_back(label);out.classes=std::max(out.classes,label+1);
    }
  }
  out.values.assign(reinterpret_cast<const char*>(values.data()),values.size()*4);out.labels.assign(reinterpret_cast<const char*>(labels.data()),labels.size()*4);
  if(!target.empty()){
    out.label_mapping=J::array();if(!numeric_labels)for(const auto&[label,id]:mapping)out.label_mapping.push_back({{"label",label},{"class",id}});
  }
  return out;
}
} // namespace decision_programs_csv
