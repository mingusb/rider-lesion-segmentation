#include "class_tree_adapter.hpp"
#include "class_io.hpp"

#include <bit>
#include <climits>
#include <limits>
#include <utility>

namespace class_tree_adapter {
namespace {
using U = std::uint64_t;
using W = std::uint32_t;
constexpr U none = std::numeric_limits<U>::max();
void need(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
U word(const std::string& b, U at, unsigned n) {
  need(at <= b.size() && n <= b.size() - at, "truncated adapter word");
  U value = 0;
  for (unsigned i = 0; i < n; ++i)
    value |= U(static_cast<unsigned char>(b[at + i])) << (8 * i);
  return value;
}
void append(std::string& b, U value, unsigned n) {
  for (unsigned i = 0; i < n; ++i) b.push_back(char(value >> (8 * i)));
}
bool digest(const std::string& s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
std::string raw_digest(const std::string& s) {
  need(digest(s), "requires lowercase 64-character SHA256");
  std::string out;
  auto value = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  for (unsigned i = 0; i < 64; i += 2)
    out.push_back(char(value(s[i]) * 16 + value(s[i + 1])));
  return out;
}
struct Old {
  std::int32_t feature;
  W cut;
  unsigned missing;
  U left, right;
  std::int64_t label;
};
struct Tree {
  W f, k;
  U depth, leaves = 0;
  std::vector<Old> node;
  std::vector<W> postorder;
};
Tree read(const std::string& b, Limits limits) {
  need(limits.maximum_nodes > 0 && limits.maximum_nodes <= INT32_MAX,
       "invalid explicit adapter node capacity");
  need(b.size() >= 40 && b.compare(0, 8, "CLSTREE1") == 0,
       "requires CLSTREE1; other predicate/schema formats are unsupported");
  U f = word(b, 8, 8), k = word(b, 16, 8), n = word(b, 24, 8);
  need(f >= 1 && f <= INT32_MAX && k >= 2 && k <= UINT32_MAX,
       "dimensions exceed signed feature/U32 class representation");
  need(n > 0 && n <= limits.maximum_nodes && n <= (SIZE_MAX - 40) / 33
           && b.size() == 40 + 33 * n,
       "CLSTREE1 extent/count/capacity differs");
  Tree t{W(f), W(k), word(b, 32, 8), 0, {}, {}};
  need(t.depth < n, "declared depth exceeds a tree bound");
  t.node.resize(n);
  std::vector<W> parents(n, 0);
  for (U i = 0; i < n; ++i) {
    auto& v = t.node[i];
    v.feature = std::bit_cast<std::int32_t>(W(word(b, 40 + 4*i, 4)));
    v.cut = W(word(b, 40 + 4*n + 4*i, 4));
    v.missing = unsigned(word(b, 40 + 8*n + i, 1));
    v.left = word(b, 40 + 9*n + 8*i, 8);
    v.right = word(b, 40 + 17*n + 8*i, 8);
    v.label = std::bit_cast<std::int64_t>(word(b, 40 + 25*n + 8*i, 8));
    if (v.label >= 0) {
      need(U(v.label) < k && v.left == none && v.right == none,
           "unresolved/out-of-range terminal or terminal successors");
      ++t.leaves;
    } else {
      need(v.label == -1 && v.feature >= 0 && U(v.feature) < f,
           "branch label/feature differs from completed axis schema");
      need(v.missing <= 1 && (v.cut & 0x7fffffffU) <= 0x7f800000U,
           "NaN cut or invalid missing direction is unsupported");
      need(v.left < n && v.right < n && v.left != v.right,
           "branch children are invalid");
      need(++parents[v.left] == 1 && ++parents[v.right] == 1,
           "shared physical parents are unsupported by CLSTREE1");
    }
  }
  need(parents[0] == 0 && std::all_of(parents.begin()+1, parents.end(),
       [](W p) { return p == 1; }), "root/unique-parent identity differs");
  struct Frame { W id; U depth; bool exit; };
  std::vector<Frame> stack{{0, 0, false}};
  std::vector<unsigned char> seen(n, 0);
  U reached = 0, depth = 0;
  t.postorder.reserve(n);
  while (!stack.empty()) {
    auto frame = stack.back(); stack.pop_back();
    if (frame.exit) { t.postorder.push_back(frame.id); continue; }
    need(!seen[frame.id], "cycle or shared reachable node");
    seen[frame.id] = 1; ++reached;
    depth = std::max(depth, frame.depth);
    const auto& v = t.node[frame.id];
    stack.push_back({frame.id, frame.depth, true});
    if (v.label < 0) {
      stack.push_back({W(v.right), frame.depth+1, false});
      stack.push_back({W(v.left), frame.depth+1, false});
    }
  }
  need(reached == n && depth == t.depth && n == 2*t.leaves-1,
       "reachability/depth/full-binary-tree identity differs");
  return t;
}
std::string encode_old(const Tree& t) {
  std::string b = "CLSTREE1";
  append(b,t.f,8); append(b,t.k,8); append(b,t.node.size(),8); append(b,t.depth,8);
  for (const auto& v:t.node) append(b,W(v.feature),4);
  for (const auto& v:t.node) append(b,v.cut,4);
  for (const auto& v:t.node) append(b,v.missing,1);
  for (const auto& v:t.node) append(b,v.left,8);
  for (const auto& v:t.node) append(b,v.right,8);
  for (const auto& v:t.node) append(b,U(v.label),8);
  return b;
}
}  // namespace

Result adapt(const std::string& b, const std::string& sha, Limits limits) {
  need(digest(sha) && dpnative::sha256(b) == sha, "original SHA256 differs");
  auto t = read(b, limits);
  Result r{t.f,t.k,W(t.node.size()),W(t.node.size()-1),t.depth,t.leaves,
           U(t.node.size())-t.leaves,sha,{},{},{}};
  r.original_to_canonical.resize(t.node.size());
  for (W i=0;i<r.nodes;++i) r.original_to_canonical[t.postorder[i]]=i;
  auto& out=r.canonical_bytes;
  out="CLSGDAG1"; append(out,1,4); append(out,r.features,4); append(out,r.classes,4);
  append(out,r.root,4); append(out,r.nodes,4); append(out,1,4);
  out += raw_digest(sha);
  for (W i=0;i<r.nodes;++i) {
    const auto& v=t.node[t.postorder[i]];
    if (v.label>=0) {
      append(out,W(-1),4); append(out,U(v.label),4); append(out,0,4); append(out,0,4);
    } else {
      std::int32_t sf=v.missing ? -2-v.feature : v.feature;
      W left=r.original_to_canonical[v.left], right=r.original_to_canonical[v.right];
      need(left<i && right<i, "internal postorder failure");
      append(out,W(sf),4); append(out,v.cut,4); append(out,left,4); append(out,right,4);
    }
  }
  need(out.size()==64+16*U(r.nodes), "canonical byte accounting differs");
  r.canonical_sha256=dpnative::sha256(out);
  need(inverse(r,b)==b, "exact inverse transport differs");
  return r;
}

std::string inverse(const Result& r, const std::string& b) {
  need(digest(r.original_sha256) && dpnative::sha256(b)==r.original_sha256,
       "inverse original SHA256 differs");
  auto t=read(b, {r.nodes});
  const auto& c=r.canonical_bytes;
  need(digest(r.canonical_sha256) && dpnative::sha256(c)==r.canonical_sha256,
       "inverse canonical SHA256 differs");
  need(c.size()==64+16*U(r.nodes) && c.compare(0,8,"CLSGDAG1")==0
       && word(c,8,4)==1 && word(c,12,4)==t.f && word(c,16,4)==t.k
       && word(c,20,4)==r.root && word(c,24,4)==t.node.size()
       && word(c,28,4)==1 && c.substr(32,32)==raw_digest(r.original_sha256),
       "inverse canonical header/identity differs");
  need(r.features==t.f && r.classes==t.k && r.nodes==t.node.size()
       && r.maximum_depth==t.depth && r.leaves==t.leaves
       && r.branches==t.node.size()-t.leaves
       && r.original_to_canonical.size()==t.node.size(), "inverse metadata differs");
  std::vector<W> back(r.nodes, UINT32_MAX);
  for (W old=0;old<r.nodes;++old) {
    W id=r.original_to_canonical[old];
    need(id<r.nodes && back[id]==UINT32_MAX, "inverse mapping is not bijective");
    back[id]=old;
  }
  need(r.original_to_canonical[0]==r.root, "inverse designated root differs");
  for (W old=0;old<r.nodes;++old) {
    auto& v=t.node[old]; W id=r.original_to_canonical[old]; U at=64+16*U(id);
    auto sf=std::bit_cast<std::int32_t>(W(word(c,at,4)));
    W payload=W(word(c,at+4,4)), left=W(word(c,at+8,4)), right=W(word(c,at+12,4));
    if (v.label>=0) {
      need(sf==-1 && payload==U(v.label) && left==0 && right==0,
           "inverse class leaf differs");
      v.label=payload;
    } else {
      need(sf!=-1 && left<id && right<id && left!=right,
           "inverse canonical topology differs");
      std::int64_t f=sf>=0 ? sf : -std::int64_t(sf)-2;
      need(f==v.feature && payload==v.cut && unsigned(sf<0)==v.missing
           && back[left]==v.left && back[right]==v.right,
           "inverse active predicate/children words differ");
      v.feature=std::int32_t(f); v.cut=payload; v.missing=sf<0;
      v.left=back[left]; v.right=back[right];
    }
  }
  auto restored=encode_old(t);
  need(restored==b, "inverse original bytes differ");
  return restored;
}
}  // namespace class_tree_adapter
