#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <vector>
#include <span>
#include <string>
namespace rank_disk_ledger {
inline constexpr std::uint64_t none=UINT64_MAX;
// These are exact opaque numerical words. The ledger never evaluates a split.
// kind: 0 axis, 1 weighted plane, 2 completed native-supported leaf.
struct Node {std::uint64_t id=none,left=none,right=none,first_term=0;std::uint32_t term_count=0;std::int32_t label=-1,kind=2,feature=-1;std::uint32_t cut_bits=0;std::uint64_t threshold_bits=0;bool operator==(const Node&)const=default;};
struct Term {std::uint64_t id=none;std::int32_t feature=0;std::uint32_t weight_bits=0;bool operator==(const Term&)const=default;};
struct Binding {std::string source_sha256,builder_sha256;bool operator==(const Binding&)const=default;};
struct Options {std::uint64_t maximum_page_payload=4*1024*1024;};
struct Prefix {std::uint64_t pages=0,nodes=0,node_high_water=0,terms=0,committed_bytes=0,partial_tail_bytes=0;std::string chain_sha256;bool operator==(const Prefix&)const=default;};
struct Sink {std::function<void(const Node&)>node;std::function<void(const Term&)>term;};
// A reader validates a complete page before delivering any of its records. A
// later corrupt page throws; already delivered earlier committed pages remain
// valid. Only an incomplete final page is ignored and explicitly counted.
Prefix read(const std::filesystem::path&,const Binding&,const Sink& = {},Options = {});
// Read an authenticated immutable committed prefix while its Writer may append.
// No file lock is taken. Only expected.committed_bytes is read; later complete
// or partial appends are ignored. Binding, expected and sinks are snapshotted.
// Every page and the exact final counts/high-water/chain must match expected,
// whose partial_tail_bytes must be zero. Sink effects are PROVISIONAL until
// successful return; no numerical/source acceptance is established here.
// Caller authenticates expected and rechecks its head before publication.
// Committed bytes must remain immutable throughout; this is not protection
// against a hostile concurrent writer that rewrites authenticated old bytes.
Prefix read_prefix(const std::filesystem::path&,Binding,Prefix expected,Sink = {},Options = {});

class Writer {struct Impl;std::unique_ptr<Impl>p_;explicit Writer(std::unique_ptr<Impl>);
public:
 static Writer create(const std::filesystem::path&,Binding,Options={});
 // Resume validates every committed page, rejects corruption/binding changes,
 // and truncates only the recognized incomplete trailing page under an exclusive
 // file lock. Read/recovery memory is bounded by maximum_page_payload.
 static Writer resume(const std::filesystem::path&,Binding,Options={});
 ~Writer();Writer(Writer&&)noexcept;Writer&operator=(Writer&&)noexcept;
 Writer(const Writer&)=delete;Writer&operator=(const Writer&)=delete;
 // One bounded batch is one durable page. Node IDs may arrive in arbitrary stable-ID order (disk-indexed duplicate
 // detection); term IDs are consecutive. Forward child/term refs are permitted;
 // no unsupported leaf
 // placeholder is permitted. On any I/O error this handle becomes unusable;
 // reopening resolves the valid prefix (a failed sync has uncertain durability).
 Prefix append(std::span<const Node>,std::span<const Term>);
 Prefix prefix()const;
 std::optional<Node> node(std::uint64_t id)const;
 std::optional<Term> term(std::uint64_t id)const;
};
struct Sealed {Prefix prefix;std::uint64_t root_id=0;Binding binding;};
// A seal checks dense stable IDs, all references, unique parents, root reachability
// and term extents using disk-backed indices/queues. It performs no routing,
// feature/weight evaluation, source qualification or class proof. INDEX must be new.
Sealed seal(const std::filesystem::path&ledger,const std::filesystem::path&index,const Binding&,std::uint64_t root_id,Options={});
struct PageOptions {std::uint32_t nodes_per_page=1024;std::uint64_t maximum_terms=65536;};
struct NodePage {std::uint64_t page_id=0,first_id=0;std::vector<Node>nodes;std::vector<Term>terms;std::vector<std::uint64_t>original_first_term;};
class PagedReader {struct Impl;std::unique_ptr<Impl>p_;public:
 PagedReader(const std::filesystem::path&ledger,const std::filesystem::path&index,const Binding&,Options={},PageOptions={});
 ~PagedReader();PagedReader(PagedReader&&)noexcept;PagedReader&operator=(PagedReader&&)noexcept;
 PagedReader(const PagedReader&)=delete;PagedReader&operator=(const PagedReader&)=delete;
 const Sealed&metadata()const;
 // Stable-ID page; offsets are structurally remapped into its bounded local term
 // pool. Exact term IDs/words and original offsets are retained alongside nodes.
 NodePage read_page(std::uint64_t page_id)const;
};
// This is a storage ledger, not a complete-model source certificate: all referenced IDs,
// unique-parent/root closure and numerical/source acceptance must be validated
// independently before any exported model can be described as accepted.
}
