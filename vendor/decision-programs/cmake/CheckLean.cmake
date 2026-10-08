# Portable elaboration and independent kernel replay, without private tools.
if(NOT DEFINED DP_PROOF_SOURCE_ROOT OR NOT DEFINED DP_PROOF_OUTPUT_ROOT)
  message(FATAL_ERROR "Proof source and output directories are required")
endif()
if(NOT DP_LEAN_EXECUTABLE)
  find_program(dp_lean_from_path lean REQUIRED)
  set(DP_LEAN_EXECUTABLE "${dp_lean_from_path}")
endif()
get_filename_component(dp_lean_bin "${DP_LEAN_EXECUTABLE}" DIRECTORY)
find_program(dp_leanchecker leanchecker HINTS "${dp_lean_bin}" REQUIRED)
execute_process(COMMAND "${DP_LEAN_EXECUTABLE}" --short-version
  OUTPUT_VARIABLE dp_lean_version OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE dp_result)
if(NOT dp_result EQUAL 0 OR NOT dp_lean_version STREQUAL "4.34.1")
  message(FATAL_ERROR "These modules require Lean 4.34.1; found ${dp_lean_version}")
endif()
file(GLOB_RECURSE dp_toolchains "${DP_PROOF_SOURCE_ROOT}/lean-toolchain")
foreach(dp_toolchain IN LISTS dp_toolchains)
  file(READ "${dp_toolchain}" dp_declared)
  string(STRIP "${dp_declared}" dp_declared)
  if(NOT dp_declared STREQUAL "leanprover/lean4:v4.34.1")
    message(FATAL_ERROR "Unsupported proof toolchain: ${dp_toolchain}")
  endif()
endforeach()
# Actual local import order; duplicate DynamicAtomRestriction copies have the
# same module identity. TacticProbe playgrounds are not maintained proofs.
set(dp_modules
  balanced_bins/BalancedBins
  region_proof_synthesis/CoverRefinement region_proof_synthesis/RegionEnvelope
  score_diagram_congruence/ScoreDiagramCongruence
  score_add_apply_congruence/ScoreAddApplyCongruence
  class_apply_congruence/ClassApplyCongruence
  dynamic_atom_restriction/DynamicAtomRestriction
  adjacent_predicate_bypass/AdjacentPredicateBypass literal_path_bound/LiteralPathBound
  numeric_triple_sweep/NumericTripleSweep ordered_branch_laws/OrderedBranchLaws
  structural_factoring/TreeFactor
  converter_guarantees/CorrectnessCompletion converter_guarantees/RankBoxProgress
  converter_guarantees/SizeBounds converter_guarantees/GraftBounds
  converter_guarantees/SharedDecisionDAG converter_guarantees/DecisionEquations
  converter_guarantees/OnlineArenaContracts converter_guarantees/CollectedRoots
  converter_guarantees/SignatureEnumeration converter_guarantees/MixedRadixCoverage
  converter_guarantees/StreamingGridFold converter_guarantees/RankSourceCorrespondence
  converter_guarantees/ConcreteConstruction converter_guarantees/ForestRankInstance
  converter_guarantees/ApplicabilityCache converter_guarantees/HardAxisRewrites
  converter_guarantees/HardRewriteExhaustion converter_guarantees/OrderedArithmetic
  converter_guarantees/PairedBounds)
file(SHA256 "${DP_PROOF_SOURCE_ROOT}/dynamic_atom_restriction/DynamicAtomRestriction.lean" dp_dynamic_hash)
foreach(dp_duplicate adjacent_predicate_bypass literal_path_bound)
  file(SHA256 "${DP_PROOF_SOURCE_ROOT}/${dp_duplicate}/DynamicAtomRestriction.lean" dp_duplicate_hash)
  if(NOT dp_duplicate_hash STREQUAL dp_dynamic_hash)
    message(FATAL_ERROR "Duplicate DynamicAtomRestriction sources differ")
  endif()
endforeach()
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef dp_run_id)
set(dp_output "${DP_PROOF_OUTPUT_ROOT}/${dp_run_id}")
file(MAKE_DIRECTORY "${dp_output}/modules" "${dp_output}/logs")
file(WRITE "${dp_output}/lean-toolchain" "leanprover/lean4:v4.34.1\n")
file(WRITE "${dp_output}/source-sha256.txt" "")
foreach(dp_source IN LISTS dp_modules)
  file(COPY "${DP_PROOF_SOURCE_ROOT}/${dp_source}.lean" DESTINATION "${dp_output}/modules")
  file(SHA256 "${DP_PROOF_SOURCE_ROOT}/${dp_source}.lean" dp_hash)
  file(APPEND "${dp_output}/source-sha256.txt" "${dp_hash}  ${dp_source}.lean\n")
endforeach()
foreach(dp_source IN LISTS dp_modules)
  get_filename_component(dp_module "${dp_source}" NAME)
  message(STATUS "Elaborating and replaying ${dp_module}")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "PATH=${dp_lean_bin}:$ENV{PATH}" "LEAN_PATH=${dp_output}/modules" "LEAN_NUM_THREADS=1"
    "${DP_LEAN_EXECUTABLE}" -j 1 -o "${dp_module}.olean" "${dp_module}.lean"
    WORKING_DIRECTORY "${dp_output}/modules" RESULT_VARIABLE dp_result
    OUTPUT_FILE "${dp_output}/logs/${dp_module}-compile.log"
    ERROR_FILE "${dp_output}/logs/${dp_module}-compile-errors.log")
  if(NOT dp_result EQUAL 0)
    message(FATAL_ERROR "Lean elaboration failed for ${dp_module}; logs: ${dp_output}/logs")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "PATH=${dp_lean_bin}:$ENV{PATH}" "LEAN_PATH=${dp_output}/modules" "LEAN_NUM_THREADS=1"
    "${dp_leanchecker}" --verbose "${dp_module}"
    WORKING_DIRECTORY "${dp_output}/modules" RESULT_VARIABLE dp_result
    OUTPUT_FILE "${dp_output}/logs/${dp_module}-kernel.log"
    ERROR_FILE "${dp_output}/logs/${dp_module}-kernel-errors.log")
  if(NOT dp_result EQUAL 0)
    message(FATAL_ERROR "Lean kernel replay failed for ${dp_module}; logs: ${dp_output}/logs")
  endif()
endforeach()
list(LENGTH dp_modules dp_count)
file(SHA256 "${DP_LEAN_EXECUTABLE}" dp_compiler_hash)
file(SHA256 "${dp_leanchecker}" dp_checker_hash)
file(WRITE "${dp_output}/result.json"
  "{\"complete\":true,\"modules\":${dp_count},\"lean_version\":\"${dp_lean_version}\",\"lean_sha256\":\"${dp_compiler_hash}\",\"leanchecker_sha256\":\"${dp_checker_hash}\",\"kernel_replay\":true,\"CUDA_implementation_refinement_claim\":false}\n")
message(STATUS "Replayed ${dp_count} Lean modules; evidence: ${dp_output}")
