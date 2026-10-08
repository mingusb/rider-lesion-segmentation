# Specialized, source-qualified categorical RL pipeline. No archived objects.
set(dp_rl_units
  session.cpp
  class_identity.cu
  resident_session.cu
  rl_category_lowering.cu
  rl_category_policy.cu
  class_rank_gpu_binary_grid_validator.cu
  class_rank_gpu_score_diagram.cpp
  class_rank_gpu_score_diagram.cu
  class_rank_gpu_leaf_partitions.cu
  class_rank_gpu_score_factors.cu
  class_rank_gpu_score_diagram_factor_audit.cu
  class_rank_gpu_online_cache_initialized.cu
  class_rank_gpu_online_cache.cpp
  class_rank_gpu_dag_collect_initialized.cu
  class_rank_regional_model_export.cpp
  class_rank_gpu_score_source_apply.cpp
  class_rank_gpu_score_source_apply.cu
  class_rank_gpu_score_add_apply.cpp
  class_rank_gpu_score_add_apply.cu
  class_rank_gpu_score_add_apply_audit.cu
)
set(dp_rl_manifest_files
  checks_resident.cpp
  checks_streaming_file_hash.cpp
  class_apply_rl_cli.cpp
  class_cuda.hpp
  class_identity.cu
  class_identity.hpp
  class_io.hpp
  class_rank_disk_ledger.hpp
  class_rank_gpu_binary_grid_validator.cu
  class_rank_gpu_binary_grid_validator.hpp
  class_rank_gpu_binary_grid_validator_receipt.hpp
  class_rank_gpu_bounds.hpp
  class_rank_gpu_class_apply.cu
  class_rank_gpu_class_apply.hpp
  class_rank_gpu_class_apply_receipt.hpp
  class_rank_gpu_class_apply_source_receipt.hpp
  class_rank_gpu_dag_collect.hpp
  class_rank_gpu_dag_collect_initialized.cu
  class_rank_gpu_dag_inference.hpp
  class_rank_gpu_guard_lookup.hpp
  class_rank_gpu_leaf_partitions.cu
  class_rank_gpu_leaf_partitions.hpp
  class_rank_gpu_leaf_partitions_receipt.hpp
  class_rank_gpu_online_cache.cpp
  class_rank_gpu_online_cache.hpp
  class_rank_gpu_online_cache_initialized.cu
  class_rank_gpu_score_add_apply.cpp
  class_rank_gpu_score_add_apply.cu
  class_rank_gpu_score_add_apply.hpp
  class_rank_gpu_score_add_apply_audit.cu
  class_rank_gpu_score_diagram.cpp
  class_rank_gpu_score_diagram.cu
  class_rank_gpu_score_diagram.hpp
  class_rank_gpu_score_diagram_factor_audit.cu
  class_rank_gpu_score_diagram_factor_audit.hpp
  class_rank_gpu_score_factors.cu
  class_rank_gpu_score_factors.hpp
  class_rank_gpu_score_pool_identity.hpp
  class_rank_gpu_score_source_apply.cpp
  class_rank_gpu_score_source_apply.cu
  class_rank_gpu_score_source_apply.hpp
  class_rank_gpu_task_resolution.hpp
  class_rank_grid_stream_parallel_cuda.hpp
  class_rank_grid_whole_cuda.hpp
  class_rank_regional_model_export.cpp
  class_rank_regional_model_export.hpp
  class_xgboost.hpp
  cooperative_policy1_device.cuh
  policy_state.hpp
  portable_build_identity.hpp
  resident_effective_key.cuh
  resident_policy1_device.cuh
  resident_session.cu
  resident_session.hpp
  rl_category_lowering.cu
  rl_category_lowering.hpp
  rl_category_policy.cu
  rl_category_policy.hpp
  session.cpp
  session.hpp
  session_resident_loop.cpp
  streaming_file_hash.hpp
)
set(dp_rl_manifest "#pragma once\n#include <array>\nnamespace public_rl_source_manifest {\nstruct Entry { const char* name; const char* sha256; };\n")
list(LENGTH dp_rl_manifest_files dp_rl_manifest_count)
string(APPEND dp_rl_manifest "inline constexpr std::array<Entry, ${dp_rl_manifest_count}> files{{\n")
foreach(dp_rl_file IN LISTS dp_rl_manifest_files)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${dp_rl_file}")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${dp_rl_file}" dp_rl_hash)
  string(APPEND dp_rl_manifest "  {\"${dp_rl_file}\", \"${dp_rl_hash}\"},\n")
endforeach()
string(APPEND dp_rl_manifest "}};\n}\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/public_rl_source_manifest.hpp" "${dp_rl_manifest}")
add_library(dp_rl_core STATIC ${dp_rl_units})
target_link_libraries(dp_rl_core PUBLIC dp_cuda_options)
target_include_directories(dp_rl_core BEFORE PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(dp_rl_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
add_executable(rl_session session_resident_loop.cpp)
target_link_libraries(rl_session PRIVATE dp_rl_core)
add_executable(class_apply_rl class_apply_rl_cli.cpp)
target_link_libraries(class_apply_rl PRIVATE dp_rl_core)
add_executable(rl_resident_checks checks_resident.cpp)
target_link_libraries(rl_resident_checks PRIVATE dp_rl_core)
add_executable(rl_hash_checks checks_streaming_file_hash.cpp)
target_link_libraries(rl_hash_checks PRIVATE dp_rl_core)
install(TARGETS rl_session class_apply_rl rl_resident_checks rl_hash_checks RUNTIME DESTINATION libexec/decision-programs)

if(BUILD_TESTING)
  add_test(NAME rl_existing_output_preserved COMMAND "${CMAKE_COMMAND}"
    "-DDP_RL_TEST_EXECUTABLE=$<TARGET_FILE:rl_session>"
    "-DDP_RL_TEST_ROOT=${CMAKE_BINARY_DIR}/test-artifacts/rl-existing-output"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckRefusedOutput.cmake")
  set_tests_properties(rl_existing_output_preserved PROPERTIES LABELS host)
  add_test(NAME rl_resident_metadata COMMAND rl_resident_checks --cpu)
  set_tests_properties(rl_resident_metadata PROPERTIES LABELS host)
  add_test(NAME rl_resident_cuda COMMAND rl_resident_checks --gpu)
  set_tests_properties(rl_resident_cuda PROPERTIES LABELS gpu RUN_SERIAL TRUE)
endif()
