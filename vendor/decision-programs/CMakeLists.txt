cmake_minimum_required(VERSION 3.30)
project(decision-programs VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)
option(DP_ENABLE_CUDA "Build the maintained CUDA numerical backends" ON)
option(DP_BUILD_REGIONAL_XAI "Build the specialized CLSRMDL1 XAI tools" ON)
option(DP_BUILD_DICOM "Build optional DCMTK image/segmentation preparation" OFF)
option(DP_INSTALL_PROOFS "Install Lean source files and their toolchain declarations" ON)
option(DP_CUDA_LINEINFO "Retain CUDA line information for diagnostic collection" OFF)
set(DP_LEAN_EXECUTABLE "" CACHE FILEPATH "Lean 4.34.1 executable for the check-proofs target")
if(NOT UNIX)
  message(FATAL_ERROR "The native consumers require POSIX APIs. On Windows, build inside WSL2.")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
find_package(nlohmann_json 3.11 REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(Threads REQUIRED)

add_library(dp_host_options INTERFACE)
target_include_directories(dp_host_options INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/native")
target_link_libraries(dp_host_options INTERFACE nlohmann_json::nlohmann_json OpenSSL::Crypto Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(dp_host_options INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>")

# The public frontend intentionally has no CUDA linkage: help, capability
# reporting and environment diagnostics remain usable without a GPU toolkit.
add_executable(decision-programs native/decision_programs_cli.cpp)
target_link_libraries(decision-programs PRIVATE dp_host_options)
install(TARGETS decision-programs RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_library(dp_tree_adapter STATIC native/class_tree_adapter.cpp)
target_link_libraries(dp_tree_adapter PUBLIC dp_host_options)
add_executable(class_tree_adapter native/class_tree_adapter_cli.cpp)
target_link_libraries(class_tree_adapter PRIVATE dp_tree_adapter)
set(dp_backend_targets class_tree_adapter)

if(DP_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Detect the local GPU by default; use 86 for the qualified SM86 build.
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
  endif()
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.4)
    message(FATAL_ERROR "The C++23 CUDA backend requires NVIDIA CUDA 13.4 or newer; use DP_ENABLE_CUDA=OFF for host metadata tools.")
  endif()
  # CMake versions predating CUDA 13.4 do not yet advertise its C++23 flag.
  set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
  set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
  set(CMAKE_CUDA_STANDARD 23)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  # NVCC has already identified the toolkit. Optional CUDA library discovery
  # should search its roots and CMake prefixes, not every executable PATH entry
  # (WSL inherits many slow Windows directories). Preserve the caller's setting
  # for all other dependency discovery.
  if(DEFINED CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH)
    set(dp_saved_find_environment_path "${CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH}")
    set(dp_restore_find_environment_path TRUE)
  else()
    set(dp_restore_find_environment_path FALSE)
  endif()
  set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
  find_package(CUDAToolkit 13.4 REQUIRED)
  if(dp_restore_find_environment_path)
    set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH "${dp_saved_find_environment_path}")
  else()
    unset(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH)
  endif()
  find_path(DP_CCCL_INCLUDE_DIR thrust/version.h HINTS ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES cccl REQUIRED)
  add_library(dp_cuda_options INTERFACE)
  target_link_libraries(dp_cuda_options INTERFACE dp_host_options CUDA::cudart CUDA::cuda_driver)
  target_include_directories(dp_cuda_options INTERFACE "${DP_CCCL_INCLUDE_DIR}")
  target_compile_options(dp_cuda_options INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--ftz=false;--fmad=false;--prec-div=true;--prec-sqrt=true;-Xcompiler=-fno-fast-math,-ffp-contract=off>")
  if(DP_CUDA_LINEINFO)
    target_compile_options(dp_cuda_options INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
  endif()

  add_library(dp_class_runtime STATIC native/class_runtime.cu)
  target_link_libraries(dp_class_runtime PUBLIC dp_cuda_options)
  add_library(dp_runtime_fixture STATIC native/class_runtime_fixture.cu)
  target_link_libraries(dp_runtime_fixture PUBLIC dp_class_runtime)
  add_library(dp_class_conversion STATIC native/class_study_convert.cu)
  target_link_libraries(dp_class_conversion PUBLIC dp_class_runtime)
  add_library(dp_class_study STATIC
    native/class_study_data.cu native/class_study_train.cu native/class_study_oof.cu
    native/class_study_composition.cu native/class_study_process.cu native/class_study_holdout.cu
    native/class_study_grouped_segmentation.cu native/class_study_inner_metrics.hpp native/class_study_oof_aggregation.cuh native/class_study_meta_bootstrap.cuh native/class_study_teacher_blend.cuh native/class_study_base_mean.cuh native/class_study_fit_sampling.cuh
    native/class_study_train_loss.hpp native/class_study_group_loss_math.hpp
    native/class_study_group_loss_cuda.hpp native/class_study_group_loss_cuda.cuh)
  target_link_libraries(dp_class_study PUBLIC dp_class_conversion)

  function(dp_cuda_backend name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE dp_class_runtime)
    list(APPEND dp_backend_targets ${name})
    set(dp_backend_targets "${dp_backend_targets}" PARENT_SCOPE)
  endfunction()
  dp_cuda_backend(class_runtime_cli native/class_runtime_cli.cpp)
  target_link_libraries(class_runtime_cli PRIVATE dp_runtime_fixture)
  dp_cuda_backend(class_tree_runtime_compare native/class_tree_runtime_compare.cu)
  target_link_libraries(class_tree_runtime_compare PRIVATE dp_tree_adapter)
  dp_cuda_backend(class_model_evaluate native/class_model_evaluate.cu)
  target_link_libraries(class_model_evaluate PRIVATE dp_runtime_fixture)
  dp_cuda_backend(class_model_simplify native/class_model_simplify.cu)
  dp_cuda_backend(class_model_convert native/class_model_convert.cpp)
  target_link_libraries(class_model_convert PRIVATE dp_class_conversion)
  dp_cuda_backend(class_model_train native/class_model_train.cu)
  dp_cuda_backend(class_study native/class_study_main.cpp)
  target_link_libraries(class_study PRIVATE dp_class_study)
  if(DP_BUILD_DICOM)
    find_package(DCMTK REQUIRED COMPONENTS dcmdata ofstd oflog)
    dp_cuda_backend(class_segmentation_prepare native/class_study_segmentation_prepare.cu native/class_study_dicom.cpp)
    target_include_directories(class_segmentation_prepare PRIVATE ${DCMTK_INCLUDE_DIRS})
    target_link_libraries(class_segmentation_prepare PRIVATE DCMTK::dcmdata)
    dp_cuda_backend(class_segmentation_export native/class_study_segmentation_export.cu native/class_study_dicom.cpp)
    target_include_directories(class_segmentation_export PRIVATE ${DCMTK_INCLUDE_DIRS})
    target_link_libraries(class_segmentation_export PRIVATE DCMTK::dcmdata)
  endif()
  dp_cuda_backend(class_segmentation_score native/class_study_segmentation_score.cu)
  dp_cuda_backend(decision_programs_model_tools native/decision_programs_model_tools.cu)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/native/rl/CMakeLists.txt")
    add_subdirectory(native/rl)
  endif()

  # Bind installed training receipts to compilation inputs without a runtime
  # dependency on the developer's source checkout. No source paths are embedded.
  set(dp_train_manifest_files class_model_train.cu class_model_dataset.cuh
    class_model_dataset_contract.hpp class_native_training.cuh class_model_io.cuh
    class_model_mnist_adapter.cuh class_model_mnist_dataset.cuh
    class_model_native_adapter.cuh class_io.hpp class_native_model_json.hpp class_cuda.hpp CMakeLists.txt)
  list(SORT dp_train_manifest_files)
  set(dp_manifest_files "{")
  set(dp_manifest_separator "")
  foreach(dp_file IN LISTS dp_train_manifest_files)
    if(dp_file STREQUAL "CMakeLists.txt")
      set(dp_source "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
    else()
      set(dp_source "${CMAKE_CURRENT_SOURCE_DIR}/native/${dp_file}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${dp_source}")
    file(SHA256 "${dp_source}" dp_hash)
    string(APPEND dp_manifest_files "${dp_manifest_separator}\"${dp_file}\":\"${dp_hash}\"")
    set(dp_manifest_separator ",")
  endforeach()
  string(APPEND dp_manifest_files "}")
  string(SHA256 dp_manifest_hash "${dp_manifest_files}")
  set(dp_manifest "{\"files\":${dp_manifest_files},\"sha256\":\"${dp_manifest_hash}\",\"role\":\"maintained compilation-source metadata; executable pin is separate\"}")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
  file(WRITE "${CMAKE_BINARY_DIR}/generated/dp_train_source_manifest.hpp"
    "#pragma once\n#define DP_TRAIN_SOURCE_MANIFEST_JSON R\"dpmanifest(${dp_manifest})dpmanifest\"\n")
  target_compile_definitions(class_model_train PRIVATE DP_TRAIN_SOURCE_MANIFEST=1)
  target_include_directories(class_model_train PRIVATE "${CMAKE_BINARY_DIR}/generated")
endif()

if(DP_BUILD_REGIONAL_XAI)
  add_library(dp_regional_xai STATIC native/class_rank_xai_export.cpp native/class_rank_regional_model_export.cpp)
  target_link_libraries(dp_regional_xai PUBLIC dp_host_options)
  # The shared codec file also contains the unrelated CUDA collector producer.
  # These tools only consume its structural codec, so discard unused producer
  # sections rather than introducing a GPU dependency into host equation export.
  target_compile_options(dp_regional_xai PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(dp_regional_xai INTERFACE -Wl,--gc-sections)
  add_executable(class_rank_xai_export native/class_rank_xai_export_cli.cpp)
  target_link_libraries(class_rank_xai_export PRIVATE dp_regional_xai)
  list(APPEND dp_backend_targets class_rank_xai_export)
  if(DP_ENABLE_CUDA)
    add_executable(class_rank_xai_explain native/class_rank_xai_explain_cli.cpp native/class_rank_xai_cuda.cu)
    target_link_libraries(class_rank_xai_explain PRIVATE dp_regional_xai dp_cuda_options)
    list(APPEND dp_backend_targets class_rank_xai_explain)
  endif()
endif()
install(TARGETS ${dp_backend_targets} RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/decision-programs)

if(BUILD_TESTING)
  foreach(dp_check class_study_model_summary_checks class_study_inner_metrics_checks class_study_nested_holdout_checks decision_programs_model_format_checks class_model_dataset_contract_checks class_study_batch_options_checks class_symbolic_test)
    add_executable(${dp_check} native/${dp_check}.cpp)
    target_link_libraries(${dp_check} PRIVATE dp_host_options)
    target_compile_options(${dp_check} PRIVATE -UNDEBUG)
    add_test(NAME ${dp_check} COMMAND ${dp_check})
    set_tests_properties(${dp_check} PROPERTIES LABELS host)
  endforeach()
  add_executable(class_io_checks native/class_io_checks.cpp)
  target_link_libraries(class_io_checks PRIVATE dp_host_options)
  target_compile_options(class_io_checks PRIVATE -UNDEBUG)
  add_test(NAME class_io_checks COMMAND "${CMAKE_COMMAND}"
    "-DDP_TEST_EXECUTABLE=$<TARGET_FILE:class_io_checks>"
    "-DDP_TEST_ROOT=${CMAKE_BINARY_DIR}/test-artifacts/parser"
    "-DDP_TEST_ARGUMENT=${CMAKE_CURRENT_SOURCE_DIR}/cmake/fixtures/xgboost-parser.json"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunFreshFixture.cmake")
  set_tests_properties(class_io_checks PROPERTIES LABELS host)
  add_test(NAME class_tree_adapter_self_test COMMAND "${CMAKE_COMMAND}"
    "-DDP_TEST_EXECUTABLE=$<TARGET_FILE:class_tree_adapter>"
    "-DDP_TEST_ROOT=${CMAKE_BINARY_DIR}/test-artifacts/tree-adapter" "-DDP_TEST_ARGUMENT=self-test"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunFreshFixture.cmake")
  set_tests_properties(class_tree_adapter_self_test PROPERTIES LABELS host)
  if(DP_BUILD_REGIONAL_XAI)
    add_executable(class_rank_xai_export_checks native/class_rank_xai_export_checks.cpp)
    target_link_libraries(class_rank_xai_export_checks PRIVATE dp_regional_xai)
    add_test(NAME class_rank_xai_export_checks COMMAND "${CMAKE_COMMAND}"
      "-DDP_TEST_EXECUTABLE=$<TARGET_FILE:class_rank_xai_export_checks>"
      "-DDP_TEST_ROOT=${CMAKE_BINARY_DIR}/test-artifacts/regional-xai"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunFreshFixture.cmake")
    set_tests_properties(class_rank_xai_export_checks PROPERTIES LABELS host)
  endif()
  add_executable(class_study_checkpoint_checks native/class_study_checkpoint_checks.cpp)
  target_link_libraries(class_study_checkpoint_checks PRIVATE dp_host_options)
  add_test(NAME class_study_checkpoint_checks COMMAND "${CMAKE_COMMAND}"
    "-DDP_TEST_EXECUTABLE=$<TARGET_FILE:class_study_checkpoint_checks>"
    "-DDP_TEST_ROOT=${CMAKE_BINARY_DIR}/test-artifacts/checkpoint"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunFreshFixture.cmake")
  set_tests_properties(class_study_checkpoint_checks PROPERTIES LABELS host)
  if(DP_ENABLE_CUDA)
    add_executable(class_study_composed_conversion_checks native/class_study_composed_conversion_checks.cu)
    target_link_libraries(class_study_composed_conversion_checks PRIVATE dp_class_study)
    add_executable(class_study_grouped_segmentation_integration_checks native/class_study_grouped_segmentation_integration_checks.cpp)
    target_link_libraries(class_study_grouped_segmentation_integration_checks PRIVATE dp_class_study)
    add_executable(class_study_grouped_segmentation_checks native/class_study_grouped_segmentation_checks.cpp)
    target_link_libraries(class_study_grouped_segmentation_checks PRIVATE dp_class_study)
    add_test(NAME class_study_grouped_segmentation_metadata COMMAND class_study_grouped_segmentation_checks)
    set_tests_properties(class_study_grouped_segmentation_metadata PROPERTIES LABELS host)
    if(DP_BUILD_DICOM)
      add_test(NAME class_segmentation_prepare_host COMMAND class_segmentation_prepare --self-test-host)
      set_tests_properties(class_segmentation_prepare_host PROPERTIES LABELS host)
      add_test(NAME class_segmentation_prepare_cuda COMMAND class_segmentation_prepare --self-test)
      set_tests_properties(class_segmentation_prepare_cuda PROPERTIES LABELS gpu RUN_SERIAL TRUE)
      add_test(NAME class_segmentation_export_cuda COMMAND class_segmentation_export --self-test)
      set_tests_properties(class_segmentation_export_cuda PROPERTIES LABELS gpu RUN_SERIAL TRUE)
    endif()
    add_test(NAME class_segmentation_score_cuda COMMAND class_segmentation_score --self-test)
    set_tests_properties(class_segmentation_score_cuda PROPERTIES LABELS gpu RUN_SERIAL TRUE)
    add_executable(class_study_train_checks native/class_study_train_checks.cpp)
    target_link_libraries(class_study_train_checks PRIVATE dp_class_study)
    add_test(NAME class_study_train_checks COMMAND class_study_train_checks)
    set_tests_properties(class_study_train_checks PROPERTIES LABELS host)
    add_executable(class_study_train_loss_checks native/class_study_train_loss_checks.cu)
    target_link_libraries(class_study_train_loss_checks PRIVATE dp_class_study)
    add_test(NAME class_study_train_loss_metadata_checks COMMAND class_study_train_loss_checks)
    set_tests_properties(class_study_train_loss_metadata_checks PROPERTIES LABELS host)
    # CUDA mode takes an explicit caller-owned pinned native-library plan:
    # class_study_train_loss_checks --gpu-check /absolute/library-plan.json

    add_executable(class_study_holdout_checks native/class_study_holdout_checks.cu)
    target_link_libraries(class_study_holdout_checks PRIVATE dp_class_study)
    add_test(NAME class_study_holdout_metadata_checks COMMAND class_study_holdout_checks --metadata-only)
    set_tests_properties(class_study_holdout_metadata_checks PROPERTIES LABELS host)
    add_test(NAME class_study_holdout_cuda_checks COMMAND "${CMAKE_COMMAND}"
      "-DDP_TEST_EXECUTABLE=$<TARGET_FILE:class_study_holdout_checks>"
      "-DDP_TEST_ROOT=${CMAKE_BINARY_DIR}/test-artifacts/holdout"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunFreshFixture.cmake")
    set_tests_properties(class_study_holdout_cuda_checks PROPERTIES LABELS gpu RUN_SERIAL TRUE)
    add_test(NAME class_runtime_cuda_fixture COMMAND class_runtime_cli fixture)
    set_tests_properties(class_runtime_cuda_fixture PROPERTIES LABELS gpu RUN_SERIAL TRUE)
  endif()
  add_test(NAME cli_help COMMAND decision-programs --help)
  add_test(NAME cli_version COMMAND decision-programs --version)
  set_tests_properties(cli_help cli_version PROPERTIES LABELS host)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME cli_transport_checks COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/native/decision_programs_cli_checks.py"
      "$<TARGET_FILE:decision-programs>")
    set_tests_properties(cli_transport_checks PROPERTIES LABELS host)
    if(DP_ENABLE_CUDA)
      add_test(NAME class_study_nested_holdout_integration COMMAND "${CMAKE_COMMAND}"
        "-DDP_TEST_EXECUTABLE=${Python3_EXECUTABLE}"
        "-DDP_TEST_ARGUMENT=${CMAKE_CURRENT_SOURCE_DIR}/native/class_study_nested_holdout_integration_checks.py;$<TARGET_FILE:decision-programs>"
        "-DDP_TEST_ROOT=${CMAKE_BINARY_DIR}/test-artifacts/nested-integration"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunFreshFixture.cmake")
      set_tests_properties(class_study_nested_holdout_integration PROPERTIES
        LABELS gpu RUN_SERIAL TRUE SKIP_REGULAR_EXPRESSION "SKIP: XGBOOST_LIBRARY")
    endif()
  endif()
endif()

if(DP_INSTALL_PROOFS)
  install(FILES cmake/CheckLean.cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/decision-programs/cmake)
  install(DIRECTORY formal/ DESTINATION ${CMAKE_INSTALL_DATADIR}/decision-programs/proofs
    FILES_MATCHING PATTERN "*.lean" PATTERN "lean-toolchain" PATTERN "README*"
    PATTERN ".lake" EXCLUDE PATTERN "TacticProbe*.lean" EXCLUDE)
endif()
add_custom_target(check-proofs COMMAND "${CMAKE_COMMAND}"
  "-DDP_PROOF_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/formal"
  "-DDP_PROOF_OUTPUT_ROOT=${CMAKE_BINARY_DIR}/proof-checks"
  "-DDP_LEAN_EXECUTABLE=${DP_LEAN_EXECUTABLE}"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckLean.cmake"
  COMMENT "Elaborate and independently replay maintained Lean modules" USES_TERMINAL)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/decision-programs)
endif()
set(CPACK_GENERATOR TGZ)
set(CPACK_PACKAGE_CONTACT "Decision Programs maintainers")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Decision Programs: C++23/CUDA class-model conversion, training and scientific interrogation tools")
include(CPack)
