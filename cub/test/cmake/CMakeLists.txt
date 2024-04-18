# Check source code for issues that can be found by pattern matching:
add_test(
  NAME cub.test.cmake.check_source_files
  COMMAND
    "${CMAKE_COMMAND}"
      -D "CUB_SOURCE_DIR=${CUB_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_LIST_DIR}/check_source_files.cmake"
)

foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)
  cub_get_target_property(config_dialect ${cub_target} DIALECT)
  set(config_meta_target ${config_prefix}.tests)

  add_executable(
endforeach()
