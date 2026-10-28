include(${CMAKE_SOURCE_DIR}/cmake/vulkan_shaders.cmake)

add_vulkan_shader_target(vulkan_shaders
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(GLOB test_src
  "ComputeSum/*.cpp"
  "ComputeGaussianBlur/*.cpp"
  "Roofline/*.cpp"
  "*.cpp")

add_executable(vulkan_tests ${test_src})

add_vulkan_shader_target(vulkan_rasterizer_test_shader
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/examples/vulkan/ComputeBarycentricRasterizer
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

target_sources(vulkan_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/examples/vulkan/ComputeBarycentricRasterizer/ComputeBarycentric.cpp)

target_include_directories(vulkan_tests
  PRIVATE
  ComputeSum
  ComputeGaussianBlur
  Roofline
  ${PROJECT_SOURCE_DIR}/examples/vulkan/ComputeBarycentricRasterizer
  ${CMAKE_CURRENT_BINARY_DIR}/shaders)

add_dependencies(vulkan_tests vulkan_shaders vulkan_rasterizer_test_shader)

target_link_libraries(vulkan_tests
  PRIVATE
  gtest_main
  mat
  stb
  timer
  vulkan)

add_executable(vulkan_transfer_benchmark benchmarks/BufferTransferBenchmark.cpp)
target_link_libraries(vulkan_transfer_benchmark PRIVATE vulkan)

add_executable(vulkan_gaussian_benchmark
  benchmarks/GaussianBlurBenchmark.cpp
  ComputeGaussianBlur/ComputeGaussianBlur.cpp)
target_include_directories(vulkan_gaussian_benchmark PRIVATE
  ComputeGaussianBlur
  ${CMAKE_CURRENT_BINARY_DIR}/shaders)
add_dependencies(vulkan_gaussian_benchmark vulkan_shaders)
target_link_libraries(vulkan_gaussian_benchmark PRIVATE mat vulkan)

add_executable(vulkan_mapping_benchmark benchmarks/BufferMappingBenchmark.cpp)
target_link_libraries(vulkan_mapping_benchmark PRIVATE vulkan)
