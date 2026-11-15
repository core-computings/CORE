#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>

#include "ComputeCopy.h"
#include "ComputeFma.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanQueryPool.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

namespace core {
namespace test {
namespace {

// Records `record` into a fresh command buffer bracketed by GPU timestamps,
// submits it, waits for completion, and returns the GPU time in milliseconds.
double TimeDispatchMs(core::vulkan::VulkanContext& context,
                      core::vulkan::VulkanCommandBuffer& command_buffer,
                      core::vulkan::VulkanFence& fence, core::vulkan::VulkanQueryPool& query_pool,
                      const std::function<void(VkCommandBuffer)>& record) {
  fence.Reset();
  command_buffer.BeginCommandBufferWithReset();

  query_pool.Reset(command_buffer.buffer());
  query_pool.Query(command_buffer.buffer(), 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  record(command_buffer.buffer());
  query_pool.Query(command_buffer.buffer(), 1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  const VkMemoryBarrier host_barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
  };
  vkCmdPipelineBarrier(command_buffer.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);

  VkSubmitInfo submit_info{};
  command_buffer.Submit(fence.fence, submit_info);
  vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX);

  query_pool.GetQueryResults();
  const auto timestamps = query_pool.GetTimeStamps();
  return (timestamps[1] - timestamps[0]) * (context.timestamp_period / 1000000.0);
}

}  // namespace

// Measures the two roofline-model ceilings with microbenchmarks and derives
// the ridge point (peak compute throughput / peak memory bandwidth). Kernels
// with arithmetic intensity below the ridge point are memory-bound, above it
// compute-bound.
TEST(Roofline, RidgePoint) {
  core::vulkan::QueueFamilyType queue_family_type = core::vulkan::QueueFamilyType::Compute;
  core::vulkan::VulkanContext context(true, queue_family_type, nullptr);
  context.Init();
  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanFence fence(&context);
  core::vulkan::VulkanQueryPool query_pool(&context, VK_QUERY_TYPE_TIMESTAMP);

  // Early runs also serve as warm-up while DVFS ramps the GPU clock up; the
  // minimum over all runs is the peak.
  constexpr int kRuns = 6;

  // ---- Peak compute throughput: register-resident FMA chains ----
  constexpr uint32_t kFmaGroups = 2048;
  constexpr uint32_t kFmaThreads = kFmaGroups * core::vulkan::ComputeFma::kLocalSizeX;
  constexpr int kFmaIterations = 1024;

  core::vulkan::VulkanBuffer fma_dst(
      &context, kFmaThreads * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  auto fma =
      std::make_unique<core::vulkan::ComputeFma>(&context, fma_dst, kFmaIterations, kFmaGroups);
  fma->Init();

  double fma_ms = std::numeric_limits<double>::max();
  for (int run = 0; run < kRuns; ++run) {
    fma_ms = std::min(fma_ms, TimeDispatchMs(context, command_buffer, fence, query_pool,
                                             [&](VkCommandBuffer cmd) { fma->Run(cmd); }));
  }

  const double total_flop = static_cast<double>(kFmaThreads) * kFmaIterations *
                            core::vulkan::ComputeFma::kFlopPerIteration;
  const double peak_gflops = total_flop / (fma_ms * 1e6);

  // The chains converge toward 0.1; a finite result proves the loop ran.
  float fma_sample = 0.0f;
  fma_dst.MapData([&fma_sample](void* data) { memcpy(&fma_sample, data, sizeof(float)); });
  EXPECT_TRUE(std::isfinite(fma_sample));

  double fma16_ms = 0.0;
  double peak_fp16_gflops = 0.0;
  if (context.float16_support().arithmetic) {
    auto fma16 = std::make_unique<core::vulkan::ComputeFma>(
        &context, fma_dst, kFmaIterations, kFmaGroups, core::vulkan::FmaPrecision::kFloat16);
    fma16->Init();
    fma16_ms = std::numeric_limits<double>::max();
    for (int run = 0; run < kRuns; ++run) {
      fma16_ms = std::min(fma16_ms, TimeDispatchMs(context, command_buffer, fence, query_pool,
                                                   [&](VkCommandBuffer cmd) { fma16->Run(cmd); }));
    }
    peak_fp16_gflops = static_cast<double>(kFmaThreads) * kFmaIterations *
                       static_cast<double>(fma16->FlopPerIteration()) / (fma16_ms * 1e6);
    float fma16_sample = 0.0f;
    fma_dst.MapData([&fma16_sample](void* data) { memcpy(&fma16_sample, data, sizeof(float)); });
    EXPECT_NEAR(fma16_sample, 4.0f, 0.01f);
    EXPECT_GT(peak_fp16_gflops, 0.0);
  }

  // ---- Peak memory bandwidth: streaming vec4 copy ----
  // 8M vec4 = 128 MiB per buffer, far larger than the GPU's caches.
  constexpr int kCopyCount = 8 * 1024 * 1024;
  constexpr VkDeviceSize kCopyBytes = static_cast<VkDeviceSize>(kCopyCount) * 16;

  core::vulkan::VulkanBuffer copy_src(
      &context, kCopyBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  core::vulkan::VulkanBuffer copy_dst(
      &context, kCopyBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  core::vulkan::VulkanBuffer staging(
      &context, kCopyBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  staging.MapData([](void* data) {
    float* floats = static_cast<float*>(data);
    for (int i = 0; i < kCopyCount * 4; ++i) {
      floats[i] = static_cast<float>(i % 1024);
    }
  });
  staging.CopyToBuffer(copy_src);

  auto copy = std::make_unique<core::vulkan::ComputeCopy>(&context, copy_src, copy_dst, kCopyCount);
  copy->Init();

  double copy_ms = std::numeric_limits<double>::max();
  for (int run = 0; run < kRuns; ++run) {
    copy_ms = std::min(copy_ms, TimeDispatchMs(context, command_buffer, fence, query_pool,
                                               [&](VkCommandBuffer cmd) { copy->Run(cmd); }));
  }

  const double bytes_moved = 2.0 * static_cast<double>(kCopyBytes);  // read + write
  const double peak_gbps = bytes_moved / (copy_ms * 1e6);

  // Verify the copy actually happened before trusting the bandwidth number.
  copy_dst.CopyToBuffer(staging);
  float first = -1.0f, middle = -1.0f, last = -1.0f;
  staging.MapData([&first, &middle, &last](void* data) {
    const float* floats = static_cast<const float*>(data);
    constexpr int kFloatCount = kCopyCount * 4;
    first = floats[0];
    middle = floats[kFloatCount / 2];
    last = floats[kFloatCount - 1];
  });
  EXPECT_FLOAT_EQ(first, 0.0f);
  EXPECT_FLOAT_EQ(middle, static_cast<float>((kCopyCount * 4 / 2) % 1024));
  EXPECT_FLOAT_EQ(last, static_cast<float>((kCopyCount * 4 - 1) % 1024));

  // ---- Ridge point ----
  const double ridge_point = peak_gflops / peak_gbps;  // FLOP per byte

  printf("Peak compute throughput: %.1f GFLOP/s (fp32, best of %d runs, %.3f ms)\n", peak_gflops,
         kRuns, fma_ms);
  if (peak_fp16_gflops > 0.0) {
    printf("Peak fp16 throughput   : %.1f GFLOP/s (best of %d runs, %.3f ms)\n", peak_fp16_gflops,
           kRuns, fma16_ms);
  }
  printf("Peak memory bandwidth  : %.1f GB/s (best of %d runs, %.3f ms)\n", peak_gbps, kRuns,
         copy_ms);
  printf("Ridge point            : %.1f FLOP/byte\n", ridge_point);

  EXPECT_GT(peak_gflops, 0.0);
  EXPECT_GT(peak_gbps, 0.0);
}

}  // namespace test
}  // namespace core
