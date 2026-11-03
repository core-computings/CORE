#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "RasterizerBaseline.h"
#include "VulkanCommandBuffer.h"
#include "VulkanQueryPool.h"
#include "VulkanSync.h"

namespace {

using core::test::Rasterizer;
using core::test::RasterizerBaseline;
using core::test::Triangle;
using core::vulkan::QueueFamilyType;
using core::vulkan::SubmitSyncInfo;
using core::vulkan::VulkanCommandBuffer;
using core::vulkan::VulkanContext;
using core::vulkan::VulkanFence;
using core::vulkan::VulkanQueryPool;

void Benchmark(VulkanContext& context, int width, int height, const char* name,
               const Triangle& triangle) {
  RasterizerBaseline baseline(&context, width, height);
  Rasterizer candidate(&context, width, height);
  baseline.Init();
  candidate.Init();
  VulkanCommandBuffer command(&context);
  VulkanFence fence(&context);
  VulkanQueryPool queries(&context, VK_QUERY_TYPE_TIMESTAMP);
  std::vector<int> expected(static_cast<size_t>(width) *
                            static_cast<size_t>(height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      expected[static_cast<size_t>(y * width + x)] =
          core::test::RasterizerCovers(triangle, x, y) ? 1 : 0;
    }
  }
  std::vector<double> samples[2];
  constexpr int kWarmups = 5;
  constexpr int kSamples = 31;
  for (int sample = -kWarmups; sample < kSamples; ++sample) {
    for (int turn = 0; turn < 2; ++turn) {
      const int mode = (sample + kWarmups + turn) % 2;
      Rasterizer& rasterizer = mode == 0 ? baseline : candidate;
      rasterizer.rasterized.MapData([&rasterizer](void* bytes) {
        std::memset(bytes, 0,
                    static_cast<size_t>(rasterizer.rasterized.Size()));
      });
      command.BeginCommandBufferWithReset();
      queries.Reset(command.buffer());
      queries.Query(command.buffer(), 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
      if (mode == 0) {
        baseline.Run(command.buffer(), triangle);
      } else {
        candidate.Run(command.buffer(), triangle);
      }
      queries.Query(command.buffer(), 1, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
      VkMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(
          command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
      fence.Reset();
      command.Submit(fence.fence, SubmitSyncInfo{});
      VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE,
                               UINT64_MAX));
      queries.GetQueryResults();
      const double microseconds = queries.GetElapsedMilliseconds(0, 1) * 1000.0;
      if (sample >= 0) {
        samples[mode].push_back(microseconds);
        std::printf("sample %dx%d %s mode=%d index=%d us=%.3f\n", width, height,
                    name, mode, sample, microseconds);
      }
      rasterizer.rasterized.MapData([&expected](void* bytes) {
        if (std::memcmp(bytes, expected.data(),
                        expected.size() * sizeof(int)) != 0) {
          throw std::runtime_error("Rasterizer benchmark output mismatch");
        }
      });
    }
  }
  for (std::vector<double>& values : samples) {
    std::sort(values.begin(), values.end());
  }
  std::printf(
      "RESULT %dx%d %s warmups=%d samples=%d "
      "baseline_us=%.3f/%.3f/%.3f candidate_us=%.3f/%.3f/%.3f verified=yes\n",
      width, height, name, kWarmups, kSamples, samples[0][3], samples[0][15],
      samples[0][27], samples[1][3], samples[1][15], samples[1][27]);
}

}

int main() {
  VulkanContext context(true, QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &properties);
  std::printf(
      "GPU=%s; GPU timestamps include dispatch/barrier, exclude host work; "
      "p10/median/p90; alternating paired paths\n",
      properties.deviceName);
  for (int size : {1024, 2048}) {
    Benchmark(context, size, size, "small",
              {.a = glm::ivec2(101, 107),
               .b = glm::ivec2(132, 107),
               .c = glm::ivec2(101, 138)});
    Benchmark(context, size, size, "demo",
              {.a = glm::ivec2(300, 800),
               .b = glm::ivec2(500, 100),
               .c = glm::ivec2(700, 800)});
    Benchmark(context, size, size, "full",
              {.a = glm::ivec2(0, 0),
               .b = glm::ivec2(size - 1, 0),
               .c = glm::ivec2(0, size - 1)});
  }
}
