#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "VulkanCommandBuffer.h"
#include "VulkanSync.h"
#include "benchmarks/RasterizerBaseline.h"

namespace core::test {
namespace {

TEST(VulkanRasterizerBoundsTest,
     PreservesClippingEdgesDegeneratesAndAccumulation) {
  vulkan::VulkanContext context(true);
  context.Init();
  vulkan::VulkanCommandBuffer command(&context);
  vulkan::VulkanFence fence(&context);
  const Rasterizer::WorkgroupSize groups[] = {{.x = 1, .y = 1},
                                              {.x = 8, .y = 4},
                                              {.x = 16, .y = 16},
                                              {.x = 32, .y = 1}};
  const std::vector<Triangle> triangles = {
      {.a = glm::ivec2(3, 5), .b = glm::ivec2(11, 5), .c = glm::ivec2(3, 12)},
      {.a = glm::ivec2(30, 23),
       .b = glm::ivec2(36, 23),
       .c = glm::ivec2(36, 28)},
      {.a = glm::ivec2(-8, -5),
       .b = glm::ivec2(15, -5),
       .c = glm::ivec2(-8, 17)},
      {.a = glm::ivec2(31, 22),
       .b = glm::ivec2(50, 22),
       .c = glm::ivec2(31, 40)},
      {.a = glm::ivec2(0, 0), .b = glm::ivec2(36, 0), .c = glm::ivec2(0, 28)},
      {.a = glm::ivec2(5, 9), .b = glm::ivec2(5, 9), .c = glm::ivec2(5, 9)},
      {.a = glm::ivec2(2, 7), .b = glm::ivec2(8, 7), .c = glm::ivec2(14, 7)},
      {.a = glm::ivec2(11, 2), .b = glm::ivec2(11, 8), .c = glm::ivec2(11, 14)},
      {.a = glm::ivec2(2, 2), .b = glm::ivec2(2, 17), .c = glm::ivec2(19, 2)},
      {.a = glm::ivec2(-16383, -16383),
       .b = glm::ivec2(16383, -16383),
       .c = glm::ivec2(-16383, 16383)},
      {.a = glm::ivec2(-9, 3), .b = glm::ivec2(-1, 3), .c = glm::ivec2(-9, 17)},
      {.a = glm::ivec2(37, 3), .b = glm::ivec2(42, 3), .c = glm::ivec2(37, 17)},
      {.a = glm::ivec2(3, -9), .b = glm::ivec2(17, -9), .c = glm::ivec2(3, -1)},
      {.a = glm::ivec2(3, 29),
       .b = glm::ivec2(17, 29),
       .c = glm::ivec2(3, 35)}};
  for (const Rasterizer::WorkgroupSize& group : groups) {
    Rasterizer rasterizer(&context, 37, 29, group);
    rasterizer.Init();
    for (int round = 0; round < 2; ++round) {
      std::vector<int> expected(37 * 29, -7);
      rasterizer.rasterized.MapData([](void* bytes) {
        std::fill_n(static_cast<int*>(bytes), 37 * 29, -7);
      });
      for (const Triangle& triangle : triangles) {
        if (round == 0) {
          std::fill(expected.begin(), expected.end(), -7);
          rasterizer.rasterized.MapData([](void* bytes) {
            std::fill_n(static_cast<int*>(bytes), 37 * 29, -7);
          });
        }
        command.BeginCommandBufferWithReset();
        rasterizer.Run(command.buffer(), triangles.back());
        rasterizer.Run(command.buffer(), triangle);
        rasterizer.Run(command.buffer(), triangles.back());
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(
            command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        fence.Reset();
        command.Submit(fence.fence, vulkan::SubmitSyncInfo{});
        VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence,
                                 VK_TRUE, UINT64_MAX));
        for (int y = 0; y < 29; ++y) {
          for (int x = 0; x < 37; ++x) {
            if (RasterizerCovers(triangle, x, y)) {
              expected[static_cast<size_t>(y * 37 + x)] = 1;
            }
          }
        }
        rasterizer.rasterized.MapData([&expected](void* bytes) {
          const int* actual = static_cast<const int*>(bytes);
          for (size_t index = 0; index < expected.size(); ++index) {
            ASSERT_EQ(actual[index], expected[index]) << index;
          }
        });
      }
    }
  }
}

}
}
