#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ComputeGaussianBlur.h"
#include "GaussianBlurReference.h"
#include "VulkanCommandBuffer.h"
#include "VulkanSync.h"
#include "benchmarks/GaussianBlurBaseline.h"

namespace core::test {
namespace {

class GaussianBlurShaderTest : public testing::Test {
 protected:
  void SetUp() override { context_.Init(); }

  template <typename Pipeline>
  void Dispatch(Pipeline& pipeline, vulkan::VulkanCommandBuffer& command,
                 vulkan::VulkanFence& fence) {
    fence.Reset();
    command.BeginCommandBufferWithReset();
    pipeline.Run(command.buffer());
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    command.Submit(fence.fence);
    VK_CHECK(vkWaitForFences(context_.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
  }

  void CheckShape(int width, int height) {
    constexpr size_t kGuardSize = 37;
    constexpr float kSentinel = -8192.0f;
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count + kGuardSize) * sizeof(float);
    constexpr VkMemoryPropertyFlags kHostMemory =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vulkan::VulkanBuffer source(&context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
    vulkan::VulkanBuffer destination(&context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      kHostMemory);
    vulkan::ComputeGaussianBlur candidate(&context_, source, destination, width, height);
    BaselineBlur baseline(&context_, source, destination, width, height);
    candidate.Init();
    baseline.Init();
    vulkan::VulkanCommandBuffer command(&context_);
    vulkan::VulkanFence fence(&context_);
    for (int round = 0; round < 3; ++round) {
      SCOPED_TRACE(testing::Message() << "round " << round);
      std::vector<float> input = GaussianInput(width, height);
      if (round == 1) {
        std::fill(input.begin(), input.end(), 0.0f);
        input[count / 2] = 16.0f;
      } else if (round == 2) {
        for (size_t index = 0; index < count; ++index) {
          input[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 13.0f;
        }
      }
      std::vector<float> expected = GaussianReference(input, width, height);
      input.resize(count + kGuardSize, kSentinel);
      expected.resize(count + kGuardSize, kSentinel);
      source.MapData([&input, bytes](void* data) { std::memcpy(data, input.data(), bytes); });
      destination.MapData([bytes, kSentinel](void* data) {
        std::fill_n(static_cast<float*>(data), bytes / sizeof(float), kSentinel);
      });
      Dispatch(baseline, command, fence);
      std::vector<float> original(count + kGuardSize);
      destination.MapData([&original, bytes](void* data) {
        std::memcpy(original.data(), data, bytes);
      });
      if (round == 2) {
        for (size_t index = 0; index < count; ++index) {
          ASSERT_NEAR(original[index], expected[index], 0.000002f) << index;
        }
      } else {
        ASSERT_EQ(original, expected);
      }
      destination.MapData([bytes, kSentinel](void* data) {
        std::fill_n(static_cast<float*>(data), bytes / sizeof(float), kSentinel);
      });
      Dispatch(candidate, command, fence);
      destination.MapData([&original](void* data) {
        EXPECT_EQ(std::memcmp(original.data(), data, original.size() * sizeof(float)), 0);
      });
      source.MapData([&input](void* data) {
        EXPECT_TRUE(std::equal(input.begin(), input.end(), static_cast<float*>(data)));
      });
    }
  }

  vulkan::VulkanContext context_{true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE};
};

TEST_F(GaussianBlurShaderTest, BordersAndPartialWorkgroups) {
  for (const std::array<int, 2>& shape :
       {std::array<int, 2>{1, 1}, {1, 33}, {33, 1}, {2, 17}, {17, 2}, {3, 3},
        {15, 17}, {16, 16}, {17, 15}, {17, 17}, {31, 33}, {35, 19}}) {
    SCOPED_TRACE(testing::Message() << shape[0] << "x" << shape[1]);
    CheckShape(shape[0], shape[1]);
  }
}

}
}
