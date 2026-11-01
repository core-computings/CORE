#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ComputeSum.h"
#include "VulkanCommandBuffer.h"
#include "VulkanSync.h"

namespace core::test {
namespace {

void CheckSum(vulkan::VulkanContext& context, int width, int height) {
  const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<int32_t> values(count);
  int64_t total = 0;
  for (size_t i = 0; i < count; ++i) {
    values[i] = static_cast<int32_t>((i * 13) % 31) - 15;
    total += values[i];
  }
  ASSERT_GE(total, std::numeric_limits<int32_t>::min());
  ASSERT_LE(total, std::numeric_limits<int32_t>::max());
  const VkDeviceSize bytes = count * sizeof(int32_t);
  constexpr VkMemoryPropertyFlags kHost =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  vulkan::VulkanBuffer staging(&context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost);
  vulkan::VulkanBuffer source(
      &context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vulkan::VulkanBuffer destination(
      &context, sizeof(int32_t),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHost);
  staging.MapData([&values, bytes](void* data) { std::memcpy(data, values.data(), bytes); });
  staging.CopyToBuffer(source);
  vulkan::ComputeSum sum(&context, source, destination, width, height);
  sum.Init();
  vulkan::VulkanCommandBuffer command(&context);
  for (int round = 0; round < 2; ++round) {
    const uint32_t seed = round == 0 ? 0U : 123U;
    command.BeginCommandBufferWithReset();
    vkCmdFillBuffer(command.buffer(), destination.buffer, 0, sizeof(int32_t), seed);
    VkMemoryBarrier reset{};
    reset.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    reset.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    reset.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &reset, 0, nullptr, 0, nullptr);
    sum.Run(command.buffer());
    VkMemoryBarrier readback{};
    readback.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    readback.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readback, 0, nullptr, 0, nullptr);
    vulkan::VulkanFence fence(&context);
    fence.Reset();
    command.Submit(fence.fence);
    VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
    destination.MapData([total, seed](void* data) {
      EXPECT_EQ(*static_cast<const int32_t*>(data), total + static_cast<int64_t>(seed));
    });
  }
}

}

TEST(ComputeSumReduction, HandlesPartialAndMultipleTiles) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  const std::array<std::array<int, 2>, 12> shapes = {
      std::array<int, 2>{1, 1}, {15, 17}, {16, 16}, {1, 4095}, {4096, 1},
      {1, 4097}, {37, 29}, {1, 262143}, {262144, 1}, {1, 262145},
      {1024, 1024}, {1003, 769}};
  for (const std::array<int, 2>& shape : shapes) {
    SCOPED_TRACE(::testing::Message() << shape[0] << "x" << shape[1]);
    CheckSum(context, shape[0], shape[1]);
  }
}

TEST(ComputeSumReduction, RejectsInvalidRanges) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  constexpr VkMemoryPropertyFlags kHost =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  vulkan::VulkanBuffer source(&context, 4 * sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             kHost);
  vulkan::VulkanBuffer destination(&context, sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  kHost);
  EXPECT_THROW(vulkan::ComputeSum(&context, source, destination, 0, 1), std::invalid_argument);
  EXPECT_THROW(vulkan::ComputeSum(&context, source, destination, 1, -1), std::invalid_argument);
  EXPECT_THROW(vulkan::ComputeSum(&context, source, destination, 3, 2), std::invalid_argument);
  EXPECT_THROW(vulkan::ComputeSum(&context, source, destination,
                                 std::numeric_limits<int>::max(), 2), std::invalid_argument);
  vulkan::VulkanBuffer too_small(&context, 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHost);
  EXPECT_THROW(vulkan::ComputeSum(&context, source, too_small, 1, 1), std::invalid_argument);
  CheckSum(context, 2, 2);
}

}
