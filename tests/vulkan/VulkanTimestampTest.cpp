#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanQueryPool.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

TEST(VulkanTimestampTest, MeasuresMultipleIntervalsAcrossRecordings) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device, &family_count, families.data());
  const uint32_t family = context.GetQueueFamilyIndices().compute_family.value();
  const uint32_t valid_bits = families.at(family).timestampValidBits;
  ASSERT_GT(valid_bits, 0U);
  const uint64_t mask = UINT64_MAX >> (64U - valid_bits);
  vulkan::VulkanCommandBuffer command(&context);
  vulkan::VulkanFence fence(&context);
  vulkan::VulkanBuffer output(
      &context, 4 * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  const std::array<uint32_t, 3> counts = {1, 2, 5};
  for (const uint32_t count : counts) {
    vulkan::VulkanQueryPool queries(&context, VK_QUERY_TYPE_TIMESTAMP, count);
    EXPECT_THROW(queries.GetElapsedMilliseconds(0, 0), std::logic_error);
    for (uint32_t round = 1; round <= 2; ++round) {
      output.MapData([](void* data) { std::memset(data, 0, 4 * sizeof(uint32_t)); });
      fence.Reset();
      command.BeginCommandBufferWithReset();
      queries.Reset(command.buffer());
      EXPECT_THROW(queries.GetElapsedMilliseconds(0, 0), std::logic_error);
      queries.Query(command.buffer(), 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
      EXPECT_THROW(queries.Query(command.buffer(), count), std::invalid_argument);
      for (uint32_t index = 1; index < count; ++index) {
        vkCmdFillBuffer(command.buffer(), output.buffer,
                        static_cast<VkDeviceSize>(index - 1) * sizeof(uint32_t),
                        sizeof(uint32_t), round * 100 + index);
        queries.Query(command.buffer(), index);
      }
      const VkMemoryBarrier host_barrier{
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .pNext = nullptr,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      };
      vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
      command.Submit(fence.fence);
      ASSERT_EQ(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX),
                VK_SUCCESS);
      queries.GetQueryResults();
      const std::vector<uint64_t> timestamps = queries.GetTimeStamps();
      ASSERT_EQ(timestamps.size(), count);
      double sum_ms = 0.0;
      for (uint32_t index = 1; index < count; ++index) {
        const uint64_t ticks = (timestamps[index] - timestamps[index - 1]) & mask;
        const double expected_ms =
            static_cast<double>(ticks) * static_cast<double>(context.timestamp_period) / 1000000.0;
        const double actual_ms = queries.GetElapsedMilliseconds(index - 1, index);
        EXPECT_DOUBLE_EQ(actual_ms, expected_ms);
        EXPECT_TRUE(std::isfinite(actual_ms));
        EXPECT_GE(actual_ms, 0.0);
        sum_ms += actual_ms;
      }
      EXPECT_DOUBLE_EQ(queries.GetElapsedMilliseconds(0, 0), 0.0);
      EXPECT_NEAR(sum_ms, queries.GetElapsedMilliseconds(0, count - 1), 1e-6);
      EXPECT_THROW(queries.GetElapsedMilliseconds(0, count), std::invalid_argument);
      if (count > 1) {
        EXPECT_THROW(queries.GetElapsedMilliseconds(1, 0), std::invalid_argument);
      }
      output.MapData([count, round](void* data) {
        std::array<uint32_t, 4> values{};
        std::memcpy(values.data(), data, sizeof(values));
        for (uint32_t index = 0; index < values.size(); ++index) {
          EXPECT_EQ(values[index], index + 1 < count ? round * 100 + index + 1 : 0U);
        }
      });
    }
  }
}

TEST(VulkanTimestampTest, PreservesDefaultCountAndRejectsInvalidInputs) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  EXPECT_THROW(vulkan::VulkanQueryPool(&context, VK_QUERY_TYPE_TIMESTAMP, 0),
               std::invalid_argument);
  EXPECT_THROW(vulkan::VulkanQueryPool(&context, VK_QUERY_TYPE_OCCLUSION, 4), std::runtime_error);
  vulkan::VulkanQueryPool queries(&context, VK_QUERY_TYPE_TIMESTAMP);
  EXPECT_EQ(queries.GetTimeStamps().size(), 2U);
  EXPECT_THROW(queries.Reset(VK_NULL_HANDLE), std::invalid_argument);
  EXPECT_THROW(queries.Query(VK_NULL_HANDLE, 0), std::invalid_argument);
  EXPECT_THROW(queries.GetElapsedMilliseconds(2, 2), std::invalid_argument);
  EXPECT_THROW(queries.GetElapsedMilliseconds(0, 1), std::logic_error);
}

}
}
}
