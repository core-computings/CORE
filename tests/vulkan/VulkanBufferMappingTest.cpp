#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "VulkanBuffer.h"
#include "VulkanContext.h"

namespace core {
namespace test {

TEST(VulkanBufferMappingTest, RepeatedUpdatesRemainVisibleToGpuAndHost) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  constexpr VkDeviceSize kBytes = 192;
  constexpr VkMemoryPropertyFlags kHostMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  vulkan::VulkanBuffer source(
      &context, kBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      kHostMemory);
  vulkan::VulkanBuffer device(
      &context, kBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vulkan::VulkanBuffer result(&context, kBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostMemory);
  for (uint32_t iteration = 0; iteration < 8; ++iteration) {
    source.MapDataPersistent([iteration](void* data) {
      uint32_t* words = static_cast<uint32_t*>(data);
      for (uint32_t index = 0; index < kBytes / sizeof(uint32_t); ++index) {
        words[index] = iteration * 71 + index;
      }
    });
    const std::array<vulkan::BatchBufferCopy, 2> copies = {
        vulkan::BatchBufferCopy{.source = &source,
                                .destination = &device,
                                .source_offset = 0,
                                .destination_offset = 0,
                                .size = VK_WHOLE_SIZE},
        vulkan::BatchBufferCopy{.source = &device,
                                .destination = &result,
                                .source_offset = 0,
                                .destination_offset = 0,
                                .size = VK_WHOLE_SIZE}};
    vulkan::VulkanBuffer::CopyBuffers(copies);
    result.MapDataPersistent([iteration](void* data) {
      const uint32_t* words = static_cast<const uint32_t*>(data);
      for (uint32_t index = 0; index < kBytes / sizeof(uint32_t); ++index) {
        EXPECT_EQ(words[index], iteration * 71 + index);
      }
    });
  }
}

TEST(VulkanBufferMappingTest, MappingSurvivesCallbackFailureAndOwnershipTransfer) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  constexpr VkMemoryPropertyFlags kHostMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  vulkan::VulkanBuffer destination(
      &context, sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHostMemory);
  destination.MapDataPersistent([](void* data) { *static_cast<uint32_t*>(data) = 99; });
  {
    vulkan::VulkanBuffer source(
        &context, sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHostMemory);
    EXPECT_THROW(
        source.MapDataPersistent([](void* data) {
          *static_cast<uint32_t*>(data) = 42;
          throw std::runtime_error("callback failure");
        }),
        std::runtime_error);
    source.MapDataPersistent([](void* data) { EXPECT_EQ(*static_cast<uint32_t*>(data), 42U); });
    destination = std::move(source);
    EXPECT_THROW(source.MapDataPersistent([](void*) {}), std::runtime_error);
  }
  vulkan::VulkanBuffer* alias = &destination;
  destination = std::move(*alias);
  destination.MapDataPersistent([](void* data) { EXPECT_EQ(*static_cast<uint32_t*>(data), 42U); });
  vulkan::VulkanBuffer result(
      &context, sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostMemory);
  destination.CopyToBuffer(result);
  result.MapDataPersistent([](void* data) { EXPECT_EQ(*static_cast<uint32_t*>(data), 42U); });
  EXPECT_THROW(destination.MapData([](void*) { throw std::runtime_error("transient failure"); }),
               std::runtime_error);
  destination.MapDataPersistent([](void* data) { EXPECT_EQ(*static_cast<uint32_t*>(data), 42U); });
  destination.MapData([](void* data) { *static_cast<uint32_t*>(data) = 71; });
  destination.MapDataPersistent([](void* data) { EXPECT_EQ(*static_cast<uint32_t*>(data), 71U); });
}

TEST(VulkanBufferMappingTest, RejectsEmptyAndNonHostBuffersWithoutCallingCallback) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();
  vulkan::VulkanBuffer empty;
  vulkan::VulkanBuffer device(
      &context, sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  bool called = false;
  const auto callback = [&called](void*) { called = true; };
  EXPECT_THROW(empty.MapData(callback), std::runtime_error);
  EXPECT_THROW(empty.MapDataPersistent(callback), std::runtime_error);
  EXPECT_THROW(device.MapData(callback), std::runtime_error);
  EXPECT_THROW(device.MapDataPersistent(callback), std::runtime_error);
  EXPECT_FALSE(called);
}

}
}
