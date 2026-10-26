#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include "VulkanBuffer.h"
#include "VulkanContext.h"

namespace core {
namespace test {

class VulkanBufferRecordingTest : public testing::Test {
 protected:
  void SetUp() override { context_.Init(); }

  std::unique_ptr<vulkan::VulkanBuffer> MakeBuffer(
      VkDeviceSize size = 64,
      VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT) {
    return std::make_unique<vulkan::VulkanBuffer>(
        &context_, size, usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  }

  void SubmitAndWait(const vulkan::VulkanCommandBuffer& command) {
    vulkan::VulkanFence fence(&context_);
    fence.Reset();
    command.Submit(fence.fence);
    ASSERT_EQ(vkWaitForFences(context_.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX),
              VK_SUCCESS);
  }

  vulkan::VulkanContext context_{true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE};
};

class VulkanBufferTransferTest : public VulkanBufferRecordingTest,
                                 public testing::WithParamInterface<bool> {
 protected:
  void CopyBuffers(std::span<const vulkan::BatchBufferCopy> copies) {
    if (!GetParam()) {
      vulkan::VulkanBuffer::CopyBuffers(copies);
      return;
    }
    const vulkan::VulkanCommandBuffer command =
        vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context_);
    vulkan::VulkanBuffer::CopyBuffers(command.buffer(), copies);
    SubmitAndWait(command);
  }
};

TEST_P(VulkanBufferTransferTest, UploadsAndReadsBackAnOrderedChain) {
  std::unique_ptr<vulkan::VulkanBuffer> source = MakeBuffer();
  std::unique_ptr<vulkan::VulkanBuffer> result = MakeBuffer();
  vulkan::VulkanBuffer allocation(
      &context_, 64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vulkan::VulkanBuffer device;
  device = std::move(allocation);
  source->MapData([](void* data) {
    uint32_t* words = static_cast<uint32_t*>(data);
    for (uint32_t index = 0; index < 16; ++index) {
      words[index] = index * 7 + 3;
    }
  });
  std::array<vulkan::BatchBufferCopy, 2> copies{};
  copies[0].source = source.get();
  copies[0].destination = &device;
  copies[1].source = &device;
  copies[1].destination = result.get();

  CopyBuffers(copies);

  result->MapData([](void* data) {
    const uint32_t* words = static_cast<const uint32_t*>(data);
    for (uint32_t index = 0; index < 16; ++index) {
      EXPECT_EQ(words[index], index * 7 + 3);
    }
  });
}

TEST_P(VulkanBufferTransferTest, CopiesSourceRemainderWithOffsetsAndPreservesOtherBytes) {
  std::unique_ptr<vulkan::VulkanBuffer> source = MakeBuffer(16);
  std::unique_ptr<vulkan::VulkanBuffer> destination = MakeBuffer(32);
  source->MapData([](void* data) {
    uint32_t* words = static_cast<uint32_t*>(data);
    for (uint32_t index = 0; index < 4; ++index) {
      words[index] = index + 10;
    }
  });
  destination->MapData([](void* data) {
    uint32_t* words = static_cast<uint32_t*>(data);
    for (uint32_t index = 0; index < 8; ++index) {
      words[index] = 99;
    }
  });
  const vulkan::BatchBufferCopy copy{
      .source = source.get(), .destination = destination.get(),
      .source_offset = 4, .destination_offset = 8, .size = VK_WHOLE_SIZE};
  CopyBuffers(std::span(&copy, 1));
  destination->MapData([](void* data) {
    const uint32_t* words = static_cast<uint32_t*>(data);
    const std::array<uint32_t, 8> expected = {99, 99, 11, 12, 13, 99, 99, 99};
    for (size_t index = 0; index < expected.size(); ++index) {
      EXPECT_EQ(words[index], expected[index]);
    }
  });
}

TEST_P(VulkanBufferTransferTest, AllowsNonOverlappingSelfCopyAndRejectsOverlap) {
  std::unique_ptr<vulkan::VulkanBuffer> buffer = MakeBuffer();
  buffer->MapData([](void* data) { *static_cast<uint32_t*>(data) = 123; });
  vulkan::BatchBufferCopy copy{
      .source = buffer.get(), .destination = buffer.get(),
      .source_offset = 0, .destination_offset = 4, .size = 4};
  CopyBuffers(std::span(&copy, 1));
  buffer->MapData([](void* data) { EXPECT_EQ(static_cast<uint32_t*>(data)[1], 123U); });
  copy.size = 8;
  EXPECT_THROW(CopyBuffers(std::span(&copy, 1)), std::invalid_argument);
}

TEST_P(VulkanBufferTransferTest, RejectsInvalidRangesBeforeExecutingAnyCopy) {
  std::unique_ptr<vulkan::VulkanBuffer> source = MakeBuffer();
  std::unique_ptr<vulkan::VulkanBuffer> destination = MakeBuffer();
  source->MapData([](void* data) { *static_cast<uint32_t*>(data) = 42; });
  destination->MapData([](void* data) { *static_cast<uint32_t*>(data) = 0; });
  std::array<vulkan::BatchBufferCopy, 2> copies{};
  copies[0].source = source.get();
  copies[0].destination = destination.get();
  copies[1] = copies[0];
  for (const VkDeviceSize size : {VkDeviceSize{0}, VkDeviceSize{65},
                                 std::numeric_limits<VkDeviceSize>::max() - 1}) {
    copies[1].size = size;
    EXPECT_THROW(CopyBuffers(copies), std::out_of_range);
  }
  copies[1].size = 4;
  copies[1].source_offset = 64;
  EXPECT_THROW(CopyBuffers(copies), std::out_of_range);
  copies[1].source_offset = 0;
  copies[1].destination_offset = 63;
  EXPECT_THROW(CopyBuffers(copies), std::out_of_range);
  destination->MapData([](void* data) { EXPECT_EQ(*static_cast<uint32_t*>(data), 0U); });
}

TEST_P(VulkanBufferTransferTest, RejectsMissingBuffersUsageAndDifferentContexts) {
  std::unique_ptr<vulkan::VulkanBuffer> source = MakeBuffer();
  std::unique_ptr<vulkan::VulkanBuffer> wrong_usage =
      MakeBuffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  vulkan::BatchBufferCopy copy{};
  EXPECT_THROW(CopyBuffers(std::span(&copy, 1)), std::invalid_argument);
  copy.source = source.get();
  copy.destination = wrong_usage.get();
  EXPECT_THROW(CopyBuffers(std::span(&copy, 1)), std::invalid_argument);
  copy.source = wrong_usage.get();
  copy.destination = source.get();
  EXPECT_THROW(CopyBuffers(std::span(&copy, 1)), std::invalid_argument);
  vulkan::VulkanContext other_context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  other_context.Init();
  vulkan::VulkanBuffer other(&other_context, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  copy.source = source.get();
  copy.destination = &other;
  EXPECT_THROW(CopyBuffers(std::span(&copy, 1)), std::invalid_argument);
}

TEST_P(VulkanBufferTransferTest, EmptyBatchIsANoOpAndLegacyCopyStillWorks) {
  EXPECT_NO_THROW(CopyBuffers({}));
  std::unique_ptr<vulkan::VulkanBuffer> source = MakeBuffer();
  std::unique_ptr<vulkan::VulkanBuffer> destination = MakeBuffer();
  source->MapData([](void* data) { *static_cast<uint32_t*>(data) = 321; });
  source->CopyToBuffer(*destination);
  destination->MapData([](void* data) { EXPECT_EQ(*static_cast<uint32_t*>(data), 321U); });
}

TEST_F(VulkanBufferRecordingTest, DefersExecutionAndReusesCallerCommands) {
  std::unique_ptr<vulkan::VulkanBuffer> source = MakeBuffer();
  std::unique_ptr<vulkan::VulkanBuffer> intermediate = MakeBuffer();
  std::unique_ptr<vulkan::VulkanBuffer> result = MakeBuffer();
  const std::array<vulkan::BatchBufferCopy, 3> copies = {
      vulkan::BatchBufferCopy{.source = source.get(),
                            .destination = intermediate.get(),
                            .source_offset = 0, .destination_offset = 0, .size = 64},
      vulkan::BatchBufferCopy{.source = intermediate.get(),
                            .destination = result.get(),
                            .source_offset = 0, .destination_offset = 0, .size = 64},
      vulkan::BatchBufferCopy{.source = source.get(),
                            .destination = result.get(),
                            .source_offset = 0, .destination_offset = 4, .size = 4}};
  vulkan::VulkanCommandBuffer command(&context_);
  for (uint32_t round = 1; round <= 2; ++round) {
    result->MapData([](void* data) { std::fill_n(static_cast<uint32_t*>(data), 16, 0U); });
    command.BeginCommandBufferWithReset();
    vulkan::VulkanBuffer::CopyBuffers(command.buffer(), copies);
    result->MapData([](void* data) {
      const uint32_t* words = static_cast<const uint32_t*>(data);
      for (size_t index = 0; index < 16; ++index) {
        EXPECT_EQ(words[index], 0U);
      }
    });
    source->MapData([round](void* data) {
      uint32_t* words = static_cast<uint32_t*>(data);
      for (uint32_t index = 0; index < 16; ++index) {
        words[index] = round * 100 + index;
      }
    });
    SubmitAndWait(command);
    result->MapData([round](void* data) {
      const uint32_t* words = static_cast<const uint32_t*>(data);
      for (uint32_t index = 0; index < 16; ++index) {
        EXPECT_EQ(words[index], round * 100 + (index == 1 ? 0 : index));
      }
    });
  }
}

TEST_F(VulkanBufferRecordingTest, RejectedBatchLeavesCallerRecordingUsable) {
  std::unique_ptr<vulkan::VulkanBuffer> source = MakeBuffer();
  std::unique_ptr<vulkan::VulkanBuffer> result = MakeBuffer();
  source->MapData([](void* data) { *static_cast<uint32_t*>(data) = 19; });
  result->MapData([](void* data) { std::fill_n(static_cast<uint32_t*>(data), 16, 0U); });
  std::array<vulkan::BatchBufferCopy, 2> copies{};
  copies[0].source = source.get();
  copies[0].destination = result.get();
  copies[1] = copies[0];
  copies[1].size = 65;
  const vulkan::VulkanCommandBuffer command =
      vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context_);
  EXPECT_THROW(vulkan::VulkanBuffer::CopyBuffers(command.buffer(), copies), std::out_of_range);
  EXPECT_THROW(vulkan::VulkanBuffer::CopyBuffers(VK_NULL_HANDLE, std::span(copies.data(), 1)),
               std::invalid_argument);
  EXPECT_NO_THROW(vulkan::VulkanBuffer::CopyBuffers(VK_NULL_HANDLE, {}));
  copies[0].destination_offset = 4;
  copies[0].size = 4;
  vulkan::VulkanBuffer::CopyBuffers(command.buffer(), std::span(copies.data(), 1));
  SubmitAndWait(command);
  result->MapData([](void* data) {
    const uint32_t* words = static_cast<const uint32_t*>(data);
    EXPECT_EQ(words[0], 0U);
    EXPECT_EQ(words[1], 19U);
  });
}

INSTANTIATE_TEST_SUITE_P(SynchronousAndRecorded, VulkanBufferTransferTest, testing::Bool());

}  // namespace test
}  // namespace core
