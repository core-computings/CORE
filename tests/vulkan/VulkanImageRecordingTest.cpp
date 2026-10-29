#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanImage.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkMemoryPropertyFlags kHostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

class VulkanImageRecordingTest : public testing::Test {
 protected:
  void SetUp() override {
    context_.Init();
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context_.physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(context_.physical_device, &count, families.data());
    const uint32_t family = context_.GetQueueFamilyIndices().compute_family.value();
    ASSERT_NE(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT, 0U);
  }

  void RecordUpload(VkCommandBuffer command, vulkan::VulkanBuffer& source,
                    vulkan::VulkanImage& image, uint32_t levels, uint32_t layers,
                    bool mipmaps) {
    image.TransitionImageLayout(command, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kFormat, levels, layers);
    source.CopyToImage(command, image, image.image_width, image.image_height, layers);
    if (mipmaps) {
      image.GenerateMipmaps(command);
    } else {
      image.TransitionImageLayout(command, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kFormat, levels,
                                  layers);
    }
  }

  std::vector<uint32_t> ReadImage(vulkan::VulkanImage& image, uint32_t levels,
                                 uint32_t layers) {
    std::vector<VkBufferImageCopy> regions;
    uint32_t width = image.image_width;
    uint32_t height = image.image_height;
    VkDeviceSize bytes = 0;
    for (uint32_t level = 0; level < levels; ++level) {
      VkBufferImageCopy region{};
      region.bufferOffset = bytes;
      region.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .mipLevel = level,
                                 .baseArrayLayer = 0,
                                 .layerCount = layers};
      region.imageExtent = {.width = width, .height = height, .depth = 1};
      regions.push_back(region);
      bytes += static_cast<VkDeviceSize>(width) * height * layers * sizeof(uint32_t);
      width = std::max(width / 2, 1U);
      height = std::max(height / 2, 1U);
    }
    vulkan::VulkanBuffer destination(&context_, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     kHostMemory);
    const vulkan::VulkanCommandBuffer command =
        vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context_);
    VkImageMemoryBarrier image_barrier{};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image.image;
    image_barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .baseMipLevel = 0,
                                      .levelCount = levels,
                                      .baseArrayLayer = 0,
                                      .layerCount = layers};
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &image_barrier);
    vkCmdCopyImageToBuffer(command.buffer(), image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           destination.buffer, static_cast<uint32_t>(regions.size()),
                           regions.data());
    const VkMemoryBarrier host_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
    SubmitAndWait(command);
    std::vector<uint32_t> result(static_cast<size_t>(bytes / sizeof(uint32_t)));
    destination.MapData([&](void* data) { std::memcpy(result.data(), data, bytes); });
    return result;
  }

  void SubmitAndWait(const vulkan::VulkanCommandBuffer& command) {
    vulkan::VulkanFence fence(&context_);
    fence.Reset();
    command.Submit(fence.fence);
    ASSERT_EQ(vkWaitForFences(context_.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX),
              VK_SUCCESS);
  }

  std::vector<uint32_t> Upload(uint32_t width, uint32_t height, uint32_t layers,
                              uint32_t levels, bool recorded, bool constant) {
    const size_t pixels = static_cast<size_t>(width) * height * layers;
    std::vector<uint32_t> input(pixels);
    for (size_t index = 0; index < pixels; ++index) {
      input[index] = constant ? 0xff804020U
                              : (0xff000000U |
                                 (static_cast<uint32_t>(index * 7919U) & 0x00ffffffU));
    }
    vulkan::VulkanBuffer source(&context_, pixels * sizeof(uint32_t),
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHostMemory);
    source.MapData([&](void* data) { std::memcpy(data, input.data(), source.Size()); });
    vulkan::VulkanImage image(
        &context_, width, height, kFormat,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_TILING_OPTIMAL, levels, VK_SAMPLE_COUNT_1_BIT,
        layers == 6 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, layers);
    const bool mipmaps = levels > 1 || constant;
    if (recorded) {
      const vulkan::VulkanCommandBuffer command =
          vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context_);
      RecordUpload(command.buffer(), source, image, levels, layers, mipmaps);
      SubmitAndWait(command);
    } else {
      image.TransitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kFormat, levels, layers);
      source.CopyToImage(image, width, height, layers);
      if (mipmaps) {
        image.GenerateMipmaps();
      } else {
        image.TransitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kFormat, levels,
                                    layers);
      }
    }
    std::vector<uint32_t> result = ReadImage(image, levels, layers);
    EXPECT_TRUE(std::equal(input.begin(), input.end(), result.begin()));
    if (constant) {
      EXPECT_TRUE(std::all_of(result.begin(), result.end(),
                              [](uint32_t pixel) { return pixel == 0xff804020U; }));
    }
    return result;
  }

  void ExpectCompatible(uint32_t width, uint32_t height, uint32_t layers, uint32_t levels,
                        bool constant = false) {
    const std::vector<uint32_t> synchronous = Upload(width, height, layers, levels, false, constant);
    const std::vector<uint32_t> recorded = Upload(width, height, layers, levels, true, constant);
    EXPECT_EQ(recorded, synchronous);
  }

  void ExpectLayerMipmaps(uint32_t width, uint32_t height, uint32_t layers,
                         VkFormat format, bool checker) {
    uint32_t levels = 1;
    for (uint32_t size = std::max(width, height); size > 1; size /= 2) {
      ++levels;
    }
    const size_t face_pixels = static_cast<size_t>(width) * height;
    std::vector<uint32_t> input(face_pixels * layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
      for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
          const uint32_t red = checker ? ((x + y) % 2) * 255U : layer * 31U;
          input[face_pixels * layer + static_cast<size_t>(y) * width + x] =
              0xff000000U | ((255U - layer * 31U) << 16) | ((layer * 31U) << 8) | red;
        }
      }
    }
    vulkan::VulkanBuffer source(&context_, input.size() * sizeof(uint32_t),
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHostMemory);
    source.MapData([&](void* data) { std::memcpy(data, input.data(), source.Size()); });
    vulkan::VulkanImage image;
    image = vulkan::VulkanImage(
        &context_, width, height, format,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_TILING_OPTIMAL, levels, VK_SAMPLE_COUNT_1_BIT,
        layers == 6 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, layers);
    vulkan::VulkanCommandBuffer command(&context_);
    for (uint32_t round = 0; round < 2; ++round) {
      command.BeginCommandBufferWithReset();
      image.TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format, levels, layers);
      source.CopyToImage(command.buffer(), image, width, height, layers);
      image.GenerateMipmaps(command.buffer());
      SubmitAndWait(command);
      const std::vector<uint32_t> result = ReadImage(image, levels, layers);
      ASSERT_GE(result.size(), input.size());
      EXPECT_TRUE(std::equal(input.begin(), input.end(), result.begin()));
      size_t offset = input.size();
      uint32_t mip_width = width;
      uint32_t mip_height = height;
      for (uint32_t level = 1; level < levels; ++level) {
        mip_width = std::max(mip_width / 2, 1U);
        mip_height = std::max(mip_height / 2, 1U);
        for (uint32_t layer = 0; layer < layers; ++layer) {
          const uint32_t red = checker ? (format == VK_FORMAT_R8G8B8A8_SRGB ? 188U : 128U)
                                       : layer * 31U;
          const uint32_t expected = 0xff000000U | ((255U - layer * 31U) << 16) |
                                    ((layer * 31U) << 8) | red;
          for (size_t pixel = 0; pixel < static_cast<size_t>(mip_width) * mip_height; ++pixel) {
            ASSERT_LT(offset, result.size());
            for (uint32_t shift : {0U, 8U, 16U, 24U}) {
              const int actual_channel = static_cast<int>((result[offset] >> shift) & 255U);
              const int expected_channel = static_cast<int>((expected >> shift) & 255U);
              ASSERT_NEAR(actual_channel, expected_channel, 2)
                  << "level=" << level << " layer=" << layer << " pixel=" << pixel;
            }
            ++offset;
          }
        }
      }
      EXPECT_EQ(offset, result.size());
    }
  }

  vulkan::VulkanContext context_{true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE};
};

TEST_F(VulkanImageRecordingTest, MatchesSynchronousTexturesAndSixDistinctCubeFaces) {
  ExpectCompatible(1, 1, 1, 1);
  ExpectCompatible(7, 3, 1, 1);
  ExpectCompatible(4, 4, 6, 1);
}

TEST_F(VulkanImageRecordingTest, MatchesEveryMipIncludingOddAndOneDimensionalImages) {
  ExpectCompatible(8, 4, 1, 4);
  ExpectCompatible(7, 3, 1, 3);
  ExpectCompatible(1, 8, 1, 4, true);
  ExpectCompatible(1, 1, 1, 1, true);
  ExpectCompatible(4, 4, 6, 3, true);
}

TEST_F(VulkanImageRecordingTest, DefersMultipleUploadsAndAllowsCommandBufferReuse) {
  vulkan::VulkanBuffer source(&context_, 64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHostMemory);
  vulkan::VulkanImage first(&context_, 4, 4, kFormat,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vulkan::VulkanImage second(&context_, 4, 4, kFormat,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vulkan::VulkanCommandBuffer command(&context_);
  for (uint32_t value : {0xff123456U, 0xffabcdefU}) {
    source.MapData([](void* data) { std::memset(data, 0, 64); });
    command.BeginCommandBufferWithReset();
    EXPECT_THROW(first.TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                             VK_IMAGE_LAYOUT_GENERAL, kFormat),
                 std::invalid_argument);
    EXPECT_THROW(source.CopyToImage(command.buffer(), first, 4, 4, 0), std::invalid_argument);
    RecordUpload(command.buffer(), source, first, 1, 1, false);
    RecordUpload(command.buffer(), source, second, 1, 1, false);
    source.MapData([value](void* data) { std::fill_n(static_cast<uint32_t*>(data), 16, value); });
    SubmitAndWait(command);
    EXPECT_EQ(ReadImage(first, 1, 1), std::vector<uint32_t>(16, value));
    EXPECT_EQ(ReadImage(second, 1, 1), std::vector<uint32_t>(16, value));
  }
}

TEST_F(VulkanImageRecordingTest, GeneratesEveryCubeFaceWithLinearAndSrgbFiltering) {
  ExpectLayerMipmaps(8, 8, 6, kFormat, true);
  ExpectLayerMipmaps(8, 8, 6, VK_FORMAT_R8G8B8A8_SRGB, true);
}

TEST_F(VulkanImageRecordingTest, HandlesOddCubeFacesSingleLevelAndNonCubeArrays) {
  ExpectLayerMipmaps(7, 7, 6, VK_FORMAT_R8G8B8A8_SRGB, false);
  ExpectLayerMipmaps(1, 1, 6, kFormat, false);
  ExpectLayerMipmaps(1, 8, 3, kFormat, false);
}

TEST_F(VulkanImageRecordingTest, RejectsUnsupportedMipmapsBeforeRecordingAndPreservesMoveState) {
  vulkan::VulkanCommandBuffer command(&context_);
  command.BeginCommandBuffer();
  vulkan::VulkanImage empty;
  EXPECT_THROW(empty.GenerateMipmaps(command.buffer()), std::invalid_argument);
  vulkan::VulkanImage integer_image(
      &context_, 4, 4, VK_FORMAT_R8G8B8A8_UINT,
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      VK_IMAGE_TILING_OPTIMAL, 3);
  EXPECT_THROW(integer_image.GenerateMipmaps(command.buffer()), std::invalid_argument);
  vulkan::VulkanImage no_source(
      &context_, 4, 4, kFormat, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      VK_IMAGE_TILING_OPTIMAL, 3);
  vulkan::VulkanImage moved;
  moved = std::move(no_source);
  EXPECT_THROW(no_source.GenerateMipmaps(command.buffer()), std::invalid_argument);
  EXPECT_THROW(moved.GenerateMipmaps(command.buffer()), std::invalid_argument);
  EXPECT_THROW(integer_image.GenerateMipmaps(VK_NULL_HANDLE), std::invalid_argument);
  SubmitAndWait(command);
}

}
}
}
