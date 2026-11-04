#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "VulkanCompute.h"
#include "VulkanSampler.h"

namespace core::test {
namespace {

using Color = std::array<float, 4>;
constexpr VkMemoryPropertyFlags kHostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr std::array<uint32_t, 3> kWidths = {4, 2, 1};
constexpr std::array<size_t, 3> kOffsets = {0, 64, 80};

class SamplerProbe : public vulkan::VulkanCompute {
 public:
  explicit SamplerProbe(vulkan::VulkanContext* context)
      : VulkanCompute(context) {}

  void Bind(const vulkan::VulkanImage& image,
            const vulkan::VulkanSampler& sampler,
            const vulkan::VulkanBuffer& queries,
            const vulkan::VulkanBuffer& output) {
    CreateCombinedImageSamplerDescriptorSet(0, image.image_view,
                                            sampler.sampler);
    CreateStorageBufferDescriptorSet(1, queries);
    CreateStorageBufferDescriptorSet(2, output);
    vkUpdateDescriptorSets(context_->logical_device,
                           static_cast<uint32_t>(writes_.size()),
                           writes_.data(), 0, nullptr);
  }

  void Run(VkCommandBuffer command, uint32_t count) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &descriptor_set_, 0,
                            nullptr);
    vkCmdDispatch(command, count, 1, 1);
  }

 private:
  std::vector<vulkan::BindingInfo> GetBindingInfo() const override {
    return {{.binding = 0,
             .descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1,
             .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2,
             .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT}};
  }

  const std::vector<uint32_t>& LoadShaderCode() const override {
    static const std::vector<uint32_t> code =
#include "SamplerProbe.comp.spv"
        ;
    return code;
  }
};

class VulkanSamplerTest : public testing::Test {
 protected:
  void SetUp() override {
    context_.Init();
    for (size_t level = 0; level < kWidths.size(); ++level) {
      const uint32_t width = kWidths[level];
      for (uint32_t y = 0; y < width; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
          const size_t offset = kOffsets[level] + 4 * (y * width + x);
          pixels_[offset] = static_cast<uint8_t>(32 + 48 * x);
          pixels_[offset + 1] = static_cast<uint8_t>(16 + 64 * y);
          pixels_[offset + 2] = static_cast<uint8_t>(40 + 80 * level);
          pixels_[offset + 3] = 255;
        }
      }
    }
    image_ = vulkan::VulkanImage(
        &context_, 4, 4, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_TILING_OPTIMAL, 3);
    vulkan::VulkanBuffer staging(&context_, pixels_.size(),
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHostMemory);
    staging.MapData([this](void* bytes) {
      std::memcpy(bytes, pixels_.data(), pixels_.size());
    });
    const vulkan::VulkanCommandBuffer command =
        vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context_);
    image_.TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_FORMAT_R8G8B8A8_UNORM, 3);
    std::array<VkBufferImageCopy, 3> regions{};
    for (size_t level = 0; level < regions.size(); ++level) {
      regions[level].bufferOffset = kOffsets[level];
      regions[level].imageSubresource = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .mipLevel = static_cast<uint32_t>(level),
          .baseArrayLayer = 0,
          .layerCount = 1};
      regions[level].imageExtent = {
          .width = kWidths[level], .height = kWidths[level], .depth = 1};
    }
    vkCmdCopyBufferToImage(command.buffer(), staging.buffer, image_.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(regions.size()),
                           regions.data());
    VkImageMemoryBarrier image_barrier{};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = image_.image;
    image_barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .baseMipLevel = 0,
                                      .levelCount = 3,
                                      .baseArrayLayer = 0,
                                      .layerCount = 1};
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &image_barrier);
    vulkan::VulkanFence fence(&context_);
    fence.Reset();
    command.Submit(fence.fence);
    VK_CHECK(vkWaitForFences(context_.logical_device, 1, &fence.fence, VK_TRUE,
                             UINT64_MAX));
  }

  Color Texel(int x, int y, size_t level,
              const vulkan::SamplerOptions& options) const {
    const int width = static_cast<int>(kWidths[level]);
    const auto address = [width](int coordinate, VkSamplerAddressMode mode) {
      if (mode == VK_SAMPLER_ADDRESS_MODE_REPEAT) {
        return ((coordinate % width) + width) % width;
      }
      if (mode == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT) {
        const int mirrored =
            ((coordinate % (2 * width)) + (2 * width)) % (2 * width);
        return mirrored < width ? mirrored : (2 * width) - mirrored - 1;
      }
      if (mode == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) {
        return std::clamp(coordinate, 0, width - 1);
      }
      return coordinate;
    };
    x = address(x, options.address_mode_u);
    y = address(y, options.address_mode_v);
    if (x < 0 || y < 0 || x >= width || y >= width) {
      if (options.border_color == VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK) {
        return {0.0f, 0.0f, 0.0f, 0.0f};
      }
      if (options.border_color == VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE) {
        return {1.0f, 1.0f, 1.0f, 1.0f};
      }
      return {0.0f, 0.0f, 0.0f, 1.0f};
    }
    const size_t offset =
        kOffsets[level] + 4 * static_cast<size_t>(y * width + x);
    Color color{};
    for (size_t channel = 0; channel < color.size(); ++channel) {
      color[channel] = static_cast<float>(pixels_[offset + channel]) / 255.0f;
    }
    return color;
  }

  Color SampleLevel(const Color& query, size_t level, VkFilter filter,
                    const vulkan::SamplerOptions& options) const {
    const float width = static_cast<float>(kWidths[level]);
    if (filter == VK_FILTER_NEAREST) {
      return Texel(static_cast<int>(std::floor(query[0] * width)),
                   static_cast<int>(std::floor(query[1] * width)), level,
                   options);
    }
    const float x = query[0] * width - 0.5f;
    const float y = query[1] * width - 0.5f;
    const int left = static_cast<int>(std::floor(x));
    const int top = static_cast<int>(std::floor(y));
    Color result{};
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        const float weight_x =
            dx == 0 ? 1.0f - (x - std::floor(x)) : x - std::floor(x);
        const float weight_y =
            dy == 0 ? 1.0f - (y - std::floor(y)) : y - std::floor(y);
        const Color texel = Texel(left + dx, top + dy, level, options);
        for (size_t channel = 0; channel < result.size(); ++channel) {
          result[channel] += weight_x * weight_y * texel[channel];
        }
      }
    }
    return result;
  }

  Color Reference(const Color& query,
                  const vulkan::SamplerOptions& options) const {
    const float lod = std::clamp(query[2], options.min_lod, options.max_lod);
    const VkFilter filter =
        lod <= 0.0f ? options.mag_filter : options.min_filter;
    const float level = std::clamp(lod, 0.0f, 2.0f);
    if (options.mipmap_mode == VK_SAMPLER_MIPMAP_MODE_NEAREST) {
      return SampleLevel(query, static_cast<size_t>(std::floor(level + 0.5f)),
                         filter, options);
    }
    const size_t lower = static_cast<size_t>(std::floor(level));
    const size_t upper = std::min(lower + 1, size_t{2});
    const Color first = SampleLevel(query, lower, filter, options);
    const Color second = SampleLevel(query, upper, filter, options);
    Color result{};
    for (size_t channel = 0; channel < result.size(); ++channel) {
      result[channel] = first[channel] + (level - std::floor(level)) *
                                             (second[channel] - first[channel]);
    }
    return result;
  }

  void Check(const vulkan::SamplerOptions& options, bool legacy = false) {
    const std::array<float, 7> lods = {-1.0f, 0.0f,  0.25f, 0.75f,
                                       1.25f, 1.75f, 3.0f};
    const std::array<std::array<float, 2>, 7> coordinates = {
        {{0.125f, 0.375f},
         {0.375f, 0.625f},
         {0.5f, 0.5f},
         {-0.375f, 0.625f},
         {1.375f, -0.375f},
         {0.9375f, 1.0625f},
         {-0.0625f, -0.0625f}}};
    std::vector<Color> queries;
    for (float lod : lods) {
      for (const std::array<float, 2>& coordinate : coordinates) {
        queries.push_back({coordinate[0], coordinate[1], lod, 0.0f});
      }
    }
    const VkDeviceSize bytes = queries.size() * sizeof(Color);
    vulkan::VulkanBuffer input(&context_, bytes,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
    vulkan::VulkanBuffer output(
        &context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory);
    input.MapData([&queries, bytes](void* data) {
      std::memcpy(data, queries.data(), bytes);
    });
    const std::unique_ptr<vulkan::VulkanSampler> sampler =
        legacy ? std::make_unique<vulkan::VulkanSampler>(&context_)
               : std::make_unique<vulkan::VulkanSampler>(&context_, options);
    SamplerProbe probe(&context_);
    probe.Init();
    probe.Bind(image_, *sampler, input, output);
    vulkan::VulkanCommandBuffer command(&context_);
    vulkan::VulkanFence fence(&context_);
    for (int round = 0; round < 2; ++round) {
      output.MapData([bytes](void* data) { std::memset(data, 0xff, bytes); });
      command.BeginCommandBufferWithReset();
      probe.Run(command.buffer(), static_cast<uint32_t>(queries.size()));
      VkMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(
          command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
      fence.Reset();
      command.Submit(fence.fence);
      VK_CHECK(vkWaitForFences(context_.logical_device, 1, &fence.fence,
                               VK_TRUE, UINT64_MAX));
      output.MapData([this, &queries, &options](void* data) {
        const Color* actual = static_cast<const Color*>(data);
        for (size_t index = 0; index < queries.size(); ++index) {
          const Color expected = Reference(queries[index], options);
          for (size_t channel = 0; channel < expected.size(); ++channel) {
            EXPECT_NEAR(actual[index][channel], expected[channel], 0.006f)
                << "query " << index << " channel " << channel;
          }
        }
      });
    }
  }

  vulkan::VulkanContext context_{true};
  vulkan::VulkanImage image_;
  std::array<uint8_t, 84> pixels_{};
};

TEST_F(VulkanSamplerTest, PreservesDefaultSampling) {
  Check({}, true);
  Check({});
}

TEST_F(VulkanSamplerTest, FiltersAndAddressesTextureCoordinates) {
  const std::array<VkSamplerAddressMode, 4> modes = {
      VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER};
  for (VkSamplerAddressMode mode : modes) {
    SCOPED_TRACE(mode);
    Check({.mag_filter = VK_FILTER_NEAREST,
           .min_filter = VK_FILTER_NEAREST,
           .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
           .address_mode_u = mode,
           .address_mode_v = mode,
           .address_mode_w = mode});
    Check({.address_mode_u = mode,
           .address_mode_v = mode,
           .address_mode_w = mode});
  }
  Check({.mag_filter = VK_FILTER_NEAREST});
  Check({.min_filter = VK_FILTER_NEAREST,
         .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .address_mode_v = VK_SAMPLER_ADDRESS_MODE_REPEAT});
  Check({.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST});
  Check({.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
         .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
         .border_color = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK});
  Check({.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
         .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
         .border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE});
}

TEST_F(VulkanSamplerTest, ClampsMipLevels) {
  Check({.min_lod = 0.0f, .max_lod = 0.0f});
  Check({.min_lod = 1.0f, .max_lod = 1.0f});
  Check({.min_lod = 0.25f, .max_lod = 1.75f});
  Check({.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
         .min_lod = 1.25f,
         .max_lod = 2.0f});
}

TEST_F(VulkanSamplerTest, RejectsUnsupportedOptionsAndRecovers) {
  const std::vector<vulkan::SamplerOptions> invalid = {
      {.mag_filter = VK_FILTER_CUBIC_EXT},
      {.min_filter = VK_FILTER_CUBIC_EXT},
      {.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_MAX_ENUM},
      {.address_mode_u = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE},
      {.address_mode_v = VK_SAMPLER_ADDRESS_MODE_MAX_ENUM},
      {.address_mode_w = VK_SAMPLER_ADDRESS_MODE_MAX_ENUM},
      {.border_color = VK_BORDER_COLOR_MAX_ENUM},
      {.min_lod = -1.0f},
      {.min_lod = 2.0f, .max_lod = 1.0f},
      {.min_lod = std::numeric_limits<float>::quiet_NaN()},
      {.max_lod = std::numeric_limits<float>::quiet_NaN()},
      {.min_lod = std::numeric_limits<float>::infinity()},
      {.max_lod = std::numeric_limits<float>::infinity()}};
  for (const vulkan::SamplerOptions& options : invalid) {
    EXPECT_THROW(vulkan::VulkanSampler(nullptr, options),
                 std::invalid_argument);
  }
  Check({.mag_filter = VK_FILTER_NEAREST, .max_lod = 0.0f});
}

}
}
