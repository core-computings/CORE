#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanDynamicRendering.h"
#include "VulkanImage.h"
#include "VulkanRender.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkMemoryPropertyFlags kHostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr uint32_t kWidth = 61;
constexpr uint32_t kHeight = 37;
constexpr std::array<uint8_t, 4> kClearPixel = {51, 0, 153, 255};
constexpr std::array<uint8_t, 4> kTrianglePixel = {255, 153, 0, 255};
constexpr std::array<float, 4> kTriangleColor = {1.0f, 0.6f, 0.0f, 1.0f};
constexpr int kResolveTolerance = 1;

class OffscreenTriangle : public vulkan::VulkanRender {
 public:
  OffscreenTriangle(vulkan::VulkanContext* context, VkSampleCountFlagBits samples)
      : vulkan::VulkanRender(context,
                             vulkan::DynamicRenderingInfo{
                                 .color_formats = {kFormat},
                                 .depth_format = VK_FORMAT_UNDEFINED,
                                 .stencil_format = VK_FORMAT_UNDEFINED,
                             },
                             samples),
        color_(context, sizeof(kTriangleColor), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostMemory) {}

  void Init() override {
    vulkan::VulkanRender::Init();
    color_.MapData(
        [](void* data) { std::memcpy(data, kTriangleColor.data(), sizeof(kTriangleColor)); });
    CreateUniformBufferDescriptorSet(0, color_);
    vkUpdateDescriptorSets(context_->logical_device, static_cast<uint32_t>(writes_.size()),
                           writes_.data(), 0, nullptr);
  }

  void Draw(VkCommandBuffer command_buffer, VkExtent2D extent) const {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    const VkRect2D scissor{.offset = {.x = 0, .y = 0}, .extent = extent};
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0,
                            1, &descriptor_set_, 0, nullptr);
    vkCmdDraw(command_buffer, 3, 1, 0, 0);
  }

 protected:
  std::vector<vulkan::BindingInfo> GetBindingInfo() const override {
    return {{.binding = 0,
             .descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             .shader_stage = VK_SHADER_STAGE_FRAGMENT_BIT}};
  }

 private:
  VkCullModeFlags SetCullMode() const override { return VK_CULL_MODE_NONE; }

  const std::vector<uint32_t> LoadVertexShader() const override {
    static const std::vector<uint32_t> shader_code =
#include "OffscreenRender.vert.spv"
        ;
    return shader_code;
  }

  const std::vector<uint32_t> LoadFragmentShader() const override {
    static const std::vector<uint32_t> shader_code =
#include "OffscreenRender.frag.spv"
        ;
    return shader_code;
  }

  std::vector<VkVertexInputBindingDescription> GetVertexBindingDescriptions() const override {
    return {};
  }

  std::vector<VkVertexInputAttributeDescription> GetVertexAttributeDescriptions() const override {
    return {};
  }

  vulkan::VulkanBuffer color_;
};

double EdgeDistanceInPixels(uint32_t x, uint32_t y) {
  const double width = static_cast<double>(kWidth);
  const double height = static_cast<double>(kHeight);
  const double edge =
      (static_cast<double>(x) + 0.5) / width + (static_cast<double>(y) + 0.5) / height - 1.0;
  return edge / std::sqrt(1.0 / (width * width) + 1.0 / (height * height));
}

std::array<uint8_t, 4> PixelAt(const std::vector<uint8_t>& pixels, uint32_t x, uint32_t y) {
  const size_t offset = (static_cast<size_t>(y) * kWidth + x) * 4;
  return {pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]};
}

bool IsNear(const std::array<uint8_t, 4>& pixel, const std::array<uint8_t, 4>& expected) {
  for (size_t channel = 0; channel < pixel.size(); ++channel) {
    if (std::abs(static_cast<int>(pixel[channel]) - static_cast<int>(expected[channel])) >
        kResolveTolerance) {
      return false;
    }
  }
  return true;
}

bool IsBetween(const std::array<uint8_t, 4>& pixel) {
  for (size_t channel = 0; channel < pixel.size(); ++channel) {
    const int low = static_cast<int>(std::min(kClearPixel[channel], kTrianglePixel[channel]));
    const int high = static_cast<int>(std::max(kClearPixel[channel], kTrianglePixel[channel]));
    const int value = static_cast<int>(pixel[channel]);
    if (value < low - kResolveTolerance || value > high + kResolveTolerance) {
      return false;
    }
  }
  return true;
}

class VulkanOffscreenRenderTest : public testing::Test {
 protected:
  void SetUp() override { context_.Init(); }

  vulkan::VulkanImage CreateColorImage(VkImageUsageFlags usage, VkSampleCountFlagBits samples) {
    return vulkan::VulkanImage(&context_, kWidth, kHeight, kFormat, usage,
                               VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               VK_IMAGE_TILING_OPTIMAL, 1, samples);
  }

  std::vector<uint8_t> Render(VkSampleCountFlagBits samples) {
    OffscreenTriangle triangle(&context_, samples);
    triangle.Init();
    vulkan::VulkanImage output;
    output = CreateColorImage(
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_SAMPLE_COUNT_1_BIT);
    std::unique_ptr<vulkan::VulkanImage> multisampled;
    if (samples != VK_SAMPLE_COUNT_1_BIT) {
      multisampled = std::make_unique<vulkan::VulkanImage>(
          &context_, kWidth, kHeight, kFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
          VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          VK_IMAGE_TILING_OPTIMAL, 1, samples);
    }
    vulkan::VulkanBuffer readback(&context_, output.PackedSize(),
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostMemory);
    readback.MapData([&readback](void* data) {
      std::memset(data, 0, static_cast<size_t>(readback.Size()));
    });

    const vulkan::VulkanDynamicRendering dynamic_rendering(&context_);
    const VkExtent2D extent{.width = kWidth, .height = kHeight};
    const VkClearValue clear_value{.color = {.float32 = {0.2f, 0.0f, 0.6f, 1.0f}}};
    vulkan::VulkanCommandBuffer command(&context_);
    command.BeginCommandBuffer();
    output.TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, kFormat);
    if (multisampled != nullptr) {
      multisampled->TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, kFormat);
      dynamic_rendering.BeginDynamicRendering(command.buffer(), multisampled->image_view,
                                              output.image_view, extent, clear_value);
    } else {
      dynamic_rendering.BeginDynamicRendering(command.buffer(), output.image_view, extent,
                                              clear_value);
    }
    triangle.Draw(command.buffer(), extent);
    dynamic_rendering.EndDynamicRendering(command.buffer());
    output.TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, kFormat);
    output.CopyToBuffer(command.buffer(), readback);

    vulkan::VulkanFence fence(&context_);
    fence.Reset();
    command.Submit(fence.fence);
    EXPECT_EQ(vkWaitForFences(context_.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX),
              VK_SUCCESS);
    std::vector<uint8_t> pixels(static_cast<size_t>(readback.Size()));
    readback.MapData([&pixels](void* data) { std::memcpy(pixels.data(), data, pixels.size()); });
    return pixels;
  }

  vulkan::VulkanContext context_{true, vulkan::QueueFamilyType::Graphics, VK_NULL_HANDLE};
};

TEST_F(VulkanOffscreenRenderTest, CreatesGraphicsQueueWithoutSurface) {
  const vulkan::QueueFamilyIndices indices = context_.GetQueueFamilyIndices();
  ASSERT_TRUE(indices.graphics_family.has_value());
  EXPECT_FALSE(indices.compute_family.has_value());
  EXPECT_FALSE(indices.present_family.has_value());
  EXPECT_NE(context_.graphics_queue(), VK_NULL_HANDLE);
  EXPECT_EQ(context_.present_queue(), VK_NULL_HANDLE);
  EXPECT_EQ(context_.queue_family_type(), vulkan::QueueFamilyType::Graphics);
}

TEST_F(VulkanOffscreenRenderTest, RendersAndReadsBackSingleSampleTriangle) {
  const std::vector<uint8_t> pixels = Render(VK_SAMPLE_COUNT_1_BIT);
  ASSERT_EQ(pixels.size(), static_cast<size_t>(kWidth) * kHeight * 4);
  size_t inside = 0;
  size_t outside = 0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const double distance = EdgeDistanceInPixels(x, y);
      const std::array<uint8_t, 4> pixel = PixelAt(pixels, x, y);
      if (distance < -0.01) {
        ASSERT_EQ(pixel, kTrianglePixel) << "x=" << x << " y=" << y;
        ++inside;
      } else if (distance > 0.01) {
        ASSERT_EQ(pixel, kClearPixel) << "x=" << x << " y=" << y;
        ++outside;
      } else {
        EXPECT_TRUE(pixel == kTrianglePixel || pixel == kClearPixel) << "x=" << x << " y=" << y;
      }
    }
  }
  EXPECT_GT(inside, 0U);
  EXPECT_GT(outside, 0U);
}

TEST_F(VulkanOffscreenRenderTest, ResolvesMultisampledTriangleBeforeReadback) {
  const std::vector<uint8_t> pixels = Render(VK_SAMPLE_COUNT_4_BIT);
  ASSERT_EQ(pixels.size(), static_cast<size_t>(kWidth) * kHeight * 4);
  size_t blended = 0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const double distance = EdgeDistanceInPixels(x, y);
      const std::array<uint8_t, 4> pixel = PixelAt(pixels, x, y);
      if (distance <= -0.75) {
        ASSERT_TRUE(IsNear(pixel, kTrianglePixel)) << "x=" << x << " y=" << y;
      } else if (distance >= 0.75) {
        ASSERT_TRUE(IsNear(pixel, kClearPixel)) << "x=" << x << " y=" << y;
      } else {
        ASSERT_TRUE(IsBetween(pixel)) << "x=" << x << " y=" << y;
        if (!IsNear(pixel, kTrianglePixel) && !IsNear(pixel, kClearPixel)) {
          ++blended;
        }
      }
    }
  }
  EXPECT_GT(blended, 0U);
}

TEST_F(VulkanOffscreenRenderTest, RejectsInvalidReadbacksBeforeRecording) {
  vulkan::VulkanCommandBuffer command(&context_);
  vulkan::VulkanImage source = CreateColorImage(
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      VK_SAMPLE_COUNT_1_BIT);
  ASSERT_EQ(source.PackedSize(), static_cast<VkDeviceSize>(kWidth) * kHeight * 4);
  vulkan::VulkanBuffer readback(&context_, source.PackedSize(),
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostMemory);
  vulkan::VulkanBuffer small(&context_, source.PackedSize() - 4,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostMemory);
  vulkan::VulkanBuffer wrong_usage(&context_, source.PackedSize(),
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHostMemory);
  vulkan::VulkanImage no_transfer =
      CreateColorImage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SAMPLE_COUNT_1_BIT);
  vulkan::VulkanImage multisampled =
      CreateColorImage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                       VK_SAMPLE_COUNT_4_BIT);
  vulkan::VulkanImage depth(&context_, kWidth, kHeight, VK_FORMAT_D16_UNORM,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VK_IMAGE_ASPECT_DEPTH_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vulkan::VulkanImage empty;

  EXPECT_THROW(source.CopyToBuffer(VK_NULL_HANDLE, readback), std::invalid_argument);
  EXPECT_THROW(empty.CopyToBuffer(command.buffer(), readback), std::invalid_argument);
  EXPECT_THROW(source.CopyToBuffer(command.buffer(), small), std::out_of_range);
  EXPECT_THROW(source.CopyToBuffer(command.buffer(), wrong_usage), std::invalid_argument);
  EXPECT_THROW(no_transfer.CopyToBuffer(command.buffer(), readback), std::invalid_argument);
  EXPECT_THROW(multisampled.CopyToBuffer(command.buffer(), readback), std::invalid_argument);
  EXPECT_THROW(empty.PackedSize(), std::invalid_argument);
  EXPECT_THROW(depth.PackedSize(), std::invalid_argument);
  EXPECT_THROW(depth.CopyToBuffer(command.buffer(), readback), std::invalid_argument);
}

}
}
}
